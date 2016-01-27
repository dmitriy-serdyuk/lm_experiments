from __future__ import print_function

import argparse
import logging
import pprint
import os
import sys
import string

import numpy
import theano
from theano import tensor

from picklable_itertools.extras import equizip

from blocks.bricks import Tanh
from blocks.bricks.recurrent import GatedRecurrent
from blocks.bricks.sequence_generators import (
    SequenceGenerator, Readout, SoftmaxEmitter, LookupFeedback)
from blocks.graph import ComputationGraph
from blocks.algorithms import GradientDescent, Adam, CompositeRule, StepClipping, AdaDelta, Momentum, Restrict, VariableClipping
from blocks.initialization import Orthogonal, IsotropicGaussian, Constant
from blocks.filter import VariableFilter
from blocks.model import Model
from blocks.roles import WEIGHT
from blocks.initialization import Uniform
from blocks.monitoring import aggregation
from blocks.extensions import Printing, Timing
from blocks.extensions.monitoring import (TrainingDataMonitoring,
                                          DataStreamMonitoring)
from blocks.extensions.predicates import OnLogRecord
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.training import TrackTheBest
from blocks_extras.extensions.plot import Plot
from blocks.serialization import load, load_parameter_values, continue_training
from blocks.main_loop import MainLoop
from blocks.select import Selector
from blocks.search import BeamSearch
from fuel.datasets import TextFile
from fuel.schemes import ConstantScheme, ShuffledScheme
from fuel.transformers import Batch, Padding, Mapping, Unpack


sys.setrecursionlimit(10000)
floatX = theano.config.floatX
logger = logging.getLogger(__name__)


def _transpose(data):
    return data[0].T, data[1].T


def _truncate(data):
    max_length = 100
    length = len(data[0])
    if length <= max_length:
        return data
    ind = numpy.random.randint(length - max_length)
    return data[0][ind: (ind + max_length)],


def main(mode, save_path, steps, num_batches, load_params):
    chars = (list(string.ascii_uppercase) + list(range(10)) +
             [' ', '.', ',', '\'', '"', '!', '?', '<UNK>'])
    char_to_ind = {char: i for i, char in enumerate(chars)}
    ind_to_char = {v: k for k, v in char_to_ind.iteritems()}

    train_dataset = TextFile(['/Tmp/serdyuk/data/wsj_text_train'],
                             char_to_ind, bos_token=None, eos_token=None,
                             level='character')
    valid_dataset = TextFile(['/Tmp/serdyuk/data/wsj_text_valid'],
                             char_to_ind, bos_token=None, eos_token=None,
                             level='character')

    vocab_size = len(char_to_ind)
    logger.info('Dictionary size: {}'.format(vocab_size))
    if mode == 'continue':
        continue_training(save_path)
        return
    elif mode == "sample":
        main_loop = load(open(save_path, "rb"))
        generator = main_loop.model.get_top_bricks()[-1]

        sample = ComputationGraph(generator.generate(
            n_steps=steps, batch_size=1, iterate=True)).get_theano_function()

        states, outputs, costs = [data[:, 0] for data in sample()]
        print("".join([ind_to_char[s] for s in outputs]))

        numpy.set_printoptions(precision=3, suppress=True)
        print("Generation cost:\n{}".format(costs.sum()))

        freqs = numpy.bincount(outputs).astype(floatX)
        freqs /= freqs.sum()

        trans_freqs = numpy.zeros((vocab_size, vocab_size), dtype=floatX)
        for a, b in zip(outputs, outputs[1:]):
            trans_freqs[a, b] += 1
        trans_freqs /= trans_freqs.sum(axis=1)[:, None]
        return

    # Experiment configuration
    batch_size = 20
    dim = 650
    feedback_dim = 700

    valid_stream = valid_dataset.get_example_stream()
    valid_stream = Batch(valid_stream,
                         iteration_scheme=ConstantScheme(batch_size))
    valid_stream = Padding(valid_stream)
    valid_stream = Mapping(valid_stream, _transpose)

    # Build the bricks and initialize them

    transition = GatedRecurrent(name="transition", dim=dim,
                                activation=Tanh())
    generator = SequenceGenerator(
        Readout(readout_dim=vocab_size, source_names=transition.apply.states,
                emitter=SoftmaxEmitter(name="emitter"),
                feedback_brick=LookupFeedback(
                    vocab_size, feedback_dim, name='feedback'),
                name="readout"),
        transition,
        weights_init=Uniform(std=0.04), biases_init=Constant(0),
        name="generator")
    generator.push_initialization_config()
    transition.weights_init = Orthogonal()
    transition.push_initialization_config()
    generator.initialize()

    # Build the cost computation graph.
    features = tensor.lmatrix('features')
    features_mask = tensor.matrix('features_mask')
    cost_matrix = generator.cost_matrix(
        features, mask=features_mask)
    batch_cost = cost_matrix.sum()
    cost = aggregation.mean(
        batch_cost,
        features.shape[1])
    cost.name = "sequence_log_likelihood"
    char_cost = aggregation.mean(
        batch_cost, features_mask.sum())
    char_cost.name = 'character_log_likelihood'
    ppl = 2 ** (cost / numpy.log(2))
    ppl.name = 'ppl'
    bits_per_char = char_cost / tensor.log(2)
    bits_per_char.name = 'bits_per_char'
    length = features.shape[0]
    length.name = 'length'

    model = Model(batch_cost)
    if load_params:
        params = load_parameter_values(save_path)
        model.set_parameter_values(params)

    if mode == "train":
        # Give an idea of what's going on.
        logger.info("Parameters:\n" +
                    pprint.pformat(
                        [(key, value.get_value().shape) for key, value
                         in Selector(generator).get_parameters().items()],
                        width=120))

        train_stream = train_dataset.get_example_stream()
        train_stream = Mapping(train_stream, _truncate)
        train_stream = Batch(train_stream,
                             iteration_scheme=ConstantScheme(batch_size))
        train_stream = Padding(train_stream)
        train_stream = Mapping(train_stream, _transpose)

        parameters = model.get_parameter_dict()
        maxnorm_subjects = VariableFilter(roles=[WEIGHT])(parameters.values())
        algorithm = GradientDescent(
            cost=batch_cost,
            parameters=parameters.values(),
            step_rule=CompositeRule([StepClipping(1000.), 
                AdaDelta(epsilon=1e-8) #, Restrict(VariableClipping(1.0, axis=0), maxnorm_subjects)
                                     ]))
        ft = features[:6, 0]
        ft.name = 'feature_example'

        observables = [cost, ppl, char_cost, length, bits_per_char]
        for name, param in parameters.items():
            num_elements = numpy.product(param.get_value().shape)
            norm = param.norm(2) / num_elements ** 0.5
            grad_norm = algorithm.gradients[param].norm(2) / num_elements ** 0.5
            step_norm = algorithm.steps[param].norm(2) / num_elements ** 0.5
            stats = tensor.stack(norm, grad_norm, step_norm, step_norm / grad_norm)
            stats.name = name + '_stats'
            observables.append(stats)
        track_the_best_bpc = TrackTheBest('valid_bits_per_char')
        root_path, extension = os.path.splitext(save_path)

        this_step_monitoring = TrainingDataMonitoring(
            observables + [ft], prefix="this_step", after_batch=True)
        average_monitoring = TrainingDataMonitoring(
            observables + [algorithm.total_step_norm,
                           algorithm.total_gradient_norm], 
            prefix="average",
            every_n_batches=10)
        valid_monitoring = DataStreamMonitoring(
            observables, prefix="valid",
            every_n_batches=1500, before_training=False,
            data_stream=valid_stream)
        main_loop = MainLoop(
            algorithm=algorithm,
            data_stream=train_stream,
            model=model,
            extensions=[
                this_step_monitoring,
                average_monitoring,
                valid_monitoring,
                track_the_best_bpc,
                Checkpoint(save_path, ),
                Checkpoint(save_path,
                           every_n_batches=500,
                           save_separately=["model", "log"],
                           use_cpickle=True)
                    .add_condition(
                    ['after_epoch'],
                    OnLogRecord(track_the_best_bpc.notification_name),
                    (root_path + "_best" + extension,)),
                Timing(after_batch=True),
                Printing(every_n_batches=10),
                Plot(root_path,
                     [[average_monitoring.record_name(cost),
                       valid_monitoring.record_name(cost)],
                      [average_monitoring.record_name(algorithm.total_step_norm)],
                      [average_monitoring.record_name(algorithm.total_gradient_norm)],
                      [average_monitoring.record_name(ppl),
                       valid_monitoring.record_name(ppl)],
                      [average_monitoring.record_name(char_cost),
                       valid_monitoring.record_name(char_cost)],
                      [average_monitoring.record_name(bits_per_char),
                       valid_monitoring.record_name(bits_per_char)]],
                     every_n_batches=10)
            ])
        main_loop.run()

    elif mode == 'evaluate':
        with open('/data/lisatmp3/serdyuk/wsj_lms/lms/wsj_trigram_with_initial_eos/words.txt') as f:
            words = [line.split()[0] for line in f.readlines()]
            words = [[char_to_ind[c] if c in char_to_ind else char_to_ind['<UNK>'] for c in w] for w in words]
        max_word_length = max([len(w) for w in words])
        #compute_cost = theano.function([features, features_mask], cost_matrix.sum(axis=0))

        states, sample, costs = generator.generate(
            n_steps=steps, iterate=True)
        beam_search = BeamSearch(len(words), sample)
        beam_search.compile()

        total_word_cost = 0
        num_words = 0
        examples = numpy.zeros((max_word_length, len(words)),
                               dtype='int64')
        mask = numpy.zeros((max_word_length, len(words)),
                           dtype=floatX)

        for i, word in enumerate(words):
            examples[:len(word), i] = word
            mask[:len(word), i] = 1.
        for batch in valid_stream.get_epoch_iterator():
            for example, mask in equizip(batch[0].T, batch[1].T):
                example = example[:(mask.sum())]
                spc_inds = list(numpy.where(example == char_to_ind[" "])[0])
                for i, j in equizip([0] + spc_inds, spc_inds + [-1]):
                    if i > 0:
                        cost_without = compute_cost(example[:i, None], 
                                                    numpy.ones_like(example[:i, None], dtype=floatX))
                    else:
                        cost_without = 0
                    cost_with = compute_cost(example[:j, None], 
                                                numpy.ones_like(example[:j, None], dtype=floatX))
                    costs = []
                    import ipdb;ipdb.set_trace()
                    costs = numpy.exp(-compute_cost(
                        examples, mask))
                    word_prob = numpy.exp(-(cost_with - cost_without))
                    total_word_cost += numpy.log(word_prob / numpy.sum(costs))
                    num_words += 1
                    print(word_prob)

        print("Word-level perplexity")
        print(total_word_cost / num_words)
    else:
        assert False


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        "Case study of Language Models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "mode", choices=["train", "sample", "continue", "evaluate"],
        help="The mode to run. Use `train` to train a new model"
             " and `sample` to sample a sequence generated by an"
             " existing one.")
    parser.add_argument(
        "save_path", default="chain",
        help="The path to save the training process.")
    parser.add_argument(
        "--steps", type=int, default=1000,
        help="Number of steps to samples.")
    parser.add_argument(
        "--num-batches", default=10000, type=int,
        help="Train on this many batches.")
    parser.add_argument(
        "--load-params", action='store_true', default=False,
        help="Load parameters.")
    args = parser.parse_args()
    main(**vars(args))
