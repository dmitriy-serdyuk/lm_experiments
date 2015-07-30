from __future__ import print_function

import logging
import pprint
import sys
import string

from six.moves import cPickle
import numpy
import theano
from theano import tensor

from blocks.bricks import Tanh
from blocks.bricks.recurrent import GatedRecurrent, LSTM
from blocks.bricks.sequence_generators import (
    SequenceGenerator, Readout, SoftmaxEmitter, LookupFeedback)
from blocks.graph import ComputationGraph
from blocks.algorithms import GradientDescent, Scale, Adam
from blocks.initialization import Orthogonal, IsotropicGaussian, Constant
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.serialization import load, load_parameter_values
from blocks.main_loop import MainLoop
from blocks.select import Selector
from fuel.datasets import TextFile
from fuel.streams import DataStream
from fuel.schemes import ConstantScheme
from fuel.transformers import Batch, Padding, Mapping


sys.setrecursionlimit(10000)
floatX = theano.config.floatX
logger = logging.getLogger(__name__)


def _transpose(data):
    return data[0].T, data[1].T


def _truncate(data):
    return data[0][:150],


def main(mode, save_path, steps, num_batches):
    with open('/Tmp/serdyuk/data/wsj_text/wordlist.pkl') as f:
        dictionary = cPickle.load(f)
    chars = list(string.ascii_uppercase) + list(range(10)) + [' ', '.', ',', '<UNK>']
    dictionary = {char: i for i, char in enumerate(chars)}
    dataset = TextFile(['/Tmp/serdyuk/data/wsj_text/train_nounk'], 
                       dictionary, bos_token=None, eos_token=None, level='character')
    vocab_size = len(dictionary)
    logger.info('Dictionary size: {}'.format(vocab_size))

    if mode == "train":
        # Experiment configuration
        rng = numpy.random.RandomState(1123)
        batch_size = 100
        dim = 700
        feedback_dim = 700

        # Build the bricks and initialize them

        transition = LSTM(name="transition", dim=dim,
                activation=Tanh())
        generator = SequenceGenerator(
            Readout(readout_dim=vocab_size, source_names=["states"],
                    emitter=SoftmaxEmitter(name="emitter"),
                    feedback_brick=LookupFeedback(
                        vocab_size, feedback_dim, name='feedback'),
                    name="readout"),
            transition,
            weights_init=IsotropicGaussian(0.01), biases_init=Constant(0),
            name="generator")
        generator.push_initialization_config()
        transition.weights_init = Orthogonal()
        generator.initialize()

        # Give an idea of what's going on.
        logger.info("Parameters:\n" +
                    pprint.pformat(
                        [(key, value.get_value().shape) for key, value
                         in Selector(generator).get_parameters().items()],
                        width=120))
        #logger.info("Markov chain entropy: {}".format(
        #    MarkovChainDataset.entropy))
        #logger.info("Expected min error: {}".format(
        #    -MarkovChainDataset.entropy * seq_len))

        # Build the cost computation graph.
        features = tensor.lmatrix('features')
        features_mask = tensor.matrix('features_mask')
        batch_cost = generator.cost_matrix(features[:, :],
                                  mask=features_mask).sum()
        cost = aggregation.mean(
            batch_cost,
            features.shape[1])
        cost.name = "sequence_log_likelihood"
        char_cost = aggregation.mean(
                batch_cost, features_mask.sum())
        char_cost.name = 'character_log_likelihood'
        ppl = 2 ** char_cost
        ppl.name = 'ppl'
        length = features.shape[0]
        length.name = 'length'

        algorithm = GradientDescent(
            cost=cost,
            parameters=list(Selector(generator).get_parameters().values()),
            step_rule=Adam(0.0001))
        data_stream = dataset.get_example_stream()
        data_stream = Mapping(data_stream, _truncate)
        data_stream = Batch(data_stream, iteration_scheme=ConstantScheme(batch_size))
        data_stream = Padding(data_stream)
        data_stream = Mapping(data_stream, _transpose)
        params = load_parameter_values(save_path)
        model = Model(cost)
        #model.set_parameter_values(params)
        ft = features[:6, 0]
        ft.name = 'feature_example'
        main_loop = MainLoop(
            algorithm=algorithm,
            data_stream=data_stream,
            model=model,
            extensions=[FinishAfter(after_n_batches=num_batches),
                TrainingDataMonitoring([cost, ppl, char_cost, length, ft], prefix="this_step",
                                               after_batch=True),
                        TrainingDataMonitoring([cost, ppl, char_cost, length], prefix="average",
                                               every_n_batches=100),
                        Checkpoint(save_path, every_n_batches=500),
                        Printing(every_n_batches=100)])
        main_loop.run()
    elif mode == "sample":
        main_loop = load(open(save_path, "rb"))
        generator = main_loop.model.get_top_bricks()[-1]

        sample = ComputationGraph(generator.generate(
            n_steps=steps, batch_size=1, iterate=True)).get_theano_function()

        states, outputs, costs = [data[:, 0] for data in sample()]
        rev_dict = {v: k for k, v in dictionary.iteritems()}
        print("".join([rev_dict[s] for s in outputs]))

        numpy.set_printoptions(precision=3, suppress=True)
        print("Generation cost:\n{}".format(costs.sum()))

        freqs = numpy.bincount(outputs).astype(floatX)
        freqs /= freqs.sum()
        print("Frequencies:\n {} vs {}".format(freqs,
                                               MarkovChainDataset.equilibrium))

        trans_freqs = numpy.zeros((vocab_size, vocab_size), dtype=floatX)
        for a, b in zip(outputs, outputs[1:]):
            trans_freqs[a, b] += 1
        trans_freqs /= trans_freqs.sum(axis=1)[:, None]
        print("Transition frequencies:\n{}\nvs\n{}".format(
            trans_freqs, MarkovChainDataset.trans_prob))
    else:
        assert False

if __name__ == '__main__':
    main('train', './save.zip', 10, 9000000)
