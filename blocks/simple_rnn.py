from __future__ import print_function
import logging
import pprint
import math
import numpy
import os
import operator

import theano
from six.moves import input
from picklable_itertools.extras import equizip
from theano import tensor

from blocks.bricks import Tanh, Initializable, Linear, Logistic, Softmax
from blocks.bricks.cost import SquaredError
from blocks.bricks.base import application
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import SimpleRecurrent, Bidirectional
from blocks.bricks.attention import SequenceContentAttention
from blocks.bricks.parallel import Fork
from blocks.bricks.sequence_generators import (
    SequenceGenerator, Readout, SoftmaxEmitter, LookupFeedback)
from blocks.config import config
from blocks.graph import ComputationGraph
from fuel.transformers import Mapping, Batch, Padding, Filter
from fuel.datasets import OneBillionWord, TextFile
from fuel.schemes import ConstantScheme
from blocks.serialization import load_parameter_values
from blocks.algorithms import (GradientDescent, Scale,
                               StepClipping, CompositeRule, Adam)
from blocks.initialization import Orthogonal, IsotropicGaussian, Constant
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Printing, Timing
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.filter import VariableFilter
from blocks.utils import named_copy, dict_union
from ecog_dataset import ECoG

from blocks.search import BeamSearch

from fuel.streams import DataStream

config.recursion_limit = 100000
floatX = theano.config.floatX
logger = logging.getLogger(__name__)

n_epochs = 25
X_dim = 85
h_dim = 85
y_dim = 57

print('Building model ...')

X = tensor.tensor3('X', dtype=floatX)
y = tensor.matrix('y', dtype=floatX)
y_time = y.dimshuffle('x', 0, 1)


X_to_h = Linear(name='X_to_h',
        input_dim=X_dim,
        output_dim=h_dim)
X_transform = X_to_h.apply(X)
rec = SimpleRecurrent(activation=Tanh(),
        dim=h_dim,
        name='rec')
h = rec.apply(X_transform)
h_to_y = Linear(name='h_to_y',
        input_dim=h_dim,
        output_dim=y_dim)
y_hat = h_to_y.apply(h)
y_hat = Logistic().apply(y_hat)
y_hat.name = 'y_hat'

cost = SquaredError().apply(y_time, y_hat)
cost.name = 'Cross Entropy'

for brick in (rec, X_to_h, h_to_y):
    brick.weights_init = IsotropicGaussian(.01)
    brick.biases_init = Constant(0.)
    brick.initialize()

print('Bulding training process...')
algorithm = GradientDescent(cost=cost,
        params=ComputationGraph(cost).parameters,
        step_rule=CompositeRule([StepClipping(10.0),
        Adam()]))

monitor_cost = TrainingDataMonitoring([cost],
        prefix="train",
        after_epoch=True)

train_dataset = ECoG(('train',), 0)

stream = DataStream(train_dataset)

model = Model(cost)

main_loop = MainLoop(data_stream=stream, algorithm=algorithm,
        extensions=[monitor_cost,
            FinishAfter(after_n_epochs=n_epochs),
            Printing()],
        model=model)

print('Starting training ...')
main_loop.run()
