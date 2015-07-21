from __future__ import print_function
import logging
import math
import os
import numpy as np

import theano
from theano import tensor

from blocks.bricks import (Tanh, Initializable, Linear,
                           Logistic, Softmax, Rectifier, Identify)
from blocks.bricks.cost import SquaredError, CategoricalCrossEntropy, MisclassificationRate
from blocks.bricks.base import application
from blocks.bricks.recurrent import SimpleRecurrent
from blocks.bricks.sequence_generators import (
    SequenceGenerator, Readout, SoftmaxEmitter, LookupFeedback)
from blocks.config import config
from blocks.graph import ComputationGraph
from fuel.transformers import Mapping, Batch, Padding, Filter
from blocks.serialization import load_parameter_values
from blocks.algorithms import (GradientDescent, Scale,
                               StepClipping, CompositeRule, Adam)
from blocks.initialization import Orthogonal, IsotropicGaussian, Constant
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Printing, Timing
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import (TrainingDataMonitoring,
        DataStreamMonitoring)
from blocks.extensions.stopping import FinishIfNoImprovementAfter
from blocks.extensions.training import TrackTheBest
from blocks.main_loop import MainLoop
from blocks.filter import VariableFilter
from blocks.utils import named_copy, dict_union

from recurrent import SimpleContinuousRecurrent

config.recursion_limit = 100000
floatX = theano.config.floatX
logger = logging.getLogger(__name__)

x_dim = 85
y_dim = 57

dict_to_act = {'tanh': Tanh,
               'logistic': Logistic,
               'rectifier': Rectifier}
dict_to_init = {'isotropic': IsotropicGaussian,

def train_model(data, kwargs):

    print('Bulding model...')
    x = tensor.tensor3('x', dtype=floatX)
    y = tensor.tensor3('y', dtype=floatX)


    x_to_h = Linear(name='x_to_h',
            input_dim=x_dim,
            output_dim=kwargs['h_dim'])
    x_transform = x_to_h.apply(x.dimshuffle(1, 0, 2))
    rec = SimpleRecurrent(activation=dict_to_act[kwargs['act']](),
            dim=kwargs['h_dim'],
            name='rec')
    h = rec.apply(x_transform)
    h_to_y = Linear(name='h_to_y',
            input_dim=kwargs['h_dim'],
            output_dim=y_dim)
    y_hat = h_to_y.apply(h)
    y_hat = Softmax().apply(y_hat.mean(0))
    y_hat.name = 'y_hat'

    print('Defining Cost...')
    cost = CategoricalCrossEntropy().apply(y.mean(1), y_hat)
    cost.name = 'Squared Error'
    misclassification = MisclassificationRate().apply(y.mean(1).argmax(1), y_hat)
    misclassification.name = 'misclass'

    print('Initializing Parameters...')
    for brick in (rec, x_to_h, h_to_y):
        brick.weights_init = dict_to_init[kwargs['init']](kwargs['init_scale'])
        brick.biases_init = Constant(0.)
        brick.initialize()

    print('Bulding training process...')
    algorithm = GradientDescent(cost=cost,
            params=ComputationGraph(cost).parameters,
            step_rule=CompositeRule([StepClipping(10.0),
            Adam()]))

    extensions = []

    for sp, st in zip(splits, streams):
        extensions.append(DataStreamMonitoring([cost, misclassification],
                                               st,
                                               prefix=sp,
                                               after_epoch=True))

    model = Model(cost)

    main_loop = MainLoop(data_stream=streams[0], algorithm=algorithm,
            extensions=extensions+\
                [SaveBest['valid_misclass_best_so_far'],
                 FinishAfter(after_n_epochs=kwargs['max_epochs']),
                 FinishIfNoImprovementAfter['valid_misclass_best_so_far', epochs=kwargs['improve_epochs']],
                 Printing()],
            model=model)

    print('Starting training ...')
    main_loop.run()
    return main_loop.log.current_row['valid_misclass_best_so_far']
