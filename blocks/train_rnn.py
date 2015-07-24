from __future__ import print_function
import logging
import math
import os
import numpy as np

import theano
from theano import tensor

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from blocks.bricks import (Tanh, Initializable, Linear,
                           Logistic, Softmax, Rectifier, Identity,
                           WEIGHT)
from blocks.bricks.cost import SquaredError, CategoricalCrossEntropy, MisclassificationRate
from blocks.bricks.base import application
from blocks.bricks.recurrent import SimpleRecurrent
from blocks.bricks.sequence_generators import (
    SequenceGenerator, Readout, SoftmaxEmitter, LookupFeedback)
from blocks.config import config
from blocks.graph import ComputationGraph
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
from extensions import SaveTheBest

config.recursion_limit = 100000
floatX = theano.config.floatX
logger = logging.getLogger(__name__)

x_dim = 85
y_dim = 57

dict_to_act = {'tanh': Tanh,
               'logistic': Logistic,
               'rectifier': Rectifier}
dict_to_init = {'isotropic': IsotropicGaussian}

def train_model(data, opt_params, static_params):

    print('Bulding model...')
    x = tensor.tensor3('x', dtype=floatX)
    y = tensor.tensor3('y', dtype=floatX)


    x_to_h = Linear(name='x_to_h',
            input_dim=x_dim,
            output_dim=opt_params['h_dim'])
    x_transform = x_to_h.apply(x.dimshuffle(1, 0, 2))
    rec = SimpleContinuousRecurrent(activation=dict_to_act[opt_params['act']](),
            dim=opt_params['h_dim'],
            name='rec')
    h = rec.apply(x_transform)
    h_to_y = Linear(name='h_to_y',
            input_dim=opt_params['h_dim'],
            output_dim=y_dim)
    y_hat = h_to_y.apply(h)
    shape = y_hat.shape
    y_hat_time = tensor.nnet.softmax(y_hat.reshape((tensor.prod(shape[:-1]),
        shape[-1]))).reshape(shape)

    y_hat = Softmax().apply(y_hat.mean(0))
    y_hat.name = 'y_hat'

    print('Defining Cost...')
    cost = CategoricalCrossEntropy().apply(y.mean(1), y_hat)
    cost.name = 'xent'
    misclassification = MisclassificationRate().apply(y.mean(1).argmax(1), y_hat)
    misclassification.name = 'misclass'

    cg = ComputationGraph([cost])

    # Weight decay
    weights = VariableFilter(roles=[WEIGHT])(cg.variables)
    l1_wd = np.power(10., opt_params['log_l1_wd'])
    cost_final = cost + sum([l1_wd*abs(w).sum() for w in weights])
    l2_wd = np.power(10., opt_params['log_l2_wd'])
    cost_final += sum([l2_wd*(w**2).sum() for w in weights])
    cost_final.name = 'final_cost'

    print('Initializing Parameters...')
    for brick in (rec, x_to_h, h_to_y):
        brick.weights_init = dict_to_init[opt_params['init']](opt_params['init_scale'])
        brick.biases_init = Constant(0.)
        brick.initialize()

    print('Bulding training process...')
    algorithm = GradientDescent(cost=cost_final,
            params=cg.parameters,
            step_rule=CompositeRule([StepClipping(10.0),
            Adam()]))

    extensions = []

    for sp, st in data:
        extensions.append(DataStreamMonitoring([cost, misclassification],
                                               st,
                                               prefix=sp,
                                               after_epoch=True))

    model = Model(cost_final)
    train_string, train_stream = data[0]

    main_loop = MainLoop(data_stream=train_stream, algorithm=algorithm,
            extensions=extensions+\
                [SaveTheBest('valid_misclass', 'model.pkl'),
                 FinishAfter(after_n_epochs=opt_params['max_epochs']),
                 FinishIfNoImprovementAfter('valid_misclass_best_so_far', epochs=opt_params['improve_epochs']),
                 Printing()],
            model=model)

    print('Starting training ...')
    main_loop.run()
    predict = theano.function([x], y_hat_time)
    sp, st = data[-1]
    ds = st.dataset
    num_examples = ds.num_examples
    features = ds.indexables[ds.sources.index('x')]
    targets = ds.indexables[ds.sources.index('y')]
    print(features.shape)
    print(targets.shape)
    predictions = predict(features)
    time_pts = range(predictions.shape[0])
    with PdfPages('output.pdf') as pdf:
        for ii in range(10):
            pred = predictions[:,ii]
            tar = targets[ii,0].argmax()
            fig = plt.figure()
            plt.plot(pred[:,tar])
            pdf.savefig()
            plt.close(fig)
        preds = np.zeros((num_examples, pred.shape[0]))
        for ii in range(num_examples):
            pred = predictions[:,ii]
            tar = targets[ii,0].argmax()
            preds[ii] = pred[:,tar]
        mean_pred = preds.mean(0)
        std_pred = preds.std(0)
        fig = plt.figure()
        plt.plot(time_pts, mean_pred, color='black')
        plt.fill_between(time_pts, mean_pred-std_pred, mean_pred+std_pred, facecolor='yellow', alpha=.5)
        pdf.savefig()
        plt.close(fig)
    print('-------tau----------')
    print(rec.tau.get_value())

    return main_loop.log.current_row['valid_misclass_best_so_far']
