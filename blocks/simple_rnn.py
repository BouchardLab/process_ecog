#!/usr/bin/env python
import numpy
import theano
from theano import tensor
from blocks import initialization
from blocks import bricks
from blocks.bricks import recurrent
x = tensor.tensor3('x')
rnn = recurrent.LSTM(dim=1,
                     activation=bricks.Tanh(),
                     weights_init=initialization.Identity())
rnn.initialize()
h = rnn.apply(x)
f = theano.function([x], h)
print(f(numpy.ones((3, 1, 3), dtype=theano.config.floatX))) 
