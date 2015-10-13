from pylearn2.compat import OrderedDict
from pylearn2.linear.matrixmul import TopoFactorizedMatrixMul
from pylearn2.models import mlp
from pylearn2.space import Conv2DSpace, VectorSpace
from pylearn2.model_extensions.norm_constraint import MaxL2FilterNorm
from pylearn2.utils import sharedX, wraps
from pylearn2.expr.nnet import multi_class_prod_misclass

import theano.tensor as T
import numpy as np


class MultiProdFlattenerLayer(mlp.FlattenerLayer):
    @wraps(mlp.Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                      state=None, targets=None):
        rval = super(TwoProdFlattenerLayer, self).get_layer_monitoring_channels(
                state_below=state_below,
                state=state,
                targets=targets)
        if targets is not None:
            name = 'misclass'
            rval[name] = multi_class_prod_misclass(targets,
                                                   state,
                                                   self.get_output_space(),
                                                   self.raw_layer.get_output_space())
        return rval

class TwoProdFlattenerLayer(MultiProdFlattenerLayer):
    pass


class TopoFactorizedLinear(mlp.Linear):
    @wraps(mlp.Layer.set_input_space)
    def set_input_space(self, space):
        assert isinstance(space, Conv2DSpace)

        self.input_space = space
        self.requires_reformat = True
        self.input_dim = space.get_total_dimension()
        self.desired_space = VectorSpace(self.input_dim)
        self.output_space = VectorSpace(self.dim)

        rng = self.mlp.rng
        if self.irange is not None:
            raise NotImplementedError
        elif self.istdev is not None:
            assert self.sparse_init is None
            if space.num_channels > 1:
                c = rng.randn(space.num_channels, self.dim) * self.istdev
                c = sharedX(c)
                c.name = self.layer_name + '_c'
            else:
                c = None
            if space.shape[0] > 1:
                zero = rng.randn(space.shape[0], self.dim) * self.istdev
                zero = sharedX(zero)
                zero.name = self.layer_name + '_zero'
            else:
                zero = None
            if space.shape[1] > 1:
                one = rng.randn(space.shape[1], self.dim) * self.istdev
                one = sharedX(one)
                one.name = self.layer_name + '_one'
            else:
                one = None
        else:
            raise NotImplementedError

        self.transformer = TopoFactorizedMatrixMul(self.dim, zero, one, c)

        params = self.transformer.get_params()
        for param in params:
            assert param.name is not None

        if self.mask_weights is not None:
            raise NotImplementedError

    @wraps(mlp.Layer._modify_updates)
    def _modify_updates(self, updates):

        if self.mask_weights is not None:
            raise NotImplementedError

        if self.max_row_norm is not None:
            raise NotImplementedError

        if self.max_col_norm is not None or self.min_col_norm is not None:
            assert self.max_row_norm is None
            if self.max_col_norm is not None:
                max_col_norm = self.max_col_norm
            if self.min_col_norm is None:
                self.min_col_norm = 0
            params = self.transformer.get_params()
            for param in params:
                if param in updates:
                    updated_param = updates[param]
                    col_norms = T.sqrt(T.sum(T.sqr(updated_param), axis=0))
                    if self.max_col_norm is None:
                        max_col_norm = col_norms.max()
                    desired_norms = T.clip(col_norms,
                                           self.min_col_norm,
                                           max_col_norm)
                    scale = desired_norms / T.maximum(1.e-7, col_norms)
                    updates[param] = updated_param * scale

    @wraps(mlp.Layer.get_params)
    def get_params(self):

        params = self.transformer.get_params()
        for param in params:
            assert param.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval)
        if self.use_bias:
            assert self.b.name is not None
            assert self.b not in rval
            rval.append(self.b)
        return rval

    @wraps(mlp.Layer.get_weight_decay)
    def get_weight_decay(self, coeff):

        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        params = self.transformer.get_weights()
        return sum([coeff * T.sqr(param).sum() for param in params])

    @wraps(mlp.Layer.get_l1_weight_decay)
    def get_l1_weight_decay(self, coeff):

        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        params = self.transformer.get_weights()
        return sum([coeff * abs(param).sum() for param in params])

    @wraps(mlp.Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                      state=None, targets=None):
        params = self.transformer.get_params()
        rval = OrderedDict()
        for param in params:
            assert param.ndim == 2

            sq_param = T.sqr(param)

            row_norms = T.sqrt(sq_param.sum(axis=1))
            col_norms = T.sqrt(sq_param.sum(axis=0))
            name = param.name

            rval.update({name+'_row_norms_min':  row_norms.min(),
                         name+'_row_norms_mean': row_norms.mean(),
                         name+'_row_norms_max':  row_norms.max(),
                         name+'_col_norms_min':  col_norms.min(),
                         name+'_col_norms_mean': col_norms.mean(),
                         name+'_col_norms_max':  col_norms.max()})

        if (state is not None) or (state_below is not None):
            if state is None:
                state = self.fprop(state_below)

            mx = state.max(axis=0)
            mean = state.mean(axis=0)
            mn = state.min(axis=0)
            rg = mx - mn

            rval['range_x_max_u'] = rg.max()
            rval['range_x_mean_u'] = rg.mean()
            rval['range_x_min_u'] = rg.min()

            rval['max_x_max_u'] = mx.max()
            rval['max_x_mean_u'] = mx.mean()
            rval['max_x_min_u'] = mx.min()

            rval['mean_x_max_u'] = mean.max()
            rval['mean_x_mean_u'] = mean.mean()
            rval['mean_x_min_u'] = mean.min()

            rval['min_x_max_u'] = mn.max()
            rval['min_x_mean_u'] = mn.mean()
            rval['min_x_min_u'] = mn.min()

        return rval

class Tanh(TopoFactorizedLinear):
    @wraps(mlp.Layer.fprop)
    def fprop(self, state_below):

        p = self._linear_part(state_below)
        p = T.tanh(p)
        return p

class Sigmoid(TopoFactorizedLinear):
    @wraps(mlp.Layer.fprop)
    def fprop(self, state_below):

        p = self._linear_part(state_below)
        p = T.nnet.sigmoid(p)
        return p

class RectifiedLinear(TopoFactorizedLinear):
    @wraps(mlp.Layer.fprop)
    def fprop(self, state_below):

        p = self._linear_part(state_below)
        p = T.switch(p > 0., p, 0.)
        return p
