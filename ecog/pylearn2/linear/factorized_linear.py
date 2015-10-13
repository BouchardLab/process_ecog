import theano
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d

from pylearn2.linear.linear_transform import LinearTransform
from pylearn.linear.matrixmul import MatrixMul
from pylearn2.packaged_dependencies.theano_linear.conv2d \
    import Conv2d as OrigConv2D


class TopoFactorizedMatrixMul(MatrixMul):
    """
    Matrix multiply with the weight matrix factorized by the
    the topological shape on the input.
    Parameters
    ----------
    zero : WRITEME
    one : shared variable
    c : shared variable
    dim : int
    """

    def __init__(self, out_dim, zero=None, one=None, c=None, axes=('b', 0, 1, 'c'):
        """
        Sets the initial values of the matrix
        """
        params = [zero, one, c]
        self.params = [param for param in params if param is not None]
        assert len(self.params) > 0
        self._dim = out_dim
        if zero is not None:
            zero = zero.dimshuffle('x', 'x', 0, 1)
        if one is not None:
            one = one.dimshuffle('x', 0, 'x', 1)
        if c is not None:
            c = c.dimshuffle(0, 'x', 'x', 1)
        reshaped_params = [zero, one, c]
        reshaped_params = [param for param in reshaped_params if param is not None]
        W = 1.
        for param in reshaped_params:
            W = W*param
        self._W = W.reshape((-1, out_dim))

    @functools.wraps(LinearTransform.get_params)
    def get_params(self):
        return self.params

    def get_weights(self):
        return [self._W]


class TopoFactorizedConv2D(OrigConv2D):
    """
    Extend the TheanoLinear Conv2d class to support everything
    needed for a pylearn2 linear operator.

    Also extend it to handle different axis semantics.

    Parameters
    ----------
    filters : Theano shared variable
        4-tensor of shape (out channels, in channels, rows, cols)
    batch_size : WRITEME
    input_space : WRITEME
    output_axes : WRITEME
    subsample : WRITEME
    border_mode : WRITEME
    filters_shape : WRITEME
    message : WRITEME
    """

    def __init__(self, filters, batch_size, input_space,
                 output_axes=('b', 0, 1, 'c'), subsample=(1, 1),
                 border_mode='valid', filters_shape=None, message=''):
        assert batch_size is None or batch_size > 0
        self.input_space = input_space
        self.output_axes = output_axes

        super(Conv2D, self).__init__(
            filters=filters,
            img_shape=(batch_size, input_space.num_channels,
                       input_space.shape[0], input_space.shape[1]),
            subsample=subsample,
            border_mode=border_mode,
            filters_shape=filters.get_value(borrow=True).shape,
            message=message
        )

    @functools.wraps(P2LT.get_params)
    def get_params(self):
        """
        .. todo::

            WRITEME
        """
        return [self._filters]

    @functools.wraps(P2LT.get_weights_topo)
    def get_weights_topo(self, borrow):
        """
        .. todo::

            WRITEME
        """
        return np.transpose(self._filters.get_value(borrow=borrow),
                            (0, 2, 3, 1))

    def lmul(self, x):
        """
        .. todo::

            WRITEME properly

        dot(x, A)

        This method overrides the original Conv2D lmul to make it work
        with arbitrary axis orders
        """

        # x must be formatted as batch index, channel, topo dim 0, topo dim 1
        # for use with conv2d, so check what the current input space format is
        assert x.ndim == 4
        axes = self.input_space.axes
        assert len(axes) == 4

        op_axes = ('b', 'c', 0, 1)

        if tuple(axes) != op_axes:
            x = x.dimshuffle(
                axes.index('b'),
                axes.index('c'),
                axes.index(0),
                axes.index(1))

        rval = conv2d(
            x, self._filters,
            image_shape=self._img_shape,
            filter_shape=self._filters_shape,
            subsample=self._subsample,
            border_mode=self._border_mode,
        )

        # Format the output based on the output space
        axes = self.output_axes
        assert len(axes) == 4

        if tuple(axes) != op_axes:
            rval = rval.dimshuffle(
                op_axes.index(axes[0]),
                op_axes.index(axes[1]),
                op_axes.index(axes[2]),
                op_axes.index(axes[3])
            )

        return rval

    def lmul_T(self, x):
        """
        .. todo::

            WRITEME properly

        Override the original Conv2D lmul_T to make it work
        with pylearn format of topological data using dimshuffles
        """
        assert x.dtype == self._filters.dtype

        op_axes = ('b', 'c', 0, 1)
        axes = self.output_axes
        if tuple(axes) != op_axes:
            x = x.dimshuffle(
                axes.index('b'),
                axes.index('c'),
                axes.index(0),
                axes.index(1)
            )

        # dot(x, A.T)
        dummy_v = T.tensor4()
        dummy_v.name = 'dummy_v'

        # Since we made this variable, we need to put a tag on it
        if theano.config.compute_test_value == 'raise':
            dummy_v.tag.test_value = np.zeros(
                (x.tag.test_value.shape[0],
                 self.input_space.num_channels,
                 self.input_space.shape[0],
                 self.input_space.shape[1]),
                dtype=dummy_v.dtype
            )

        z_hs = conv2d(
            dummy_v, self._filters,
            image_shape=self._img_shape,
            filter_shape=self._filters_shape,
            subsample=self._subsample,
            border_mode=self._border_mode,
        )

        rval, xdummy = z_hs.owner.op.grad((dummy_v, self._filters), (x,))

        # Format the output based on the input space
        axes = self.input_space.axes
        assert len(axes) == 4

        if tuple(axes) != op_axes:
            rval = rval.dimshuffle(
                op_axes.index(axes[0]),
                op_axes.index(axes[1]),
                op_axes.index(axes[2]),
                op_axes.index(axes[3])
            )

        return rval


    def set_batch_size(self, batch_size):
        """
        .. todo::

            WRITEME
        """
        self._img_shape = tuple([batch_size] + list(self._img_shape[1:]))


def make_random_conv2D(input_space, output_space,
                       kernel_shape, batch_size=None, \
                       subsample = (1,1), border_mode = 'valid',
                       message = "", rng = None,
                       irange=None, istdev=None,
                       topo_factorized=False):
    """
    .. todo::

        WRITEME properly

    Creates a Conv2D with random kernels
    """

    rng = make_np_rng(rng, default_seed, which_method=['uniform', 'normal'])

    if irange is None:
        assert istdev is not None
        W = sharedX(rng.normal(
            -istdev, istdev,
            (output_space.num_channels, input_space.num_channels,
             kernel_shape[0], kernel_shape[1])
        ))
    else:
        assert istdev is None
        W = sharedX(rng.uniform(
            -irange, irange,
            (output_space.num_channels, input_space.num_channels,
             kernel_shape[0], kernel_shape[1])
        ))

    return Conv2D(
        filters=W,
        batch_size=batch_size,
        input_space=input_space,
        output_axes=output_space.axes,
        subsample=subsample, border_mode=border_mode,
        filters_shape=W.get_value(borrow=True).shape, message=message
    )
