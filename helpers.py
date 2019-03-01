"""Helper functions."""
import theano
import lasagne as nn


# Save output to file instead of printing
def report(text, output_file='logFile.txt'):
    f = open(output_file, 'a')
    f.write('{}\n'.format(text))
    f.close


# Layer for stochastic depth
class MultiplicationLayer(nn.layers.Layer):
    def __init__(self, incoming, C, **kwargs):
        super(MultiplicationLayer, self).__init__(incoming, **kwargs)    
        self.C = C
    def get_output_for(self, input, **kwargs):
        return self.C.get_value() * input


# Re-writing residual block to add stochastic depth
def ResBlockLayer(incoming, C=None, increase_channels=False):
    # Identity (no 'projection' option)
    if increase_channels:
        first_stride = (2, 2)
        num_filters = incoming.output_shape[1] * 2
        identity = nn.layers.ExpressionLayer(
            incoming=incoming,
            function=lambda X: X[:, :, ::2, ::2],
            output_shape=lambda s: (s[0], s[1], s[2]//2, s[3]//2))
        identity = nn.layers.PadLayer(
            incoming=identity, width=[num_filters//4, 0, 0], batch_ndim=1)
    else:
        num_filters = incoming.output_shape[1]
        first_stride = (1, 1)
        identity = incoming

    # fn: two stacks of convolutional layers
    fn = nn.layers.Conv2DLayer(
        incoming=incoming, num_filters=num_filters, filter_size=(3, 3),
        stride=first_stride, pad='same',nonlinearity=nn.nonlinearities.rectify, 
        W=nn.init.HeNormal(gain='relu'), flip_filters=False)
    fn = nn.layers.batch_norm(fn)
    fn = nn.layers.Conv2DLayer(
        incoming=fn, num_filters=num_filters, filter_size=(3, 3),
        stride=(1, 1), pad='same', nonlinearity=None,
        W=nn.init.HeNormal(gain='relu'), flip_filters=False)
    fn = nn.layers.batch_norm(fn)

    # Include option stochastic depth option here
    # C: 'switch' where 1.=ON and 0.=OFF; C=theano.shared(nn.utils.floatX(1.))
    if C is not None:
        fn = MultiplicationLayer(incoming=fn, C=C)

    # ResBlock: ReLU(C * fn + identity)
    ResBlock = nn.layers.ElemwiseSumLayer([fn, identity])
    ResBlock = nn.layers.NonlinearityLayer(
        incoming=ResBlock, nonlinearity=nn.nonlinearities.rectify)    

    return ResBlock