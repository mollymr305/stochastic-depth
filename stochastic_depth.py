import sys
import time
import datetime as dt
import gzip
import cPickle as pickle
import theano
import theano.tensor as T
import lasagne as nn
import numpy as np
from Deep_Residual_Learning_CIFAR10 import load_data
from Deep_Residual_Learning_CIFAR10 import iterate_minibatches
from __functions__ import report
from __functions__ import ResBlockLayer

# basic settings
sys.setrecursionlimit(2 ** 20)
title = 'stochastic_depth'
output_file = './output/{}.txt'.format(title)

if __name__ == '__main__':
    report('\n\nSTART: {}'.format(dt.datetime.now()), output_file)
    report('>>> {}'.format(output_file), output_file)
    # model parameters
    try:
        N = int(sys.argv[1])
    except:
        N = 9 # default depth uses n = 9.
    epochs, batchsize = 500, 128
    report('Depth: N = {}, #Layers = {}.'.format(N, 6*N+2), output_file)
    delta = 0.5 / (3 * N - 1)
    decay = [1.0 - i * delta for i in xrange(3*N)] # linear decay rule
    switches = [theano.shared(nn.utils.floatX(1.)) for i in xrange(3*N)]
    on, off = 1.0, 0.0 # nn.utils.floatX(1.), nn.utils.floatX(0.)
    # use to turn on all blocks (during testing)
    def full_depth():
        global switches, on
        for c in switches:
            c.set_value(on)
    # use to randomly on/off blocks based on decay rule (during training)
    def stochastic_depth():
        global switches, on, off, decay
        assert len(switches) == len(decay)
        drop = np.random.uniform(low=0, high=1, size=len(switches)) > decay
        for c, d in zip(switches, drop):
            c.set_value(off) if d else c.set_value(on)
    # symbolic variables
    X, Y = T.tensor4('inputs'), T.ivector('targets')
    # model architecture
    net = nn.layers.InputLayer(shape=(None, 3, 32, 32), input_var=X)
    # 16 x 32 x 32
    net = nn.layers.Conv2DLayer(
        incoming=net, num_filters=16, filter_size=(3,3), stride=(1,1), 
        nonlinearity=nn.nonlinearities.rectify, pad='same', 
        W=nn.init.HeNormal(gain='relu'), flip_filters=False)
    net = nn.layers.batch_norm(net)
    i = 0; # add i-th switch to i-th block
    for _ in range(N):
        net = ResBlockLayer(incoming=net,C=switches[i],increase_channels=False)
        i += 1
    # 32 x 16 x 16
    net = ResBlockLayer(incoming=net,C=switches[i],increase_channels=True)
    i += 1
    for _ in range(N-1):
        net = ResBlockLayer(incoming=net,C=switches[i],increase_channels=False)
        i += 1
    # 64 x 8 x 8
    net = ResBlockLayer(incoming=net,C=switches[i],increase_channels=True)
    i += 1
    for _ in range(N-1):
        net = ResBlockLayer(incoming=net,C=switches[i],increase_channels=False)
        i += 1
    net = nn.layers.GlobalPoolLayer(net)
    net = nn.layers.DenseLayer(
        incoming=net, num_units=10, W=nn.init.HeNormal(), 
        nonlinearity=nn.nonlinearities.softmax)
    report('Model OK...', output_file)
    num_params = nn.layers.count_params(net, trainable=True)
    report("Number of parameters: {}".format(num_params), output_file)
    # training function
    output = nn.layers.get_output(net, deterministic=False)
    loss = nn.objectives.categorical_crossentropy(output, Y).mean()
    accuracy = T.mean(
        T.eq(T.argmax(output, axis=1), Y), dtype=theano.config.floatX)
    all_layers = nn.layers.get_all_layers(net)
    regularization = nn.regularization.regularize_layer_params(
        layer=all_layers, penalty=nn.regularization.l2)
    loss = loss + 1e-4 * regularization
    params = nn.layers.get_all_params(net, trainable=True)
    learning_rate = theano.shared(nn.utils.floatX(0.1))
    updates = nn.updates.momentum(
        loss_or_grads=loss, params=params,
        learning_rate=learning_rate, momentum=0.9)
    training_function = theano.function([X,Y],[loss,accuracy],updates=updates)
    report('Training function OK...', output_file)
    # test/validation function
    test_output = nn.layers.get_output(net, deterministic=True)
    test_loss = nn.objectives.categorical_crossentropy(test_output, Y).mean()
    test_accuracy = T.mean(
        T.eq(T.argmax(test_output, axis=1), Y), dtype=theano.config.floatX)
    test_function = theano.function([X,Y], [test_loss,test_accuracy])
    report('Validation function OK...', output_file)
    # load data
    data = load_data()
    X_train, Y_train = data['X_train'], data['Y_train']
    X_test, Y_test = data['X_test'], data['Y_test']
    report('Data OK...', output_file)
    # start training
    # training loss, training accuracy, validation loss, validation accuracy
    TL, TA, VL, VA = [], [], [], []
    report('Starting training...', output_file)
    header = ['Epoch', 'TL', 'TA', 'VL', 'VA', 'Time']
    report('{:<10}{:<20}{:<20}{:<20}{:<20}{:<20}'.format(*header), output_file)
    learning_rate_schedule = {250:0.01, 375:0.001}
    for e in xrange(epochs):
        if (e + 1) in learning_rate_schedule:
            learning_rate.set_value(learning_rate_schedule[e + 1])
        start_time = time.time()
        t_batches, v_batches = 0, 0
        tl, ta, vl, va = 0., 0., 0., 0.
        # train with stochastic depth
        stochastic_depth()
        minibatches = iterate_minibatches(
            X_train, Y_train, batchsize, shuffle=True, augment=True)
        for data, target in minibatches:
            l, a = training_function(data, target)
            tl += l; ta += a; t_batches += 1;
        tl /= t_batches; ta /= t_batches; TL.append(tl); TA.append(ta);
        # test with full depth
        full_depth()
        minibatches = iterate_minibatches(
            X_test, Y_test, 500, shuffle=False, augment=False)
        for data, target in minibatches:
            l, a = test_function(data, target)
            vl += l; va += a; v_batches += 1;
        vl /= v_batches; va /= v_batches; VL.append(vl); VA.append(va);
        row = [e + 1, tl, ta, vl, va, time.time() - start_time]
        report('{:<10}{:<20}{:<20}{:<20}{:<20}{:<20}'.format(*row),output_file)
    report('Finished training...', output_file)
    # save training information
    f = gzip.open('./output/{}_info.pkl.gz'.format(title), 'wb')
    info = {
        'training loss': TL,
        'training accuracy': TA,
        'validation loss': VL,
        'validation accuracy': VA
    }
    pickle.dump(info, f)
    f.close()
    report('Saved training information...', output_file)
    # save weights
    weights = nn.layers.get_all_params(net)
    weights = [np.array(w.get_value()) for w in weights]
    f = gzip.open('./output/{}_weights.pkl.gz'.format(title), 'wb')
    pickle.dump(weights, f)
    f.close()
    report('Saved weights...', output_file=output_file)
    report('END: {}'.format(dt.datetime.now()), output_file)