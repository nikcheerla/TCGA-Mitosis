

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split

from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, FeaturePoolLayer
from lasagne.layers import Conv2DLayer, MaxPool2DLayer
from lasagne import nonlinearities
from lasagne import objectives
from lasagne import updates

from nolearn.lasagne import NeuralNet
from nolearn.lasagne.handlers import SaveWeights

from nolearn_utils.iterators import (
    BufferedBatchIteratorMixin,
    ShuffleBatchIteratorMixin,
    AffineTransformBatchIteratorMixin,
    make_iterator
)
from nolearn_utils.hooks import (
    SaveTrainingHistory, PlotTrainingHistory,
    EarlyStopping
)


"""Important GPU testing! Will verify whether GPU or CPU is being used!"""

from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time


vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 100

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], T.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')



def load_data(test_size=0.25, random_state=None):
    df = pd.read_csv('train.csv')
    print "Read in Data"
    X = df[df.columns[1:]].values.reshape(-1, 1, 28, 28).astype(np.float32)
    X = X / 255
    y = df['label'].values.astype(np.int32)
    print "Shaped Data"
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


image_size = 28
batch_size = 1024
n_classes = 10

print "Making mixins"
train_iterator_mixins = [
    BufferedBatchIteratorMixin,
]
TrainIterator = make_iterator('TrainIterator', train_iterator_mixins)

test_iterator_mixins = [
    BufferedBatchIteratorMixin,
]
TestIterator = make_iterator('TestIterator', test_iterator_mixins)

train_iterator_kwargs = {
    'batch_size': batch_size,
    'buffer_size': 5,
}
train_iterator = TrainIterator(**train_iterator_kwargs)

test_iterator_kwargs = {
    'batch_size': batch_size,
    'buffer_size': 5,
}
test_iterator = TestIterator(**test_iterator_kwargs)

save_training_history = SaveTrainingHistory('model_history.pkl')
plot_training_history = PlotTrainingHistory('training_history.png')

print 'Making Network'
net = NeuralNet(
    layers=[
        (InputLayer, dict(name='in', shape=(None, 1, image_size, image_size))),

        (DenseLayer, dict(name='l8', num_units=256)),
        (FeaturePoolLayer, dict(name='l8p', pool_size=2)),
        (DropoutLayer, dict(name='l8drop', p=0.5)),

        (DenseLayer, dict(name='out', num_units=10, nonlinearity=nonlinearities.softmax)),
    ],

    regression=False,
    objective_loss_function=objectives.categorical_crossentropy,

    update=updates.adam,
    update_learning_rate=1e-3,

    batch_iterator_train=train_iterator,
    batch_iterator_test=test_iterator,

    on_epoch_finished=[
        save_training_history,
        plot_training_history,
    ],

    verbose=10,
    max_epochs=20
)

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data(test_size=0.25, random_state=42)
    print "Training Network"
    net.fit(X_train, y_train)
    score = net.score(X_test, y_test)
    print 'Final score %.4f' % score
