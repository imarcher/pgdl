from pypgdl import *
from line_profiler import LineProfiler
import datetime
n = 128
lp = LineProfiler()

pl = start()
x = pl.create_placeholder_node((n, 1, 28, 28), Tag.nchw)

# conv1
w1 = pl.create_variable_node((32, 1, 5, 5), Tag.oihw)
b1 = pl.create_variable_node((32,), Tag.a)
conv1 = pl.create_convolution_node(x, w1, b1, (1, 1), (2, 2), (2, 2))
conv1_relu = pl.create_bn_relu_node(conv1)
conv1_out = pl.create_pooling_node(conv1_relu, (2, 2), (2, 2), (0, 0), (0, 0))

# conv2
w2 = pl.create_variable_node((64, 32, 5, 5), Tag.oihw)
b2 = pl.create_variable_node((64,), Tag.a)
conv2 = pl.create_convolution_node(conv1_out, w2, b2, (1, 1), (2, 2), (2, 2))
conv2_relu = pl.create_bn_relu_node(conv2)
conv2_out = pl.create_pooling_node(conv2_relu, (2, 2), (2, 2), (0, 0), (0, 0))

conv_flat = pl.create_flat_node(conv2_out)

# dense1
w3 = pl.create_variable_node((7*7*64, 1024), Tag.ab)
b3 = pl.create_variable_node((1, 1024), Tag.ab)
dense1 = pl.create_matmul_node(conv_flat, w3, b3)
dense1_relu = pl.create_bn_relu_node(dense1)

dense1_dropout = pl.create_dropout_node(dense1_relu, 0.5)

# dense1
w4 = pl.create_variable_node((1024, 10), Tag.ab)
b4 = pl.create_variable_node((1, 10), Tag.ab)
dense2 = pl.create_matmul_node(dense1_dropout, w4, b4)
dense2_relu = pl.create_bn_relu_node(dense2)


y = pl.create_placeholder_node((n, 10), Tag.ab)

pl.softmax_cross_entropy_training_accuracy(dense2_relu, y, 0.1, Gd.sgd, 1)

pl.prepare_memory()
pl.init_variables()

def hanshu():
    train_num = 40000


    # starttime = datetime.datetime.now()
    i = 1
    ii = 1
    for j in range(1):
        i = 1
        while i < train_num:
            pl.read_to_node_scalar('y', 'mnisty', i, n, y, 0)
            pl.read_to_node_arr('x', 'mnistx', i, n, x, 0)
            pl.compute_ex()
            print("第" + str(ii) + "次：")
            print(pl.get_loss())
            print(pl.get_accuracy())
            ii = ii + 1
            i = i + n

lp = LineProfiler()
lp_wrapper = lp(hanshu)
lp_wrapper()
lp.print_stats()

# endtime = datetime.datetime.now()
# print((endtime - starttime).seconds)
#
# print(ii)