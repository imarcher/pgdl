from pypgdl import *

pl = start()


def one_conv(in_node, in_c, out_c):
    w = pl.create_variable_node((out_c, in_c, 3, 3), Tag.oihw)
    b = pl.create_variable_node((out_c,), Tag.a)
    conv = pl.create_convolution_node(in_node, w, b, (1, 1), (1, 1), (1, 1))
    conv_relu = pl.create_bn_relu_node(conv)
    return conv_relu


def conv_block1(in_node, in_c, out_c):
    conv1 = one_conv(in_node, in_c, out_c)
    conv2 = one_conv(conv1, out_c, out_c)
    pool = pl.create_pooling_node(conv2, (2, 2), (2, 2), (0, 0), (0, 0))
    return pool


def conv_block2(in_node, in_c, out_c):
    conv1 = one_conv(in_node, in_c, out_c)
    conv2 = one_conv(conv1, out_c, out_c)
    conv3 = one_conv(conv2, out_c, out_c)
    pool = pl.create_pooling_node(conv3, (2, 2), (2, 2), (0, 0), (0, 0))
    return pool


def one_mul(in_node, in_c, out_c):
    w = pl.create_variable_node((in_c, out_c), Tag.ab)
    b = pl.create_variable_node((1, out_c), Tag.ab)
    d = pl.create_matmul_node(in_node, w, b)
    return d


def dense(in_node, in_c, out_c):
    d = one_mul(in_node, in_c, out_c)
    d_relu = pl.create_bn_relu_node(d)
    # d_dropout = pl.create_dropout_node(d_relu, 0.5)
    return d_relu


n = 128
x = pl.create_placeholder_node((n, 3, 32, 32), Tag.nhwc)
y = pl.create_placeholder_node((n, 10), Tag.ab)

out1 = conv_block1(x, 3, 64)
out2 = conv_block1(out1, 64, 128)
out3 = conv_block2(out2, 128, 256)
out4 = conv_block2(out3, 256, 512)
out5 = conv_block2(out4, 512, 512)

flat = pl.create_flat_node(out5)

d1 = dense(flat, 512, 256)
d2 = dense(d1, 256, 256)
d3 = one_mul(d2, 256, 10)

pl.softmax_cross_entropy_training_accuracy(d3, y, 0.01, Gd.sgd, 1)

pl.prepare_memory()
pl.init_variables()

train_num = 50000
epoch = 1

import datetime
starttime = datetime.datetime.now()

i = 1
ii = 1
for j in range(epoch):
    i = 1
    while i < train_num:

        # if ii % 10 == 0:
        pl.read_to_node_scalar('y', 'cifa10_y_train', i, n, y, 0)
        pl.read_to_node_arr('x', 'cifa10_x_train', i, n, x, 0)
        pl.compute_ex()
        print("第"+str(ii)+"次：")
        print(pl.get_loss())
        print(pl.get_accuracy())
        # else:
        #     pl.read_to_node_scalar('y', 'cifa10_y_train', i, n, y, 0)
        #     pl.read_to_node_arr('x', 'cifa10_x_train', i, n, x, 0)
        #     pl.compute()

        ii = ii + 1
        i = i + n

#long running
endtime = datetime.datetime.now()
print((endtime - starttime).seconds)


# pl.del_model('vgg16_model')
# pl.save_model('vgg16_model')

test_num = 10000
i = 1
arr = 0
iii = 0
while i < test_num:
    pl.read_to_node_scalar('y', 'cifa10_y_test', i, n, y, 0)
    pl.read_to_node_arr('x', 'cifa10_x_test', i, n, x, 0)
    pl.compute_accuracy()

    arr = arr + pl.get_accuracy()
    iii = iii + 1
    i = i + n

print("准确率："+str(arr / iii))

end(pl)
