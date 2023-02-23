from pypgdl import *
from line_profiler import LineProfiler
lp = LineProfiler()
n = 32
name = 'epsilon'

pl = start()


def one_mul(in_node, in_c, out_c):
    w = pl.create_variable_node((in_c, out_c), Tag.ab)
    b = pl.create_variable_node((1, out_c), Tag.ab)
    d = pl.create_matmul_node(in_node, w, b)
    return d


def dense1(in_node, in_c, out_c):
    d = one_mul(in_node, in_c, out_c)
    d_relu = pl.create_bn_relu_node(d)
    d_dropout = pl.create_dropout_node(d_relu, 0.5)
    return d_dropout


def dense2(in_node, in_c, out_c):
    d = one_mul(in_node, in_c, out_c)
    d_relu = pl.create_bn_relu_node(d)
    return d_relu


x = pl.create_placeholder_node((n, 2000), Tag.ab)

# dense1

d1 = dense2(x, 2000, 1024)
d2 = dense2(d1, 1024, 256)
d4 = one_mul(d2, 256, 2)

y = pl.create_placeholder_node((n, 2), Tag.ab)

pl.softmax_cross_entropy_training_accuracy(d4, y, 0.1, Gd.sgd, 1)

pl.prepare_memory()
pl.init_variables()

def hanshu():
    train_num = 400000


    # import datetime
    # starttime = datetime.datetime.now()

    i = 1
    ii = 1
    for j in range(1):
        i = 1
        while i < train_num:
            print(i)

            # if ii % 100 == 0:
            # pl.read_to_node_scalar('y', name + '_y_train', i, n, y, 0)
            pl.read_to_node_arr('x', name + '_x_train', i, n, x, 0)
            # pl.compute_ex()
            # print("第" + str(ii) + "次：")
            # print("loss:" + str(pl.get_loss()))
            # print("accuracy:" + str(pl.get_accuracy()))
            # else:
            #     pl.read_to_node_scalar('y', name + '_y_train', i, n, y, 0)
            #     pl.read_to_node_arr('x', name + '_x_train', i, n, x, 0)
            #     pl.compute()
            ii = ii + 1
            i = i + n

    # endtime = datetime.datetime.now()
    # print((endtime - starttime).seconds)


# lp.add_function(hanshu)

lp_wrapper = lp(hanshu)
lp_wrapper()
lp.print_stats()

# test_num = 100000
# i = 1
# arr = 0
# iii = 0
# while i < test_num:
#     pl.read_to_node_scalar('y', name + '_y_test', i, n, y, 0)
#     pl.read_to_node_arr('x', name + '_x_test', i, n, x, 0)
#     pl.compute_accuracy()
#
#     arr = arr + pl.get_accuracy()
#     iii = iii + 1
#     i = i + n
#
# print("准确率：" + str(arr / iii))
#
# # pl.del_model(name + '_model')
# # pl.save_model(name + '_model')
# end(pl)
