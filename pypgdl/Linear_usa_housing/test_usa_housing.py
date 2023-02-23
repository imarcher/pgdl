from pypgdl import *
from line_profiler import LineProfiler
lp = LineProfiler()
n = 32
name = 'usa_housing'

pl = start()


def one_mul(in_node, in_c, out_c):
    w = pl.create_variable_node((in_c, out_c), Tag.ab)
    b = pl.create_variable_node((1, out_c), Tag.ab)
    d = pl.create_matmul_node(in_node, w, b)
    return d


def dense(in_node, in_c, out_c):
    d = one_mul(in_node, in_c, out_c)
    d_relu = pl.create_bn_relu_node(d)
    return d_relu


x = pl.create_placeholder_node((n, 5), Tag.ab)

# dense1
d1 = dense(x, 5, 32)
d2 = one_mul(d1, 32, 1)

y = pl.create_placeholder_node((n, 1), Tag.ab)

pl.mse_training_inference(d2, y, 0.004, Gd.sgd, 1)

pl.prepare_memory()
pl.init_variables()

def hanshu():
    train_num = 4000


    # import datetime
    # starttime = datetime.datetime.now()

    i = 1
    ii = 1
    for j in range(10):
        i = 1
        while i < train_num:

            # if ii % 100 == 0:
            pl.read_to_node_real('y', name + '_y_train', i, n, y, 0)
            pl.read_to_node_arr('x', name + '_x_train', i, n, x, 0)
            pl.compute_ex()
            print("第" + str(ii) + "次：")
            print("loss:" + str(pl.get_loss()))
            # print("accuracy:" + str(pl.get_accuracy()))
            # else:
            #     pl.read_to_node_scalar('y', name + '_y_train', i, n, y, 0)
            #     pl.read_to_node_arr('x', name + '_x_train', i, n, x, 0)
            #     pl.compute()
            ii = ii + 1
            i = i + n

        if j==0:
            pl.update_sgd_lr(0.0035)
        if j==1:
            pl.update_sgd_lr(0.003)

        if j==5:
            pl.update_sgd_lr(0.002)



    # endtime = datetime.datetime.now()
    # print((endtime - starttime).seconds)


# lp.add_function(hanshu)

lp_wrapper = lp(hanshu)
lp_wrapper()
lp.print_stats()

test_num = 1000
i = 1
loss = 0
iii = 0
while i < test_num:
    pl.read_to_node_real('y', name + '_y_test', i, n, y, 0)
    pl.read_to_node_arr('x', name + '_x_test', i, n, x, 0)
    pl.compute_loss()

    loss = loss + pl.get_loss()
    iii = iii + 1
    i = i + n

print("loss：" + str(loss/iii))

# pl.del_model(name + '_model')
# pl.save_model(name + '_model')
end(pl)
