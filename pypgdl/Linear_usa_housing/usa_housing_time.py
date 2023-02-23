from pypgdl import *


def run(batch, epochs):
    n = batch
    name = 'usa_housing'
    train_num = 4000
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

    ii = 1
    for j in range(epochs):
        i = 1
        while i < train_num:
            pl.read_to_node_real('y', name + '_y_train', i, n, y, 0)
            pl.read_to_node_arr('x', name + '_x_train', i, n, x, 0)
            pl.compute_ex()
            print("第" + str(ii) + "次：")
            print("loss:" + str(pl.get_loss()))
            ii = ii + 1
            i = i + n

        if j == 0:
            pl.update_sgd_lr(0.0035)
        if j == 1:
            pl.update_sgd_lr(0.003)
        if j == 5:
            pl.update_sgd_lr(0.002)

    end(pl)
