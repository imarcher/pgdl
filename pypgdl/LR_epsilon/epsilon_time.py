from pypgdl import *


def run(batch, epochs):
    n = batch
    name = 'epsilon'
    train_num = 400000
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

    x = pl.create_placeholder_node((n, 2000), Tag.ab)

    # dense

    d1 = dense(x, 2000, 1024)
    d2 = dense(d1, 1024, 256)
    d4 = one_mul(d2, 256, 2)

    y = pl.create_placeholder_node((n, 2), Tag.ab)

    pl.softmax_cross_entropy_training_accuracy(d4, y, 0.1, Gd.sgd, 1)

    pl.prepare_memory()
    pl.init_variables()

    ii = 1
    for j in range(epochs):
        i = 1
        while i < train_num:
            pl.read_to_node_scalar('y', name + '_y_train', i, n, y, 0)
            pl.read_to_node_arr('x', name + '_x_train', i, n, x, 0)
            pl.compute_ex()
            print("第" + str(ii) + "次：")
            print("loss:" + str(pl.get_loss()))
            print("accuracy:" + str(pl.get_accuracy()))

            ii = ii + 1
            i = i + n

    end(pl)
