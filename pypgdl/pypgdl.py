import psycopg2
import logging
from enum import IntEnum


class Tag(IntEnum):
    ab = 0
    a = 1
    nhwc = 2
    nchw = 3
    ohwi = nhwc
    oihw = nchw


class Gd(IntEnum):
    sgd = 0
    adam = 1


class PgContext:
    def __init__(self, conn, cur):
        self.conn = conn
        self.cur = cur

    def create_placeholder_node(self, shape, format_tag):
        dim = len(shape)
        shape = get_shape_str(shape, dim)
        format_tag = int(format_tag)
        return execute_sql(self.cur, "select create_placeholder(%s, %s, %s)", (shape, dim, format_tag))

    def create_variable_node(self, shape, format_tag):
        dim = len(shape)
        shape = get_shape_str(shape, dim)
        format_tag = int(format_tag)
        return execute_sql(self.cur, "select create_variable(%s, %s, %s)", (shape, dim, format_tag))

    def create_reverse_node(self, node1):
        return execute_sql(self.cur, "select c_reverse_op(%s)", (node1,))

    def create_relu_node(self, node1):
        return execute_sql(self.cur, "select c_relu_op(%s)", (node1,))

    def create_bn_node(self, node1):
        return execute_sql(self.cur, "select c_bn_op(%s)", (node1,))

    def create_bn_relu_node(self, node1):
        return execute_sql(self.cur, "select c_bn_relu_op(%s)", (node1,))

    def create_dropout_node(self, node1, keep_prob):
        return execute_sql(self.cur, "select c_dropout_op(%s, %s)", (node1, keep_prob))

    def create_dropout_i_node(self, node1, keep_prob):
        return execute_sql(self.cur, "select c_dropout_i_op(%s, %s)", (node1, keep_prob))

    def create_add_node(self, node1, node2):
        return execute_sql(self.cur, "select c_add_op(%s, %s)", (node1, node2))

    def create_sub_node(self, node1, node2):
        return execute_sql(self.cur, "select c_sub_op(%s, %s)", (node1, node2))

    def create_matmul_node(self, node1, node2, node3):
        return execute_sql(self.cur, "select c_matmul_op(%s, %s, %s)", (node1, node2, node3))

    def create_convolution_node(self, node1, node2, node3, strides_dims, padding_dims_l, padding_dims_r):
        strides_dims = get_shape_str(strides_dims, len(strides_dims))
        padding_dims_l = get_shape_str(padding_dims_l, len(padding_dims_l))
        padding_dims_r = get_shape_str(padding_dims_r, len(padding_dims_r))
        return execute_sql(self.cur, "select c_convolution_op(%s, %s, %s, %s, %s, %s)",
                           (node1, node2, node3, strides_dims, padding_dims_l, padding_dims_r))

    def create_pooling_node(self, node1, strides_dims, kernel, padding_dims_l, padding_dims_r):
        strides_dims = get_shape_str(strides_dims, len(strides_dims))
        kernel = get_shape_str(kernel, len(kernel))
        padding_dims_l = get_shape_str(padding_dims_l, len(padding_dims_l))
        padding_dims_r = get_shape_str(padding_dims_r, len(padding_dims_r))
        return execute_sql(self.cur, "select c_pooling_op(%s, %s, %s, %s, %s)",
                           (node1, strides_dims, kernel, padding_dims_l, padding_dims_r))

    def create_flat_node(self, node1):
        return execute_sql(self.cur, "select c_flat_op(%s)", (node1,))

    def softmax_cross_entropy_training(self, x, y, lr, gd, execute_num):
        gd = int(gd)
        return execute_sql(self.cur, "select softmax_cross_entropy_training(%s, %s, %s, %s, %s)",
                           (x, y, lr, gd, execute_num))

    def softmax_cross_entropy_inference(self, x, y, execute_num):
        return execute_sql(self.cur, "select softmax_cross_entropy_inference(%s, %s, %s)", (x, y, execute_num))

    def softmax_cross_entropy_accuracy(self, x, y, execute_num):
        return execute_sql(self.cur, "select softmax_cross_entropy_accuracy(%s, %s, %s)", (x, y, execute_num))

    def softmax_cross_entropy_training_accuracy(self, x, y, lr, gd, execute_num):
        gd = int(gd)
        return execute_sql(self.cur, "select softmax_cross_entropy_training_accuracy(%s, %s, %s, %s, %s)"
                           , (x, y, lr, gd, execute_num))

    def predict(self, x, execute_num):
        return execute_sql(self.cur, "select predict(%s, %s)", (x, execute_num))

    def mse_training(self, x, y, lr, gd, execute_num):
        gd = int(gd)
        return execute_sql(self.cur, "select mse_training(%s, %s, %s, %s, %s)"
                           , (x, y, lr, gd, execute_num))

    def mse_training_inference(self, x, y, lr, gd, execute_num):
        gd = int(gd)
        return execute_sql(self.cur, "select mse_training_inference(%s, %s, %s, %s, %s)"
                           , (x, y, lr, gd, execute_num))

    def inference(self, node, execute_num):
        return execute_sql(self.cur, "select inference(%s, %s)"
                           , (node, execute_num))

    def read_to_node_scalar(self, column, table, start_id, all_num, node, execute_id):
        return execute_sql(self.cur, "select read_to_node_scalar(%s, %s, %s, %s, %s, %s)"
                           , (column, table, start_id, all_num, node, execute_id))

    def read_to_node_real(self, column, table, start_id, all_num, node, execute_id):
        return execute_sql(self.cur, "select read_to_node_real(%s, %s, %s, %s, %s, %s)"
                           , (column, table, start_id, all_num, node, execute_id))

    def read_to_node_arr(self, column, table, start_id, all_num, node, execute_id):
        return execute_sql(self.cur, "select read_to_node_arr(%s, %s, %s, %s, %s, %s)"
                           , (column, table, start_id, all_num, node, execute_id))

    def init_variables(self):
        return execute_sql(self.cur, "select init_variables()")

    def prepare_memory(self):
        return execute_sql(self.cur, "select prepare_memory()")

    def compute(self):
        return execute_sql(self.cur, "select compute()")

    def compute_ex(self):
        return execute_sql(self.cur, "select compute_ex()")

    def compute_accuracy(self):
        return execute_sql(self.cur, "select compute_accuracy()")

    def compute_loss(self):
        return execute_sql(self.cur, "select compute_accuracy()")

    def wait(self):
        return execute_sql(self.cur, "select wait()")

    def get_loss(self):
        return execute_sql(self.cur, "select get_loss()")

    def get_accuracy(self):
        return execute_sql(self.cur, "select get_accuracy()")

    def create_predict(self, name):
        sql = "CREATE TABLE IF NOT EXISTS " + name + " (" \
                                                     "id integer NOT NULL GENERATED ALWAYS AS IDENTITY " \
                                                     "( INCREMENT 1 START 1 MINVALUE 1 MAXVALUE 99999999 CACHE 1 )," \
                                                     "label integer NOT NULL," \
                                                     "PRIMARY KEY (id))"
        self.cur.execute(sql)

    def save_predict(self, name):
        res = execute_sql(self.cur, "select save_predict(%s)", (name,))
        if res == 0:
            print("保存预测在" + name)
        elif res == -100:
            print("预测保存失败")
        return res

    def del_predict(self, name):
        self.cur.execute("drop table if exists " + name)

    def update_sgd_lr(self, lr):
        return execute_sql(self.cur, "select update_sgd_lr(%s)", (lr,))

    def save_model(self, name):
        res = execute_sql(self.cur, "select save_model(%s)", (name,))
        if res == 0:
            print("保存模型" + name)
        elif res == -100:
            print("模型" + name + "已存在")
        return res

    def load_model(self, name):
        res = execute_sql(self.cur, "select load_model(%s)", (name,))
        if res == 0:
            print("已加载模型" + name)
        else:
            print("模型" + name + "加载错误")
        return res

    def del_model(self, name):
        self.cur.execute("drop table if exists " + name)

    def create_table_x(self, name):
        sql = "CREATE TABLE IF NOT EXISTS " + name + " (" \
                                                     "id integer NOT NULL GENERATED ALWAYS AS IDENTITY " \
                                                     "( INCREMENT 1 START 1 MINVALUE 1 MAXVALUE 99999999 CACHE 1 )," \
                                                     "x real[] NOT NULL," \
                                                     "PRIMARY KEY (id))"
        self.cur.execute(sql)
        sql1 = "CREATE INDEX IF NOT EXISTS " + name + "_index_id ON " + name + " USING btree (id ASC NULLS LAST);"
        self.cur.execute(sql1)

    def create_table_y(self, name):
        sql = "CREATE TABLE IF NOT EXISTS " + name + " (" \
                                                     "id integer NOT NULL GENERATED ALWAYS AS IDENTITY " \
                                                     "( INCREMENT 1 START 1 MINVALUE 1 MAXVALUE 99999999 CACHE 1 )," \
                                                     "y integer NOT NULL," \
                                                     "PRIMARY KEY (id))"
        self.cur.execute(sql)
        sql1 = "CREATE INDEX IF NOT EXISTS " + name + "_index_id ON " + name + " USING btree (id ASC NULLS LAST);"
        self.cur.execute(sql1)

    def create_table_y_1(self, name):
        sql = "CREATE TABLE IF NOT EXISTS " + name + " (" \
                                                     "id integer NOT NULL GENERATED ALWAYS AS IDENTITY " \
                                                     "( INCREMENT 1 START 1 MINVALUE 1 MAXVALUE 99999999 CACHE 1 )," \
                                                     "y real NOT NULL," \
                                                     "PRIMARY KEY (id))"
        self.cur.execute(sql)
        sql1 = "CREATE INDEX IF NOT EXISTS " + name + "_index_id ON " + name + " USING btree (id ASC NULLS LAST);"
        self.cur.execute(sql1)


def connect_db():
    try:
        conn = psycopg2.connect(dbname='mydb', user='postgres', password='45641146', host='127.0.0.1', port=5432)
    except Exception as e:
        logging.error(e)
    else:
        return conn
    return None


def start():
    conn = connect_db()
    cur = conn.cursor()
    context = PgContext(conn, cur)
    try:
        cur.execute("select create_manager()")
    except Exception as e:
        logging.error(e)
    return context


def end(pl):
    pl.conn.commit()
    pl.conn.close()


def execute_sql(cur, sql, val=None):
    try:
        cur.execute(sql, val)
    except Exception as e:
        logging.error(e)
    else:
        res = cur.fetchone()
        return res[0]
    print(sql)
    return None


def get_shape_str(shape, dim):
    res = '{'
    for i in range(dim - 1):
        res = res + str(shape[i]) + ','
    res = res + str(shape[dim - 1]) + '}'
    return res
