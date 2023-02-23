import psycopg
import logging
from enum import IntEnum


class Tag(IntEnum):
    ab = 0
    a = 1
    nhwc = 2
    nchw = 3


class PgContext:
    def __init__(self, conn, cur):
        self.conn = conn
        self.cur = cur

    def create_placeholder_node(self, shape, dim, format_tag):
        shape = str(shape)
        format_tag = int(format_tag)
        return execute_sql(self.cur, "select create_placeholder", (shape, dim, format_tag))

    def create_variable_node(self, shape, dim, format_tag):
        format_tag = int(format_tag)
        return execute_sql(self.cur, "select create_variable", (shape, dim, format_tag))

    def create_reverse_node(self, node1):
        return execute_sql(self.cur, "select c_reverse_op", (node1,))

    def create_add_node(self, node1, node2):
        return execute_sql(self.cur, "select c_add_op", (node1, node2))

    def create_sub_node(self, node1, node2):
        return execute_sql(self.cur, "select c_sub_op", (node1, node2))

    def create_matmul_node(self, node1, node2, node3):
        return execute_sql(self.cur, "select c_matmul_op", (node1, node2, node3))

    def softmax_cross_entropy_training(self, x, y, lr, execute_num):
        return execute_sql(self.cur, "select softmax_cross_entropy_training", (x, y, lr, execute_num))

    def softmax_cross_entropy_inference(self, x, y, execute_num):
        return execute_sql(self.cur, "select softmax_cross_entropy_inference", (x, y, execute_num))

    def softmax_cross_entropy_accuracy(self, x, y, execute_num):
        return execute_sql(self.cur, "select softmax_cross_entropy_accuracy", (x, y, execute_num))

    def softmax_cross_entropy_training_accuracy(self, x, y, lr, execute_num):
        return execute_sql(self.cur, "select softmax_cross_entropy_training_accuracy"
                           , (x, y, lr, execute_num))

    def read_to_node_scalar(self, column, table, start_id, all_num, node, execute_id):
        return execute_sql(self.cur, "select read_to_node_scalar"
                           , (column, table, start_id, all_num, node, execute_id))

    def read_to_node_arr(self, column, table, start_id, all_num, node, execute_id):
        return execute_sql(self.cur, "select read_to_node_arr"
                           , (column, table, start_id, all_num, node, execute_id))

    def init_variables(self):
        return execute_sql(self.cur, "select init_variables()")

    def prepare_memory(self):
        return execute_sql(self.cur, "select prepare_memory()")

    def compute(self):
        return execute_sql(self.cur, "select compute()")

    def compute_ex(self):
        return execute_sql(self.cur, "select compute_ex()")

    def wait(self):
        return execute_sql(self.cur, "select wait()")

    def get_loss(self):
        return execute_sql(self.cur, "select get_loss()")

    def get_accuracy(self):
        return execute_sql(self.cur, "select get_accuracy()")



def connect_db():
    try:
        conn = psycopg.connect(dbname='mydb', user='postgres', password='45641146', host='127.0.0.1', port=5432)
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
    if val is not None:
        sql = sql + str(val)
    try:
        cur.execute(sql)
    except Exception as e:
        logging.error(e)
        print(sql)
        return None
    print(sql)
    res = cur.fetchone()
    return res[0]

