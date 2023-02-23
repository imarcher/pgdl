from Conv_mnist import mnist_time, mnist_time_tf
from Linear_usa_housing import usa_housing_time, usa_housing_time_tf
from LR_epsilon import epsilon_time, epsilon_time_tf
from Vgg16_cifa10 import cifa10_time, cifa10_time_tf
from line_profiler import LineProfiler
import datetime


# lp = LineProfiler()
# lp_wrapper = lp(hanshu)
# lp_wrapper()
# lp.print_stats()

def get_res(name, batch, epochs, t1, t2):
    res = '\033[33;1m' + name + ':\033[0m' + '\n' \
                                             'batch: ' + str(batch) + '\n' \
                                                                      'epochs: ' + str(epochs) + '\n' \
                                                                                                 'pgdl: ' + str(
        t1) + 's\n' \
              'keras: ' + str(t2) + 's'
    return res


def get_run_time(func, *args, **kwds):
    starttime = datetime.datetime.now()
    func(*args, **kwds)
    endtime = datetime.datetime.now()
    return (endtime - starttime).microseconds / 1000000 + (endtime - starttime).seconds


def usa_housing(batch, epochs):
    t1 = get_run_time(usa_housing_time.run, batch, epochs)
    t2 = get_run_time(usa_housing_time_tf.run, batch, epochs)
    return get_res('usa_housing', batch, epochs, t1, t2)


def epsilon(batch, epochs):
    t1 = get_run_time(epsilon_time.run, batch, epochs)
    t2 = get_run_time(epsilon_time_tf.run, batch, epochs)
    return get_res('epsilon', batch, epochs, t1, t2)


def mnist(batch, epochs):
    t1 = get_run_time(mnist_time.run, batch, epochs)
    t2 = get_run_time(mnist_time_tf.run, batch, epochs)
    return get_res('mnist', batch, epochs, t1, t2)


def cifa10(batch, epochs):
    t1 = get_run_time(cifa10_time.run, batch, epochs)
    t2 = get_run_time(cifa10_time_tf.run, batch, epochs)
    return get_res('cifa10', batch, epochs, t1, t2)


if __name__ == '__main__':
    times = []
    times.append(usa_housing(32, 10))
    times.append(epsilon(32, 1))
    times.append(mnist(128, 1))
    times.append(cifa10(128, 1))
    for i in range(len(times)):
        print(times[i])
