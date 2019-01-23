import numpy as np


def gaussian_xy(x_limit, y_limit):
    """
    generate random (x,y) coordinates using Gaussian distribution;
    the mean of the Gaussian if set to the middle of the max - min range;
    the std of the Gaussian is set to a quarter of the max - min range;
    rejection sampling is used if the generated coordinates exceed either limit
    :param x_limit: a tuple (x_min, x_max) indicating the range in x-axis
    :param y_limit: a tuple (y_min, y_max) indicating the range in y-axis
    :return: a tuple (x,y)
    """
    x_min, x_max = x_limit
    y_min, y_max = y_limit
    assert x_min <= x_max, 'x_min <= x_max (%.4f <= %.4f)' % (x_min, x_max)
    assert y_min <= y_max, 'y_min <= y_max (%.4f <= %.4f)' % (y_min, y_max)

    x = np.random.randn() * (x_max - x_min) / 4 + (x_max + x_min) / 2
    y = np.random.randn() * (y_max - y_min) / 4 + (y_max + y_min) / 2
    if (not x_min <= x <= x_max) or (not y_min <= y <= y_max):
        x, y = gaussian_xy(x_limit, y_limit)
    return x, y


def uniform_xy(x_limit, y_limit):
    """
    generate random (x,y) coordinates using Uniform distribution
    :param x_limit: a tuple (x_min, x_max) indicating the range in x-axis
    :param y_limit: a tuple (y_min, y_max) indicating the range in y-axis
    :return: a tuple (x,y)
    """
    x_min, x_max = x_limit
    y_min, y_max = y_limit
    x = np.random.rand() * (x_max - x_min) + x_min
    y = np.random.rand() * (y_max - y_min) + y_min
    return x, y


def test_gaussian_xy(n):
    def test_and_print(x_limit, y_limit, n):
        x, y = np.zeros(n), np.zeros(n)
        try:
            for i in range(n):
                x[i], y[i] = gaussian_xy(x_limit, y_limit)
            x_ep_max, x_ep_min = np.amax(x), np.amin(x)
            y_ep_max, y_ep_min = np.amax(y), np.amin(y)
            x_ep_mean, x_ep_std = np.mean(x), np.std(x)
            y_ep_mean, y_ep_std = np.mean(y), np.std(y)
            print('Input', x_limit, y_limit, 'Output',
                  (x_ep_min, x_ep_max, x_ep_mean, x_ep_std),
                  (y_ep_min, y_ep_max, y_ep_mean, y_ep_std))
        except AssertionError as e:
            print('Input', x_limit, y_limit, 'Output', e)

    print('Test Gaussian XY')
    test_and_print((0, 10), (0, 10), n)
    test_and_print((-10, 10), (-10, 10), n)
    test_and_print((0, 0), (-10, 10), n)
    test_and_print((-10, 10), (0, 0), n)
    test_and_print((10, -10), (0, 0), n)
    test_and_print((0, 0), (10, -10), n)


def test_uniform_xy(n):
    def test_and_print(x_limit, y_limit, n):
        x, y = np.zeros(n), np.zeros(n)
        for i in range(n):
            x[i], y[i] = uniform_xy(x_limit, y_limit)
        x_ep_max, x_ep_min = np.amax(x), np.amin(x)
        y_ep_max, y_ep_min = np.amax(y), np.amin(y)
        print('Input', x_limit, y_limit, 'Output',
              (x_ep_min, x_ep_max), (y_ep_min, y_ep_max))

    print('Test Uniform XY')
    test_and_print((0, 10), (0, 10), n)
    test_and_print((-10, 10), (-10, 10), n)
    test_and_print((0, 0), (-10, 10), n)
    test_and_print((-10, 10), (0, 0), n)
    test_and_print((10, -10), (0, 0), n)
    test_and_print((0, 0), (10, -10), n)


if __name__ == '__main__':
    test_uniform_xy(1000)
    test_gaussian_xy(1000)
