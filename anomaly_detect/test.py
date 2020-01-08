import random
from anomaly_detect import IsolationForest

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


SIZE = 200
A_PCT = 0.1


def generate_data(data_size, anomaly_size):
    max_radius = 5.0
    r = np.random.uniform(0.1, max_radius, data_size)
    for i in range(int(anomaly_size * data_size)):
        r[i] = random.uniform(7.0, 10.0)

    theta = np.random.uniform(0, 2*np.pi, data_size)

    x = np.cos(theta) * r
    y = np.sin(theta) * r
    data_set = list(zip(x, y))
    return data_set


def plot(title, data_set, result):
    color = np.array(['blue'] * len(data_set))
    color[result] = 'red'
    x, y = zip(*data_set)
    plt.figure()
    plt.scatter(x, y, c=color)
    plt.xlabel('x')
    plt.ylabel('y')
    circle = plt.Circle((0, 0), 5.1, color='r', fill=False)
    plt.gcf().gca().add_artist(circle)
    plt.axis('equal')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.title(title)


def test_plot(title):
    def decorator(func):
        def wrapper(*args, **kw):
            return plot(title, *func(*args, **kw))
        return wrapper
    return decorator


@test_plot('iforest')
def test_iforest(test_set):
    anomaly = IsolationForest().decision(test_set, threshold=0.55)
    return test_set, anomaly


if __name__ == '__main__':
    data_set = generate_data(SIZE, A_PCT)
    test_iforest(data_set)
    plt.show()