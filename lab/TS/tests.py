import numpy as np
response = lambda x1, x2: 5*x1 + .1*x2 + 3


if __name__ == '__main__':
    min_x1, min_x2, max_x1, max_x2 = -10, -10, 10, 10
    xv1, xv2 = np.meshgrid(np.linspace(min_x1, max_x1, 10), np.linspace(min_x2, max_x2, 10))
    surface = response(xv1, xv2)
    x = np.random.uniform((min_x1, min_x2), (max_x1, max_x2), (10, 2))
    y_ = response(x[:, 0], x[:, 1])
    response(x)

