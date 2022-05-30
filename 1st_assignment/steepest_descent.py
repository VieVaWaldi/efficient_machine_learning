import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import time


fig = plt.gcf()
pause = 0.01


def start_plt():
    fig.show()
    fig.canvas.draw()
    plt.pause(pause)


def start_plt_3d(X, Y, Z):
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=pl.cm.RdBu)
    fig.colorbar(surf, shrink=0.5, aspect=2)


def update_plt():
    plt.pause(pause)
    fig.canvas.draw()


def st(x):
    return str(round(x, 3))

############################################################### Quadratic fn ###


def fn_quad(x):
    f = x**2
    g = 2*x
    return f, g

################################################################### Rosen fn ###


def fn_rosen(x):
    f = 100*(x[1] - x[0]**2)**2 + (x[0] - 1)**2
    g = np.array([
        - 400 * x[0] * (x[1] - x[0]**2) + (x[0] - 1),  # deriv x
        200 * (x[1] - x[0]**2) + (x[0] - 1)  # deriv y
    ])
    return f, g

################################################################ Griewank fn ###


def fn_griewank(x):

    sum = (x[0]**2)/4000 + (x[1]**2)/4000
    prod = np.cos(x[0]/np.sqrt(1)) * np.cos(x[1]/np.sqrt(2))
    f = 1 + sum - prod

    return f

    # if isinstance(x, np.ndarray):
    #     y = []
    #     for i in x:
    #         sum = i**2/4000
    #         prod = math.cos(i/math.sqrt(1))
    #         y.append(1 + sum - prod)
    #     return y


def fn_griewank_2d(x):

    sum = x**2/4000
    prod = map(math.cos, x)
    f = 1 + sum - prod

    sum_d = (4000*2*x - x**2) / 4000**2
    prod_d = -math.sin(x/math.sqrt(1))
    g = sum_d - prod_d

    return f, g


def fn_griewank_dim(x, dim):
    sum = 0
    prod = 1
    for i in range(dim):
        sum += x[i]**2
        prod *= math.cos(float(x[i]) / math.sqrt(i+1))
    return 1 + (float(sum)/4000.0) - float(prod)

###################################################### backtrack line search ###


def back_linesearch_2d(x, fn):
    """
        Hilft alpha (schrittweite) iterativ zu bestimmen.
        https://www.youtube.com/watch?v=4qDt4QUl4zE
    """
    alpha = 1

    a = fn(x - alpha * fn(x)[1])[0]
    b = fn(x)[0] - 0.5*alpha * fn(x)[1]**2

    while a > b:
        alpha = 0.5 * alpha

        a = fn(x - alpha * fn(x)[1])[0]
        b = fn(x)[0] - 0.5*alpha * fn(x)[1]**2

        plt.plot(b, fn(b)[0], 'ro')
        update_plt()
    return alpha


def back_linesearch_3d(x, fn):
    alpha = 1

    a = fn(x - alpha * fn(x)[1])[0]
    b = fn(x)[0] - 0.5*alpha * fn(x)[1]**2

    while a > b[0] and a > b[1]:
        alpha = 0.5 * alpha

        a = fn(x - alpha * fn(x)[1])[0]
        b = fn(x)[0] - 0.5*alpha * fn(x)[1]**2

    return alpha

########################################################### Steepest descent ###


def steepest_descent_2d(fn, alpha=0.1, tol=0.01, maxiter=100, is_back_line_on=False):

    x = random.randint(-300, 300)

    start_plt()
    points = np.arange(-300, 300, 0.1)
    plt.plot(points, fn(points)[0])
    plt.plot(x, fn(x)[0], 'go')
    update_plt()

    for k in range(maxiter):

        f, g = fn(x)

        if abs(g) < tol:
            break

        if is_back_line_on:
            alpha = back_linesearch_2d(x, fn)

        x = x + alpha * -g

        plt.title('pt= ' + st(x) + '\ng= ' + st(g))
        plt.plot(x, fn(x)[0], 'go')
        update_plt()


def steepest_descent_3d(fn, alpha=0.01, tol=0.01, maxiter=100, is_back_line_on=False):

    x = np.array([random.randint(-5, 5), random.randint(-5, 5)])
    # x = np.array([5, -5])

    points = np.arange(-5, 5, 0.4)
    X, Y = pl.meshgrid(points, points)
    Z = fn([X, Y])

    start_plt_3d(X, Y, Z[0])
    plt.plot(x[0], x[1], fn(x)[0], 'go', zorder=4)

    for k in range(maxiter):

        f, g = fn(x)

        if abs(g[0]) < tol and abs(g[1]) < tol:
            break

        if is_back_line_on:
            alpha = back_linesearch_3d(x, fn)

        x = x + alpha * -g

        plt.title('x= ' + st(x[0]) + ' | ' + st(x[1]) +
                  '\ng= ' + st(g[0]) + ' | ' + st(g[1]))
        plt.plot(x[0], x[1], fn(x)[0], 'go', zorder=4, markersize=2)
        update_plt()


# steepest_descent_2d(fn_quad, alpha=1.2, tol=0.01,
#                     maxiter=100, is_back_line_on=True)

steepest_descent_2d(fn_griewank, alpha=0.00001,
                    maxiter=200, is_back_line_on=False)


# steepest_descent_3d(fn_rosen, alpha=0.00001,
#                     maxiter=200, is_back_line_on=False)

steepest_descent_3d(fn_griewank, alpha=0.00001,
                    maxiter=200, is_back_line_on=False)

time.sleep(10)
