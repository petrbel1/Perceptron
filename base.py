from matplotlib import pyplot
import numpy
import percep
from preprocess import data_X, X_test, X_train, y_train, y_test

def base():
    perpy = percep.Perceptron()
    perpy.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)
    perpy.accuracy(X_train, y_train, X_test, y_test)
    x_values = numpy.linspace(0, 100, 100)
    x_demo = []
    y_demo = []
    for i in range(len(data_X)):
        x_demo.append(data_X[i][0])
        y_demo.append((data_X[i][0] * perpy.weights[0] + perpy.weights[2]) / -perpy.weights[1])
    pyplot.figure(1)
    pyplot.plot(x_values, perpy.L_1, label="обучающая выборка")
    pyplot.plot(x_values, perpy.L_2, label="тестовая выборка")
    pyplot.title('Функция потерь в зависимости от эпохи')
    pyplot.figure(2)
    t_x, t_y = zip(*data_X)
    pyplot.scatter(t_x, t_y, label="tochki", s=20)
    pyplot.plot(x_demo, y_demo, color="red", label="линия XtW + b = 0")
    pyplot.title('линия XtW + b = 0 и данные')
    pyplot.legend()
    pyplot.show()