import random
from matplotlib import pyplot
import numpy
import percep
from preprocess import data_X, X_test, X_train, y_train, y_test

def speed_check():
    perpy = percep.Perceptron()
    perpy.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.001, batch_size=32)
    perpy_2 = percep.Perceptron()
    perpy_2.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.01, batch_size=32)
    perpy_3 = percep.Perceptron()
    perpy_3.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)
    perpy_4 = percep.Perceptron()
    perpy_4.fit(X_train, y_train, X_test, y_test, epochs=100, lr=1, batch_size=32)
    perpy.accuracy(X_train, y_train, X_test, y_test)
    perpy_2.accuracy(X_train, y_train, X_test, y_test)
    perpy_3.accuracy(X_train, y_train, X_test, y_test)
    perpy_4.accuracy(X_train, y_train, X_test, y_test)
    x_values = numpy.linspace(0, 100, 100)
    pyplot.plot(x_values, perpy.L_1, label="обучающая выборка 1")
    pyplot.plot(x_values, perpy.L_2, label="тестовая выборка 1")
    pyplot.plot(x_values, perpy_2.L_1, label="обучающая выборка 2")
    pyplot.plot(x_values, perpy_2.L_2, label="тестовая выборка 2")
    pyplot.plot(x_values, perpy_3.L_1, label="обучающая выборка 3")
    pyplot.plot(x_values, perpy_3.L_2, label="тестовая выборка 3")
    pyplot.plot(x_values, perpy_4.L_1, label="обучающая выборка 4")
    pyplot.plot(x_values, perpy_4.L_2, label="тестовая выборка 4")
    pyplot.title('Функция потерь в зависимости от эпохи и learning rate')
    pyplot.legend()
    pyplot.show()
