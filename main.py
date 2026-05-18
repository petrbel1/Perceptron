import random
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

def weights_change():
    perpy = percep.Perceptron()
    perpy.weights = [0, 0, 0]
    perpy.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)
    perpy_2 = percep.Perceptron()
    perpy_2.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)
    perpy_3 = percep.Perceptron()
    perpy_3.weights = [random.uniform(1, 10), random.uniform(1, 10), random.uniform(1, 10)]
    perpy_3.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)
    perpy.accuracy(X_train, y_train, X_test, y_test)
    perpy_2.accuracy(X_train, y_train, X_test, y_test)
    perpy_3.accuracy(X_train, y_train, X_test, y_test)
    x_values = numpy.linspace(0, 100, 100)
    pyplot.plot(x_values, perpy.L_1, label="обучающая выборка 1")
    pyplot.plot(x_values, perpy.L_2, label="тестовая выборка 1")
    pyplot.plot(x_values, perpy_2.L_1, label="обучающая выборка 2")
    pyplot.plot(x_values, perpy_2.L_2, label="тестовая выборка 2")
    pyplot.plot(x_values, perpy_3.L_1, label="обучающая выборка 3")
    pyplot.plot(x_values, perpy_3.L_2, label="тестовая выборка 3")
    pyplot.title('Функция потерь в зависимости от эпохи и weights')
    pyplot.legend()
    pyplot.show()

def batch_size():
    perpy = percep.Perceptron()
    perpy.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=1)
    perpy_2 = percep.Perceptron()
    perpy_2.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=16)
    perpy_3 = percep.Perceptron()
    perpy_3.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=64)
    perpy_4 = percep.Perceptron()
    perpy_4.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=256)
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
    pyplot.title('Функция потерь в зависимости от эпохи и batch size')
    pyplot.legend()
    pyplot.show()

choice = int(input("выберите цифру. 1 - базовая работа, 2 - изучение learning rate, 3 - изучение batc size, 4 - изучение weights"))
if choice == 1:
    base()
elif choice == 2:
    speed_check()
elif choice == 3:
    batch_size()
elif choice == 4:
     weights_change()
else:
    print("Неверный ввод")