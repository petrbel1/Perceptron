import sklearn
import random
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import math
import numpy

data_X, data_y = sklearn.datasets.make_classification(
    n_samples=500,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_classes=2,
    random_state=42,
    n_clusters_per_class=1
)

scaler = sklearn.preprocessing.StandardScaler()

data_X = scaler.fit_transform(data_X)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    data_X, data_y,
    test_size=0.3,
    stratify=data_y,
    random_state=42
)

class Perceptron:
    #weights = []

    def __init__(self):
        self.weights = []
        self.weights.append(random.uniform(-1, 1))
        self.weights.append(random.uniform(-1, 1))
        self.weights.append(0)
        self.L_1 = []
        self.L_2 = []

    #def make_weights(self):

    def sigmoid(self, x):
        x = max(min(x, 500), -500)
        return 1 / (1 + math.exp(-x))

    def forward(self, data):
        results = []
        for x in data:
            res = self.weights[0] * x[0] + self.weights[1] * x[1] + self.weights[2]
            res = self.sigmoid(res)
            results.append(res)
        return results

    def compute_loss(self, y_true, y_pred):
        L = 0
        eps = 1e-15
        for i in range(len(y_true)):
            L -= (y_true[i] * math.log(y_pred[i] + eps) + (1 - y_true[i]) * math.log(1 - y_pred[i] + eps))
        return L * (1 / len(y_true))

    def compute_gradient(self, X_input, y_true, y_pred):
        result = [0, 0, 0]
        for i in range(len(y_true)):
            result[0] += (y_pred[i] - y_true[i]) * X_input[i][0]
            result[1] += (y_pred[i] - y_true[i]) * X_input[i][1]
            result[2] += (y_pred[i] - y_true[i])
        #for j in range(len(result)):
        #    result[j] /= len(y_true)
        return result

    def accuracy(self, X_training, y_training, X_validation, y_validation):
        acc1 = 0
        acc2 = 0
        predictions = self.forward(X_training)
        predictions2 = self.forward(X_validation)
        predictions = self.predict(predictions)
        predictions2 = self.predict(predictions2)
        for i in range(len(predictions)):
            if predictions[i] == y_training[i]:
                acc1 += 1
        for i in range(len(predictions2)):
            if predictions2[i] == y_validation[i]:
                acc2 += 1
        print("Accuracy on learning part: ")
        print(acc1 / len(predictions))
        print("  Accuracy on testing part: ")
        print(acc2 / len(predictions2))

    def fit(self, X_training, y_training, X_validation, y_validation, epochs, lr, batch_size):
        #self.make_weights()
        amount = len(X_training) // batch_size
        X_t_batched = numpy.array_split(X_training, amount)
        y_t_batched = numpy.array_split(y_training, amount)
        X_v_batched = numpy.array_split(X_validation, amount)
        y_v_batched = numpy.array_split(y_validation, amount)
        for epoch in range(epochs):
            for X_t, y_t in zip(X_t_batched, y_t_batched):
                predictions = self.forward(X_t)
                gradient = self.compute_gradient(X_t, y_t, predictions)
                self.weights[0] = self.weights[0] - lr * gradient[0] / len(X_training)
                self.weights[1] = self.weights[1] - lr * gradient[1] / len(X_training)
                self.weights[2] = self.weights[2] - lr * gradient[2] / len(X_training)
            print("Epoch " + str(epoch) + "\n")
            self.L_1.append(self.compute_loss(y_training, self.forward(X_training)))
            print(self.L_1[epoch])
            print("\n")
            self.L_2.append(self.compute_loss(y_validation, self.forward(X_validation)))
            print(self.L_2[epoch])
            print("\n")
            indices = numpy.random.permutation(len(X_training))
            X_training = X_training[indices]
            y_training = y_training[indices]

    def predict(self, data):
        final_data = []
        for i in data:
            if i >= 0.5:
                final_data.append(1)
            else:
                final_data.append(0)
        return final_data

def base():
    perpy = Perceptron()
    perpy.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)
    perpy.accuracy(X_train, y_train, X_test, y_test)
    x_values = numpy.linspace(0, 100, 100)
    x_demo = []
    y_demo = []
    for i in range(len(data_X)):
        x_demo.append(data_X[i][0])
        y_demo.append((data_X[i][0] * perpy.weights[0] + perpy.weights[2]) / -perpy.weights[1])
    mpl.pyplot.figure(1)
    mpl.pyplot.plot(x_values, perpy.L_1, label="обучающая выборка")
    mpl.pyplot.plot(x_values, perpy.L_2, label="тестовая выборка")
    mpl.pyplot.title('Функция потерь в зависимости от эпохи')
    mpl.pyplot.figure(2)
    t_x, t_y = zip(*data_X)
    mpl.pyplot.scatter(t_x, t_y, label="tochki", s=20)
    mpl.pyplot.plot(x_demo, y_demo, color="red", label="линия XtW + b = 0")
    mpl.pyplot.title('линия XtW + b = 0 и данные')
    mpl.pyplot.legend()
    mpl.pyplot.show()

def speed_check():
    perpy = Perceptron()
    perpy.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.001, batch_size=32)
    perpy_2 = Perceptron()
    perpy_2.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.01, batch_size=32)
    perpy_3 = Perceptron()
    perpy_3.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)
    perpy_4 = Perceptron()
    perpy_4.fit(X_train, y_train, X_test, y_test, epochs=100, lr=1, batch_size=32)
    perpy.accuracy(X_train, y_train, X_test, y_test)
    perpy_2.accuracy(X_train, y_train, X_test, y_test)
    perpy_3.accuracy(X_train, y_train, X_test, y_test)
    perpy_4.accuracy(X_train, y_train, X_test, y_test)
    x_values = numpy.linspace(0, 100, 100)
    mpl.pyplot.plot(x_values, perpy.L_1, label="обучающая выборка 1")
    mpl.pyplot.plot(x_values, perpy.L_2, label="тестовая выборка 1")
    mpl.pyplot.plot(x_values, perpy_2.L_1, label="обучающая выборка 2")
    mpl.pyplot.plot(x_values, perpy_2.L_2, label="тестовая выборка 2")
    mpl.pyplot.plot(x_values, perpy_3.L_1, label="обучающая выборка 3")
    mpl.pyplot.plot(x_values, perpy_3.L_2, label="тестовая выборка 3")
    mpl.pyplot.plot(x_values, perpy_4.L_1, label="обучающая выборка 4")
    mpl.pyplot.plot(x_values, perpy_4.L_2, label="тестовая выборка 4")
    mpl.pyplot.title('Функция потерь в зависимости от эпохи и learning rate')
    mpl.pyplot.legend()
    mpl.pyplot.show()

def weights_change():
    perpy = Perceptron()
    perpy.weights = [0, 0, 0]
    perpy.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)
    perpy_2 = Perceptron()
    perpy_2.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)
    perpy_3 = Perceptron()
    perpy_3.weights = [random.uniform(1, 10), random.uniform(1, 10), random.uniform(1, 10)]
    perpy_3.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)
    perpy.accuracy(X_train, y_train, X_test, y_test)
    perpy_2.accuracy(X_train, y_train, X_test, y_test)
    perpy_3.accuracy(X_train, y_train, X_test, y_test)
    x_values = numpy.linspace(0, 100, 100)
    mpl.pyplot.plot(x_values, perpy.L_1, label="обучающая выборка 1")
    mpl.pyplot.plot(x_values, perpy.L_2, label="тестовая выборка 1")
    mpl.pyplot.plot(x_values, perpy_2.L_1, label="обучающая выборка 2")
    mpl.pyplot.plot(x_values, perpy_2.L_2, label="тестовая выборка 2")
    mpl.pyplot.plot(x_values, perpy_3.L_1, label="обучающая выборка 3")
    mpl.pyplot.plot(x_values, perpy_3.L_2, label="тестовая выборка 3")
    mpl.pyplot.title('Функция потерь в зависимости от эпохи и weights')
    mpl.pyplot.legend()
    mpl.pyplot.show()

def batch_size():
    perpy = Perceptron()
    perpy.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=1)
    perpy_2 = Perceptron()
    perpy_2.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=16)
    perpy_3 = Perceptron()
    perpy_3.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=64)
    perpy_4 = Perceptron()
    perpy_4.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=256)
    perpy.accuracy(X_train, y_train, X_test, y_test)
    perpy_2.accuracy(X_train, y_train, X_test, y_test)
    perpy_3.accuracy(X_train, y_train, X_test, y_test)
    perpy_4.accuracy(X_train, y_train, X_test, y_test)
    x_values = numpy.linspace(0, 100, 100)
    mpl.pyplot.plot(x_values, perpy.L_1, label="обучающая выборка 1")
    mpl.pyplot.plot(x_values, perpy.L_2, label="тестовая выборка 1")
    mpl.pyplot.plot(x_values, perpy_2.L_1, label="обучающая выборка 2")
    mpl.pyplot.plot(x_values, perpy_2.L_2, label="тестовая выборка 2")
    mpl.pyplot.plot(x_values, perpy_3.L_1, label="обучающая выборка 3")
    mpl.pyplot.plot(x_values, perpy_3.L_2, label="тестовая выборка 3")
    mpl.pyplot.plot(x_values, perpy_4.L_1, label="обучающая выборка 4")
    mpl.pyplot.plot(x_values, perpy_4.L_2, label="тестовая выборка 4")
    mpl.pyplot.title('Функция потерь в зависимости от эпохи и batch size')
    mpl.pyplot.legend()
    mpl.pyplot.show()

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