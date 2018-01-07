import random

import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score

from model import Calculation, DataManager, KMeans, PlotGenerator

x, y = DataManager.load_data('data/iris.arff')
# PlotGenerator.data_set_2d(x, y)
# PlotGenerator.data_set_3d(x, y)
k_means = KMeans(x, y, 3, d_function='MAH')
k_means.random_centers()
C, CX = k_means.process(x)
# PlotGenerator.clusters_2d(CX, x)
# PlotGenerator.clusters_3d(CX, x)
error = Calculation.quantization_error(x, C, CX)
CX = Calculation.proper_classes(CX, y)
accuracy = Calculation.accuracy(CX, y)
print("Error: %f" % error)
print("Accuracy: %f" % accuracy)


x = list()
y = list()
for line in open("data/banknote_authentication_set.txt"):
    tmp_array = line.split(',')
    x.append([float(i) for i in tmp_array[0:-1]])
    y.append(float(tmp_array[-1].replace('\n', '')))

# PlotGenerator.data_set_2d(x, y)
PlotGenerator.data_set_3d(x, y)
k_means = KMeans(x, y, 3, d_function="MAH")
k_means.random_centers()
C, CX = k_means.process(x)
# PlotGenerator.clusters_2d(CX, x)
PlotGenerator.clusters_3d(CX, x)
error = Calculation.quantization_error(x, C, CX)
CX = Calculation.proper_classes(CX, y)
accuracy = Calculation.accuracy(CX, y)
print("Error: %f" % error)
print("Accuracy: %f" % accuracy)


x = list()
y = list()
for i in range(1000):
    x.append(np.random.random_sample(3,).tolist())
    y.append(float(random.randint(0, 3)))

# PlotGenerator.data_set_2d(x, y)
# PlotGenerator.data_set_3d(x, y)
k_means = KMeans(x, y, 3, d_function="MAH")
k_means.random_centers()
C, CX = k_means.process(x)
# PlotGenerator.clusters_2d(CX, x)
# PlotGenerator.clusters_3d(CX, x)
error = Calculation.quantization_error(x, C, CX)
CX = Calculation.proper_classes(CX, y)
accuracy = Calculation.accuracy(CX, y)
print("Error: %f" % error)
print("Accuracy: %f" % accuracy)


x = list()
y = list()
for i in range(1000):
    x.append(np.random.random_sample(3,).tolist())
    y.append(float(random.randint(0, 3)))
length = len(x)
time = Calculation.timer(x)
x = np.array([i / 20 * length for i in range(1, 20)])
y = time
model = linear_model.LinearRegression(fit_intercept=True)
model.fit(x[:, np.newaxis], y)
x_fit = np.linspace(0, length, 1000)
y_fit = model.predict(x_fit[:, np.newaxis])
y_pred = model.predict(x[:, np.newaxis])
r_2 = r2_score(y, y_pred)
print("Współczytnnik determinacji: %f" % r_2)
