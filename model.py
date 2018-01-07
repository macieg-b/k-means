import math
import random

import numpy as np
import randomcolor
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import inv
from scipy.io import arff


class KMeans:
    def __init__(self, x, y, k, p=2, d_function='EUC'):
        self.x = x
        self.y = y
        self.k = k
        self.p = p
        self.d_function = d_function
        self.initial_centers = ''

    def random_centers(self):
        random_indexes = np.random.randint(low=1, high=len(self.x), size=self.k)
        self.initial_centers = list([self.x[i] for i in random_indexes])

    def process(self, x):
        c = self.initial_centers

        while True:
            if self.d_function == "EUC":
                dist = Calculation.dp(x, c, self.p)
            elif self.d_function == "MAH":
                dist = Calculation.dm(x, c)

            c_x = {}
            for i in range(self.k):
                c_x[i] = []

            for i in range(len(x)):
                new_distribution = dist[0][i]
                new_center = 0
                for old_center in range(1, self.k):
                    if dist[old_center][i] < new_distribution:
                        new_center = old_center
                        new_distribution = dist[old_center][i]
                c_x[new_center].append(i)
            c_new = []

            for old_center in range(self.k):
                new_center = []
                for dim in range(len(x[0])):
                    new_center.append(np.mean([x[i][dim] for i in c_x[old_center]]))
                c_new.append(new_center)

            if c_new == c:
                return c, c_x
            else:
                c = c_new


class Calculation:
    @staticmethod
    def dp(x_vector, c_vector, p):
        result = {}
        for index in range(len(c_vector)):
            result[index] = []
            for vec in x_vector:
                temp = [pow((abs(vec[i] - c_vector[index][i])), p) for i in range(len(c_vector[0]))]
                dist = pow(sum(temp), 1 / p)
                result[index].append(dist)
        return result

    @staticmethod
    def dm(x_vector, c_vector):
        result = {}
        a_matrix = np.cov(np.array(x_vector).transpose())
        for index in range(len(c_vector)):
            result[index] = []
            for vec in x_vector:
                x_c = np.array(vec) - np.array(c_vector[index])
                dist = math.sqrt(np.dot(np.dot(x_c, inv(a_matrix)), x_c.transpose()))
                result[index].append(dist)
        return result

    @staticmethod
    def quantization_error(x_vector, c_vector, cx_vector, d_function='EUC', p=2):
        error_sum = 0

        if d_function == "EUC":
            d = Calculation.dp(x_vector, c_vector, p)
        elif d_function == "MAH":
            d = Calculation.dm(x_vector, c_vector)

        for i in range(len(c_vector)):
            for j in cx_vector[i]:
                error_sum += d[i][j]
        return error_sum


class DataManager:
    def __init__(self):
        pass

    @staticmethod
    def load_data(file_path):
        data, meta = arff.loadarff(file_path)
        x_data = []
        y_data = []
        for w in range(len(data)):
            x_data.append([])
            for k in range(len(data[0])):
                if k == (len(data[0]) - 1):
                    y_data.append(data[w][k])
                else:
                    x_data[w].append(data[w][k])
        return x_data, y_data

    @staticmethod
    def split_data(x, y, ratio):
        random.shuffle(x)
        teach_size = int(len(x) * ratio)
        return x[:teach_size], x[teach_size:], y[:teach_size], y[teach_size:]


class PlotGenerator:
    BLUE = "#7161ef"
    GREEN = "#00fa9a"
    RED = "#c8515f"

    @staticmethod
    def plot_clusters_2d(cx_vector, x_vector):
        rand_color = randomcolor.RandomColor()
        for cluster in range(len(cx_vector)):
            color = rand_color.generate()[0]
            for i in cx_vector[cluster]:
                plt.scatter(x_vector[i][0], x_vector[i][1], s=4, c=color)

        plt.title("Rzut na 2 zmienne losowe - algorytm k-środków")
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()

    @staticmethod
    def plot_clusters_3d(cx_vector, x_vector):
        fig = plt.figure()
        ax = Axes3D(fig)
        rand_color = randomcolor.RandomColor()
        for cluster in range(len(cx_vector)):
            color = rand_color.generate()[0]
            for i in cx_vector[cluster]:
                ax.scatter(x_vector[i][0], x_vector[i][1], x_vector[i][2], s=4, c=color)

        plt.title("Rzut na 3 zmienne losowe - algorytm k-środków")
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('x3')
        ax.zaxis.set_rotate_label(True)
        ax.yaxis.set_rotate_label(True)
        plt.show()

    @staticmethod
    def plot_data_set_2d(x, y):
        classes = list(set(y))
        rand_color = randomcolor.RandomColor()
        colors = rand_color.generate(count=len(classes))
        for i in range(len(x)):
            idx = classes.index(y[i])
            plt.scatter(x[i][0], x[i][1], s=4, c=colors[idx])

        plt.title("Rzut na dwie zmienne losowe")
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()

    @staticmethod
    def plot_data_set_3d(x, y):
        fig = plt.figure()
        ax = Axes3D(fig)
        classes = list(set(y))
        rand_color = randomcolor.RandomColor()
        colors = rand_color.generate(count=len(classes))
        for i in range(len(x)):
            idx = classes.index(y[i])
            ax.scatter(x[i][0], x[i][1], x[i][2], s=4, c=colors[idx])

        plt.title("Rzut na 3 zmienne losowe")
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('x3')
        plt.show()
