import numpy as np


class DataManager:
    @staticmethod
    def dp(x_vector, c_vector, p):
        result_vector = abs(np.array(x_vector) - np.array(c_vector))
        powered = np.power(result_vector, p)
        return np.sum(powered) ** (1. / p)

    @staticmethod
    def dm(x_vector, c_vector, a_matrix):
        x_sub_c = abs(np.array(x_vector) - np.array(c_vector))
        x_sub_c_t = np.transpose(x_sub_c)
        return np.sum(x_sub_c * a_matrix ** (-1) * x_sub_c_t) ** (1. / 2.)

    @staticmethod
    def k_means(x_vector, k):
        pass

    @staticmethod
    def quantization_error(x_vector, c_vector):
        pass
