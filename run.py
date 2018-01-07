import numpy as np

from model import DataManager, KMeans, Calculation, PlotGenerator

x, y = DataManager.load_data('data/iris.arff')
#
# PlotGenerator.plot_data_set_2d(x, y)
# PlotGenerator.plot_data_set_3d(x, y)
k_means = KMeans(x, y, 3)
k_means.random_centers()
C, CX = k_means.process(x)
PlotGenerator.plot_clusters_2d(CX, x)
PlotGenerator.plot_clusters_3d(CX, x)
error = Calculation.quantization_error(x, C, CX)
print(f"Clustering error: {round(error,3)}")
