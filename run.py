from model import DataManager, KMeans, Calculation, PlotGenerator

x, y = DataManager.load_data('data/iris.arff')

PlotGenerator.data_set_2d(x, y)
PlotGenerator.data_set_3d(x, y)
k_means = KMeans(x, y, 3)
k_means.random_centers()
C, CX = k_means.process(x)
PlotGenerator.clusters_2d(CX, x)
PlotGenerator.clusters_3d(CX, x)
error = Calculation.quantization_error(x, C, CX)
CX = Calculation.proper_classes(CX, y)
accuracy = Calculation.accuracy(CX, y)
print("Error: %f" % error)
print("Accuracy: %f" % accuracy)
