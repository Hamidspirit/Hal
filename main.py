from classNT import NueralNetwork
import numpy

in_nodes = 3
ou_nodes = 3
hid_nodes = 3

lr = 0.5

array = numpy.random.rand(3, 3) - 0.5

n = NueralNetwork(in_nodes, hid_nodes, ou_nodes, lr)

if __name__ == "__main__":
    print("Running the model:")