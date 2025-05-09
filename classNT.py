import numpy
import scipy

class NueralNetwork(object):

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set up nodes
        self.inodes = input_nodes
        self.onodes = output_nodes
        self.hnodes = hidden_nodes
        # set up learning rate
        self.learning_rate = learning_rate

        # link weight matrices , wih and who
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self):
        pass
    
    # query the neural network
    def query(self, input_list):
        # convert input list to 2d array
        inputs = numpy.array(input_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate signal emerging from final layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
