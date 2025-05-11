import numpy
import scipy.special

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

    def train(self, input_list, target_list):
        # this has two part first one is working out the 
        # output for given training example
        # second one is taking this output and calculate error
        # and then compare against desired output and use that to guide
        # the updating of network weight 
        # first part is already done in query function

        # convert input list and target list to 2d array
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate signal emerging from final layer
        final_outputs = self.activation_function(final_inputs)


        # calculate error (target - actual)
        output_errors = targets - final_outputs

        # hidden layer error is the output_errors, split by weights, recombined at
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.learning_rate*numpy.dot((output_errors*final_outputs*(1.0 - final_outputs)),
                                                  numpy.transpose(hidden_outputs))

        # update weight for links between hidden and output layer
        self.wih += self.learning_rate*numpy.dot((hidden_errors*hidden_outputs*(1.0 - hidden_outputs)),
                                                  numpy.transpose(inputs))

    
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
