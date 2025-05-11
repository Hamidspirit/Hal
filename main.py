import numpy
import csv

from util import res_image
from classNT import NueralNetwork

# number of inputs, hidden, output nodes in network
in_nodes = 784
hid_nodes = 120
ou_nodes = 10

# learning rate is 0.2
lr = 0.2

# array = numpy.random.rand(3, 3) - 0.5

n = NueralNetwork(in_nodes, hid_nodes, ou_nodes, lr)

with open("./mnist-datasets/mnist_train.csv", "r") as f:
    training_list = csv.reader(f)
    next(training_list)

    epochs = 5

    for e in range(epochs):
        for record in training_list:

            all_values = record

            # scale and shift the inputss
            # print(all_values[1:])
            inputs = (numpy.asarray(all_values[1:], dtype=numpy.float32) / 255.0*0.99 ) + 0.01

            # create the target output values (all 0.01, except the desired label which is 0.99)
            targets = numpy.zeros(ou_nodes) + 0.01
            # all_values[0] is the target label for this record
            targets[int(all_values[0])] = 0.99

            n.train(inputs, targets)
        

# now getting the test data and testing network
with open("./mnist-datasets/mnist_test.csv", "r") as f:
    test_list = csv.reader(f)
    next(test_list)


    # testing neural network

    #keeping score
    scorecard = []

    for record in test_list:

        values = record
        # correct answer is first value
        correct = int(values[0])
        # print(f"correct label: {correct}")

        # scale and shift inputs
        inputs_test = (numpy.asarray(values[1:], dtype=numpy.float32)/ 255.0*0.99 ) + 0.01
        # query the network
        outputs = n.query(inputs_test)
        # index of highest value coresponds to the label
        label = numpy.argmax(outputs)
        # print(f"networks answer: {label}")
        
        # keep score
        if label == correct:
            scorecard.append(1)
        else:
            scorecard.append(0)

if __name__ == "__main__":
    print("Running the model:")
    score_array = numpy.asarray(scorecard)
    performance = score_array.sum() / score_array.size
    print(performance)
