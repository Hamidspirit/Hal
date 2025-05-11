# HAL

My attempt at learning machine learning.

this is simple neural network it only has 3 layers, input, hidden and output
layer.

the main job happens in `NeuralNetwork class`, in cassNT.py file.
instantiation of class requires number of nodes or each layer and a learning rate.
```py 
n = NeuralNetwork(3, 3, 3, 0.5)
```

when calling a query function you send a signal through the network and returns output.
```py
n.query()
```
**output example:**
[[0.27436274]
[0.54603049]
[0.5833037 ]]

training happens when you call the train method obviously.

```py
n.train
```

if you want to run this network you have to download the MNIST dataset,
and put it in a folder called mnist-datasets.

you might ask why and it is because i hard codded the part where i feed 
data to network.

- then create a .venv:

```py
python -m venv .venv
```

- install required libraries:

```py
pip install requirements.txt
```

- run the main file:

```py 
python main.py
```

when i run this with learning rate of 0.5 and 5 epochs and 100 hidden nodes i get
**0.918** or **91.8%** performance.

sources:

[10 record for setup test](https://raw.githubusercontent.com/makeyourownneuralnetwork/makeyourownneuralnetwork/master/mnist_dataset/mnist_test_10.csv)

[60 record for training](https://raw.githubusercontent.com/makeyourownneuralnetwork/makeyourownneuralnetwork/master/mnist_dataset/mnist_train_100.csv)

[MNIST data CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download)
