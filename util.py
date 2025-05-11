import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def res_image(value):
    """Return an image of the value. it takes a list"""

    image_array = numpy.asarray(value[1:], dtype=numpy.float32).reshape((28,28))
    plt.imshow(image_array, cmap="Grays", interpolation='None')
    plt.savefig("output.png")

