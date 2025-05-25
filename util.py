import scipy.misc
import matplotlib.pyplot as plt
import numpy
import matplotlib
matplotlib.use('Agg')


def res_image(value):
    """Return an image of the value. it takes a list"""

    image_array = numpy.asarray(
        value[1:], dtype=numpy.float32).reshape((28, 28))
    plt.imshow(image_array, cmap="Grays", interpolation='None')
    plt.savefig("output.png")


def hand_written(image_name):
    """Return a handwritten image data to query network"""
    img_array = scipy.misc.imread(image_name, flatten=True)

    image_data = 255.0 - img_array.reshape(784)
    image_data = (image_data / 255.0 * 0.99) + 0.01
    return image_data
