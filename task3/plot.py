#!/usr/bin/python2

from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import sys, os

def plot(filename):
    z = np.loadtxt(filename)
    plt.figure()
    im = plt.imshow(z, interpolation='none', cmap=cm.hot,
                    origin='lower', extent=(0,1,0,1))
    icb = plt.colorbar(im, orientation='horizontal')
    plt.savefig(os.path.splitext(filename)[0] + '.png')
    #plt.show()
    plt.close()


if __name__ == '__main__':
    plot(sys.argv[1])
