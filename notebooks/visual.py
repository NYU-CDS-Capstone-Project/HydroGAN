import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
import h5py
import matplotlib as mpl
%matplotlib inline
import itertools

## load file
f = h5py.File("sample16.h5", mode="r")
f = np.array([*f["sample16"]])

## function for truncating the colormap
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """Function for dividing cmaps,
    Retrieved from: https://stackoverflow.com/questions/\
    18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib"""
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

import itertools
import itertools

def visualize_cube(cube=None,      ## array name
             edge_dim=None,        ## edge dimension (128 for 128 x 128 x 128 cube)
             start_cube_index=0,
             fig_size=(20,20),
             stdev_to_white=2,
             norm_multiply=600,
             color_map="Blues",
             ):
    
    cube_size = edge_dim
    start_cube_index = start_cube_index
    edge = np.array([*range(cube_size)])
    
    start = start_cube_index
    end = start_cube_index + cube_size
    
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    
    data_value = cube[start:end,
                      start:end,
                      start:end]
    
    x,y,z = edge,edge,edge
    product = [*itertools.product(x,y,z)]
    
    X = np.array([product[k][0] for k in [*range(len(product))]])
    Y = np.array([product[k][1] for k in [*range(len(product))]])
    Z = np.array([product[k][2] for k in [*range(len(product))]])
    
    ## map data to 1d array that corresponds to the axis values in the product array
    data_1dim = np.array([data_value[X[i]][Y[i]][Z[i]] for i in [*range(len(product))]])
    
    initial_mean = np.mean(data_1dim) - stdev_to_white*np.std(data_1dim)
    mask = data_1dim > 0
    mask = mask.astype(np.int)
    
    data_1dim = np.multiply(mask,data_1dim)
    ## mask X,Y,Z to match the dimensions of the data
    X, Y, Z, data_1dim = [axis[np.where(data_1dim>0)] for axis in [X,Y,Z,data_1dim]]

    s = norm_multiply*data_1dim/np.linalg.norm(data_1dim)
    
    cmap=plt.get_cmap(color_map)
    new_cmap = truncate_colormap(cmap, 0.96, 1,n=1000)
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    ax.scatter(X, Y, Z,       ## axis vals
               c=data_1dim,   ## data, mapped to 1-dim
               cmap=new_cmap,
               s=s,           ## sizes - dims multiplied by each data point's magnitude
               alpha=1)
    
    
    
    plt.show()
    