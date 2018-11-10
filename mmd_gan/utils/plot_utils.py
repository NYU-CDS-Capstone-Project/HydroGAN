import numpy as np
import matplotlib.pyplot as plt
import itertools
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
import h5py
import matplotlib as mpl
from pathlib import Path



def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """Function for dividing/truncating cmaps"""
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def visualize_cube(cube=None,      # array name
             edge_dim=None,        # edge dimension (128 for 128 x 128 x 128 cube)
             start_cube_index_x=0,
             start_cube_index_y=0,
             start_cube_index_z=0,
             fig_size=None,
#              stdev_to_white=-3,
             norm_multiply=1e2,
             size_magnitude = False,
             color_map="Blues",
             plot_show = False,
             save_fig = False):
    
    """Takes as input;
    - cube: A 3d numpy array to visualize,
    - edge_dim: edge length,
    - fig_size: Figure size for the plot,
    - norm_multiply: Multiplication factor to enable matplotlib to 'see' the particles,
    - color_map: A maplotlib colormap of your choice,
    - lognormal: Whether to apply lognormal transformation or not. False by default.
    
    Returns: 
    - The cube visualization
    
    TODO:
    - Plotting everypoint with colorscale from 0 to 1 takes a really long time
    
    
    """
        
    cube_size = edge_dim
    edge = np.array([*range(cube_size)])
    
    end_x = start_cube_index_x + cube_size
    end_y = start_cube_index_y + cube_size
    end_z = start_cube_index_z + cube_size
    
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    
    data_value = cube[start_cube_index_x:end_x,
                      start_cube_index_y:end_y,
                      start_cube_index_z:end_z]
    
    x,y,z = edge,edge,edge
    product = [*itertools.product(x,y,z)]
    
    X = np.array([product[k][0] for k in [*range(len(product))]])
    Y = np.array([product[k][1] for k in [*range(len(product))]])
    Z = np.array([product[k][2] for k in [*range(len(product))]])
    
    ## map data to 1d array that corresponds to the axis values in the product array
    data_1dim = np.array([data_value[X[i]][Y[i]][Z[i]] for i in [*range(len(product))]])
    
    
#     initial_mean = np.mean(data_1dim) - stdev_to_white*np.std(data_1dim)
#     mask = data_1dim > initial_mean
#     mask = mask.astype(np.int)
    
#     data_1dim = np.multiply(mask,data_1dim)
    ## mask X,Y,Z to match the dimensions of the data
    X, Y, Z, data_1dim = [axis[np.where(data_1dim>0)] for axis in [X,Y,Z,data_1dim]]

    
    
#     if lognormal == False:
    """
    norm_multiply = function argument
    data_1dim = data in one dimension, flattened
    nn.linalg.norm = return one of eight different matrix norms

    The nn.linalg.norm should not be used due to the fact 
    that each subcube has different norms. Use the maximum of
    the whole cube and mitigate with norm_multiply if 
    no points are seen in the 3D plot.
    """
        
#         s = norm_multiply*data_1dim/np.linalg.norm(data_1dim)

    if size_magnitude == True:
        s = norm_multiply * data_1dim
    else:
        s = norm_multiply * np.ones_like(a = data_1dim)

    # adding this for [-1,1] scaled input
    # so that s is between [0,1]
#         s = (s + 1.0) / 2.0 
    
#     else:
#         s = np.log(norm_multiply*data_1dim/np.linalg.norm(data_1dim))
    try:
        # checking min, max , mean of s
        print("Plotting s (= norm_multiply * data_1dim) stats:")
        print("s mean = " + str(s.mean()))
        print("s max = " + str(s.max()))
        print("s min = " + str(s.min()))


        """
        Truncating the colormap has the effect of showing even really small
        densities. 
        0 = white
        1 = full color
        minval = 0.99 -> even 0 densities are shown with 0.99 color
        n = number of division between minval and maxval of color
        
        """
        cmap = plt.get_cmap(color_map)
        new_cmap = truncate_colormap(cmap, minval = 0, maxval = 1,n=10)

        ## IGNORE BELOW 3D PLOT FORMATTING 

        ## plot cube

        cube_definition = [(start_cube_index_x, start_cube_index_x, start_cube_index_x),
                          (start_cube_index_x, start_cube_index_x+edge_dim, start_cube_index_x),
                          (start_cube_index_x + edge_dim, start_cube_index_x, start_cube_index_x),
                          (start_cube_index_x, start_cube_index_x, start_cube_index_x+edge_dim)]

        cube_definition_array = [
            np.array(list(item))
            for item in cube_definition
        ]

        points = []
        points += cube_definition_array
        vectors = [
            cube_definition_array[1] - cube_definition_array[0],
            cube_definition_array[2] - cube_definition_array[0],
            cube_definition_array[3] - cube_definition_array[0]
        ]


        points += [cube_definition_array[0] + vectors[0] + vectors[1]]
        points += [cube_definition_array[0] + vectors[0] + vectors[2]]
        points += [cube_definition_array[0] + vectors[1] + vectors[2]]
        points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

        points = np.array(points)

        edges = [
            [points[0], points[3], points[5], points[1]],
            [points[1], points[5], points[7], points[4]],
            [points[4], points[2], points[6], points[7]],
            [points[2], points[6], points[3], points[0]],
            [points[0], points[2], points[4], points[1]],
            [points[3], points[6], points[7], points[5]]
        ]

    #     ax.fig.add_subplot(111, projection='3d')

        faces = Poly3DCollection(edges, linewidths=1, edgecolors='k',)
        faces.set_facecolor((0,0,1,0)) ## set transparent facecolor to the cube

        ax.add_collection3d(faces)

        ax.scatter(points[:,0], points[:,1], points[:,2], s=0)

        ax.set_aspect('equal')

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

        ax.xaxis.set_major_locator(MultipleLocator(edge_dim))
        ax.yaxis.set_major_locator(MultipleLocator(edge_dim))
        ax.zaxis.set_major_locator(MultipleLocator(edge_dim))

        ax.grid(False)

        ax.set_xlim3d(0,edge_dim)
        ax.set_ylim3d(0,edge_dim)
        ax.set_zlim3d(0,edge_dim)
    #     ax.get_frame_on()

        ax.xaxis._axinfo['tick']['inward_factor'] = 0
        ax.xaxis._axinfo['tick']['outward_factor'] = 0
        ax.yaxis._axinfo['tick']['inward_factor'] = 0
        ax.yaxis._axinfo['tick']['outward_factor'] = 0
        ax.zaxis._axinfo['tick']['inward_factor'] = 0
        ax.zaxis._axinfo['tick']['outward_factor'] = 0

        ax.scatter(X, Y, Z,       ## axis vals
                   c=data_1dim,   ## data, mapped to 1-dim
                   cmap=new_cmap,
                   s=s,           ## sizes - dims multiplied by each data point's magnitude
                   alpha=0.7,
                   edgecolors="face")

        if plot_show:
            plt.show()

        if save_fig:
            fig.savefig(save_fig,bbox_inches='tight')

        plt.close(fig)
    
    except:
        pass
    
    

    
    
    
    

