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
import utils.data_utils
import timeit



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
             plot_save = False,
             save_fig = False):
    

    """Takes as input;
    - cube: A 3d numpy array to visualize,
    - edge_dim: edge length,
    - fig_size: Figure size for the plot,
    - norm_multiply: Multiplication factor to enable matplotlib to 'see' the particles,
    - color_map: A maplotlib colormap of your choice,
    
    Returns: 
    - The cube visualization
    - Also saves PNG file
    
    TODO:
    - Plot only the values > some value
    - timing of each part
    
    PROBLEMS:
    - Plotting everypoint (not truncating) with colorscale from 0 to 1 takes a really long time
    - size = magnitude -> some dots get too big to obscure view
    
    
    """
    time_start = timeit.default_timer()
        
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
    
    time_1 = timeit.default_timer()
#     print("Section 1 time = " + str((time_1 - time_start)/60))
    # avg: 0.02 sec
    
    ## map data to 1d array that corresponds to the axis values in the product array
    data_1dim = np.array([data_value[X[i]][Y[i]][Z[i]] for i in [*range(len(product))]])
    
    
#     initial_mean = np.mean(data_1dim) - stdev_to_white*np.std(data_1dim)
#     mask = data_1dim > initial_mean
#     mask = mask.astype(np.int)

    """
    Masking part of the data to speed up plotting time
    (may use just radius limiting below)
    """
    
#     data_1dim = np.multiply(mask,data_1dim)
    ## mask X,Y,Z to match the dimensions of the data
    X, Y, Z, data_1dim = [axis[np.where(data_1dim>0)] for axis in [X,Y,Z,data_1dim]]

    """
    norm_multiply = function argument
    data_1dim = data in one dimension, flattened
    nn.linalg.norm = return one of eight different matrix norms

    The nn.linalg.norm should not be used due to the fact 
    that each subcube has different norms. Use the maximum of
    the whole cube and mitigate with norm_multiply if 
    no points are seen in the 3D plot.
    """

    if size_magnitude == True:
        s = norm_multiply * data_1dim
    else:
        s = norm_multiply * np.ones_like(a = data_1dim)
        
    """
    s radius limit
    """

    # adding this for [-1,1] scaled input
    # so that s is between [0,1]
#         s = (s + 1.0) / 2.0 
    
#     else:
#         s = np.log(norm_multiply*data_1dim/np.linalg.norm(data_1dim))

    time_2 = timeit.default_timer()
#     print("Section 2 time = " + str((time_2 - time_1)/60))
    # avg: 0.03 sec

    try:
        # checking min, max , mean of s
        print("scatter size mean = " + str(s.mean()))
        print("scatter size max = " + str(s.max()))
        print("scatter size min = " + str(s.min()))


        """
        Truncating the colormap has the effect of showing even really small
        densities. 
        0 = white
        1 = full color
        minval = 0.99 -> even 0 densities are shown with 0.99 color
        n = number of division between minval and maxval of color
        
        """
        cmap = plt.get_cmap(color_map)
        new_cmap = truncate_colormap(cmap, 
                                     minval = 0.20, 
                                     maxval = 1,
                                     n=10)

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
        
        time_3 = timeit.default_timer()
#         print("Section 3 time = " + str((time_3 - time_2)/60))
        # avg: 0.0005 sec

        ax.scatter(X, Y, Z,       ## axis vals
                   c=data_1dim,   ## data, mapped to 1-dim
                   cmap=new_cmap,
                   s=s,           ## sizes - dims multiplied by each data point's magnitude
                   alpha=0.7,
                   edgecolors="face")
        
        time_4 = timeit.default_timer()
#         print("Section 4 time = " + str((time_4 - time_3)/60))
        # avg: 0.11 sec
    
        if plot_show:
            plt.show()
        
        time_5 = timeit.default_timer()
#         print("Section 5 time = " + str((time_5 - time_4)/60))
        # avg:  sec

        if plot_save:
            if save_fig:
                fig.savefig(save_fig,bbox_inches='tight')

        plt.close(fig)
    
    except:
        pass
    
    
    
def mmd_hist_plot(recon, real, epoch, file_name, 
                  plot_pdf , log_plot, plot_show,
                 redshift_fig_folder):
    """
    Args:
        recon(): generated data
        real(): real data
        epoch(integer): epoch number
        file_name(string): name of the file
        hd (integer) : if 0 it's a histogram, if 1 it's a pdf
        
    """
    if log_plot:
        try:
            recon = np.log(recon)
            real = np.log(real)
        except:
            print("Couldnt take the log of the values...")
            return
    
    
    plt.figure(figsize = (16,8))
    if plot_pdf == False:
        plt.title("Histograms of Hydrogen")
    else:
        plt.title("PDFs of Hydrogen")
    plt.xlim(min(recon.min(),real.min()),
            max(recon.max(),real.max()))
    bins = np.linspace(min(recon.min(),real.min()),
                       max(recon.max(),real.max()), 
                       100)
    real_title = "Real Sample Subcube - Only Nonzero"
    recon_title = "Generator(Noise) Subcube - Only Nonzero"
    if log_plot:
        real_title = real_title + "  (Log)"
        recon_title = recon_title + "  (Log)"
                  
    plt.hist(real, 
             bins = bins, 
             color = "blue" ,
             alpha = 0.3, 
             label = real_title,
             density = plot_pdf)
    plt.hist(recon, 
             bins = bins, 
         color = "red" ,
         alpha= 0.5, 
         label = recon_title,
            density = plot_pdf)

    plt.legend()
    if log_plot:
        plt.savefig(redshift_fig_folder + file_name +"_log_"+ str(epoch) + '.png', 
                    bbox_inches='tight')
    else:
        plt.savefig(redshift_fig_folder + file_name + str(epoch) + '.png', 
                bbox_inches='tight')
    
    if plot_show:
        plt.show() 
    plt.close()

    return



def mmd_loss_plots(fig_id, fig_title, data, show_plot, save_plot, redshift_fig_folder, t):
    """
    Args:
        fig_id(int): figure number
        fig_title(string): title of the figure
        data(): data to plot
        save_direct(string): directory to save
    """
    plt.figure(fig_id, figsize = (10,5))
    plt.title(fig_title)
    plt.plot(data)
    if save_plot:
        plt.savefig(redshift_fig_folder + fig_title +'_' + str(t) + '.png', 
                    bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()

    
def plot_minibatch_value_sum(sum_real,
                             sum_real_recon,
                             sum_noise_gen,
                             sum_noise_gen_recon,
                             save_plot,
                             show_plot,
                             redshift_fig_folder,
                             t):
                  
    plt.figure(figsize = (12,6))
    plt.title("Sum of Minibatches")
    plt.plot(sum_real, label = "sum_real", alpha = 0.9)
    plt.plot(sum_real_recon, label = "sum_real_recon", alpha = 0.3)
    plt.plot(sum_noise_gen, label = "sum_noise_gen", alpha = 0.9)
    plt.plot(sum_noise_gen_recon, label = "sum_noise_gen_recon", alpha = 0.3)
    plt.legend()
    if save_plot:
        plt.savefig(redshift_fig_folder + 'sum_minibatch_' + str(t) + '.png', 
                    bbox_inches='tight')
    if show_plot:
        plt.show() 
    
    plt.close()
    
    
    

