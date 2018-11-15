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
import torch, torchvision
import h5py
import pyfftw
import Pk_library as PKL

## model imports
from __future__ import print_function
import argparse
import numpy as np
import os
import h5py
import pickle as pkl
import random
# from mpi4py import MPI
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
%matplotlib inline

## normal noise input
from torch.distributions import normal

## plot imports
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



def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """Function for dividing/truncating cmaps"""
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def visualize_cube(cube=None,      ## array name
             edge_dim=None,        ## edge dimension (128 for 128 x 128 x 128 cube)
             start_cube_index_x=0,
             start_cube_index_y=0,
             start_cube_index_z=0,
             fig_size=None,
             norm_multiply=600,
             color_map="Blues",
             lognormal=False):
    
    """Takes as input;
    - cube: A 3d numpy array to visualize,
    - edge_dim: edge length,
    - fig_size: Figure size for the plot,
    - norm_multiply: Multiplication factor to enable matplotlib to 'see' the particles,
    - color_map: A maplotlib colormap of your choice,
    - lognormal: Whether to apply lognormal transformation or not. False by default.
    
    Returns: 
    - The cube visualization"""
    
    ## plot all the cubes in the batch
    
    # pdf's - original and regenerated
    
    # cube visuals
    
    # power spectrum
    
    
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

    if lognormal == False:
        s = norm_multiply * data_1dim
    else:
        s = np.log(norm_multiply*data_1dim)

    """
    The nn.linalg.norm should not be used due to the fact 
    that each subcube has different norms. Use the maximum of
    the whole cube and mitigate with norm_multiply if 
    no points are seen in the 3D plot.
    """
    # if lognormal == False:
    #     s = norm_multiply*data_1dim/np.linalg.norm(data_1dim)
    # else:
    #     s = np.log(norm_multiply*data_1dim/np.linalg.norm(data_1dim))
    
    cmap=plt.get_cmap(color_map)
    new_cmap = truncate_colormap(cmap, 0.99, 1,n=10)
    
    ## IGNORE BELOW 3D PLOT FORMATTING 
    
    ## plot cube
    
    cube_definition = [(start_cube_index_x, start_cube_index_x, start_cube_index_x),
                      (start_cube_index_x, start_cube_index_x+edge_dim, start_cube_index_x),
                      (start_cube_index_x+edge_dim, start_cube_index_x, start_cube_index_x),
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
    
    
    
    plt.show()

def plot_power_spec(real_cube, generated_cube,
                   threads=1, MAS="CIC", axis=0, BoxSize=75.0/2048*128):
    """Takes as input;
    - Real cube: (n x n x n) torch cuda FloatTensor,
    - Generated copy: (n x n x n) torch cuda FloatTensor,
    - constant assignments: threads, MAS, axis, BoxSize.
    
    Returns;
    - Power spectrum plots of both cubes
    in the same figure.
    """
    
    ## Assert same type
    assert ((real_cube.type() == generated_cube.type())&(real_cube.type()=="torch.FloatTensor")),\
    "Both input cubes should be torch.FloatTensor or torch.cuda().FloatTensor. Got real_cube type " + real_cube.type() + ", generated_cube type " + generated_cube.type() +"."
    ## Assert equal dimensions
    assert (real_cube.size() == generated_cube.size()),\
    "Two input cubes must have the same size. Got real_cube size " + str(real_cube.size()) + ", generated cube size " + str(generated_cube.size())
    
    ## if one or both of the cubes are cuda FloatTensors, detach them
    if real_cube.type() == "torch.cuda.FloatTensor":
        ## convert cuda FloatTensor to numpy array
        real_cube = real_cube.cpu().detach().numpy()
    else:
        real_cube = real_cube.numpy()
    
    if generated_cube.type() == "torch.cuda.FloatTensor":
        ## convert cuda FloatTensor to numpy array
        generated_cube = generated_cube.cpu().detach().numpy()
    else:
        generated_cube = generated_cube.numpy()
    
    # constant assignments
    BoxSize = BoxSize
    axis = axis
    MAS = MAS
    threads = threads

    # CALCULATE POWER SPECTRUM OF THE REAL CUBE
    # SHOULD WE DIVIDE BY WHOLE CUBE MEAN OR JUST THE MEAN OF THIS PORTION
    # Ask the Team
#     delta_real_cube /= mean_cube.astype(np.float64)
    
    delta_real_cube = real_cube
    delta_gen_cube = generated_cube
    
    delta_real_cube /= np.mean(delta_real_cube,
                              dtype=np.float64)
    delta_real_cube -= 1.0
    delta_real_cube = delta_real_cube.astype(np.float32)
    
    Pk_real_cube = PKL.Pk(delta_real_cube, BoxSize, axis, MAS, threads)
    
    
    # CALCULATE POWER SPECTRUM OF THE GENERATED CUBE
    delta_gen_cube /= np.mean(delta_gen_cube,
                             dtype=np.float64)
    delta_gen_cube -= 1.0
    delta_gen_cube = delta_gen_cube.astype(np.float32)
    
    Pk_gen_cube = PKL.Pk(delta_gen_cube, BoxSize, axis, MAS, threads)
    
    plt.figure(figsize=(10,5))
    plt.plot(np.log(Pk_real_cube.k3D), np.log(Pk_real_cube.Pk[:,0]), color="b", label="original cube")
    plt.plot(np.log(Pk_gen_cube.k3D), np.log(Pk_gen_cube.Pk[:,0]), color="r", label="jaas")
    plt.rcParams["font.size"] = 12
    plt.title("Power Spectrum Comparison")
    plt.xlabel('log(Pk.k3D)')
    plt.ylabel('log(Pk.k3D)')
    plt.legend()
    
    plt.show()
    return "Power spectrum plot complete!"


# import pyfftw
# import Pk_library as PKL
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt



import numpy as np
import matplotlib.pyplot as plt
import h5py
import numpy as np
import matplotlib.pyplot as plt


def moving_average(data_set, periods=3):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')


def power_spectrum_np(cube, mean_raw_cube):
    """
    taken from: https://astronomy.stackexchange.com/questions/26431/tests-for-code-that-computes-two-point-correlation-function-of-galaxies
    
    cube = should be in shape [. x . x. x . x . ]
    mean_raw_cube = is the mean of the whole cube of that redshift
    """

#     print(cube.shape)
    nc = cube.shape[2]                # define how many cells your box has
    boxlen = 50.0           # define length of box
    Lambda = boxlen/4.0     # define an arbitrary wave length of a plane wave
    dx = boxlen/nc          # get size of a cell

    # create plane wave density field
#     density_field = np.zeros((nc, nc, nc), dtype='float')
#     for x in range(density_field.shape[0]):
#         density_field[x,:,:] = np.cos(2*np.pi*x*dx/Lambda)
    density_field = cube

#    # get overdensity field
#     delta = density_field/np.mean(density_field) - 1
    delta = density_field / mean_raw_cube - 1

    # get P(k) field: explot fft of data that is only real, not complex
    delta_k = np.abs(np.fft.rfftn(delta).round())
    Pk_field =  delta_k**2

    # get 3d array of index integer distances to k = (0, 0, 0)
    dist = np.minimum(np.arange(nc), np.arange(nc,0,-1))
    dist_z = np.arange(nc//2+1)
    dist *= dist
    dist_z *= dist_z
    dist_3d = np.sqrt(dist[:, None, None] + dist[:, None] + dist_z)

    # get unique distances and index which any distance stored in dist_3d 
    # will have in "distances" array
    distances, _ = np.unique(dist_3d, return_inverse=True)

    # average P(kx, ky, kz) to P(|k|)
    Pk = np.bincount(_, weights=Pk_field.ravel())/np.bincount(_)

    # compute "phyical" values of k
    dk = 2*np.pi/boxlen
    k = distances*dk
    
    # moving averages
    Pk = moving_average(np.asarray(Pk), 20)
    k = moving_average(np.asarray(k), 20)

    # plot results
#     fig = plt.figure(figsize=(9,6))
#     ax1 = fig.add_subplot(111)
# #     ax1.plot(k, Pk, label=r'$P(\mathbf{k})$')
#     ax1.plot(k, np.log10(Pk), 
#              alpha = 0.2,
#              label=r'$log(P(\mathbf{k}))$')
#     ax1.legend()
#     plt.show()

    return Pk, dk, k
    
def plot_power_spec_aggregate(real_cube,        # should be inverse_transformed
                    generated_cube,   # should be inverse_transformed
                    raw_cube_mean,    # mean of the whole raw data cube (fields=z0.0)
                    save_plot,
                     show_plot,
                     redshift_fig_folder,
                     t,
                    threads=1, 
                    MAS="CIC", 
                    axis=0, 
                    BoxSize=75.0/2048*128,
                             k_steps=20):
    """Takes as input;
    - Real cube: (batch_size x 1 x n x n x n) torch cuda FloatTensor,
    - Generated copy: (batch_size x 1 x n x n x n) torch cuda FloatTensor,
    - constant assignments: threads, MAS, axis, BoxSize.
    
    Returns;
    - Power spectrum plots of both cubes
    in the same figure.
    """
    real_cube = real_cube.reshape(-1,
                                  1,
                                  real_cube.shape[2],
                                  real_cube.shape[2],
                                  real_cube.shape[2])
    generated_cube = generated_cube.reshape(-1,
                                            1,
                                            generated_cube.shape[2],
                                            generated_cube.shape[2],
                                            generated_cube.shape[2])
    
    print("number of samples of real and generated cubes = " + str(real_cube.shape[0]))
    
    
    
    
#     ## Assert same type
#     assert ((real_cube.type() == generated_cube.type())&(real_cube.type()=="torch.FloatTensor")),\
#     "Both input cubes should be torch.FloatTensor or torch.cuda().FloatTensor. Got real_cube type " + real_cube.type() + ", generated_cube type " + generated_cube.type() +"."
#     ## Assert equal dimensions
#     assert (real_cube.size() == generated_cube.size()),\
#     "Two input cubes must have the same size. Got real_cube size " + str(real_cube.size()) + ", generated cube size " + str(generated_cube.size())
    
#     ## if one or both of the cubes are cuda FloatTensors, detach them
#     if real_cube.type() == "torch.cuda.FloatTensor":
#         ## convert cuda FloatTensor to numpy array
#         real_cube = real_cube.cpu().detach().numpy()
#     else:
#         real_cube = real_cube.numpy()
    
#     if generated_cube.type() == "torch.cuda.FloatTensor":
#         ## convert cuda FloatTensor to numpy array
#         generated_cube = generated_cube.cpu().detach().numpy()
#     else:
#         generated_cube = generated_cube.numpy()
    
    # constant assignments
#     BoxSize = BoxSize
#     axis = axis
#     MAS = MAS
#     threads = threads

    
    k_log_real = []
    k_log_gen = []
    Pk_log_real = []
    Pk_log_gen = []

    plt.figure(figsize=(16,8))
    
    for cube_no in range(real_cube.shape[0]):
        
        delta_real_cube = real_cube[cube_no][0]
        delta_gen_cube = generated_cube[cube_no][0]
        
        Pk_real, dk_real, k_real = power_spectrum_np(cube = delta_real_cube, 
                                                     mean_raw_cube = raw_cube_mean)
        Pk_gen, dk_gen, k_gen = power_spectrum_np(cube = delta_gen_cube, 
                                                  mean_raw_cube = raw_cube_mean)
        
        
        # CALCULATE POWER SPECTRUM OF THE REAL CUBE
    
    #     delta_real_cube /= np.mean(delta_real_cube,
    #                               dtype=np.float64)
#         delta_real_cube /= raw_cube_mean
#         delta_real_cube -= 1.0
#         delta_real_cube = delta_real_cube.astype(np.float32)

#         Pk_real_cube = PKL.Pk(delta_real_cube, BoxSize, axis, MAS, threads)


#         # CALCULATE POWER SPECTRUM OF THE GENERATED CUBE
#     #     delta_gen_cube /= np.mean(delta_gen_cube,
#     #                              dtype=np.float64)
#         delta_gen_cube /= raw_cube_mean
#         delta_gen_cube -= 1.0
#         delta_gen_cube = delta_gen_cube.astype(np.float32)

#         Pk_gen_cube = PKL.Pk(delta_gen_cube, BoxSize, axis, MAS, threads)
        
        k_log_series_real = np.log10(np.array(k_real))
        k_log_real.append(k_log_series_real)
        k_log_series_gen = np.log10(np.array(k_gen))
        k_log_gen.append(k_log_series_gen)
        
        Pk_log_series_real = np.log10(np.array(Pk_real))
        Pk_log_real.append(Pk_log_series_real)
        Pk_log_series_gen = np.log10(np.array(Pk_gen))
        Pk_log_gen.append(Pk_log_series_gen)
        
        plt.plot(k_log_series_real, 
                 Pk_log_series_real, 
                 color="r", 
                 alpha = 0.2)
        plt.plot(k_log_series_gen, 
                 Pk_log_series_gen, 
                 color="b",
                 alpha = 0.2)
    
    # check axis and shape
    k_log_real = np.array(k_log_real)
    print ("klog_ shape = "+str(k_log_real.shape))
    k_log_gen = np.array(k_log_gen)
    print ("klog_gen shape = "+str(k_log_gen.shape))
    Pk_log_real = np.array(Pk_log_real)
    print ("Pklog_real shape = "+str(Pk_log_real.shape))
    Pk_log_gen = np.array(Pk_log_gen)
    print ("Pklog_gen shape = "+str(Pk_log_gen.shape))
    # get the means for each k value - among-cubes mean
    
    k_log_means_real = np.mean(k_log_real,
                                axis=0)
    
    print ("\nk_log_means_real = "+str(k_log_means_real))
    print ("\nk log gen whole series = "+str(k_log_gen))
    
    k_log_means_gen = np.mean(k_log_gen,
                                axis=0)
    
    # ge the standard deviations for each x ax value
    k_log_stds_real = np.std(k_log_real,
                                axis=0)
    
    print ("\nk log stds real = "+str(k_log_stds_real))
    
    k_log_stds_gen = np.std(k_log_gen,
                                axis=0)
    
    print ("\nk_log_stds_gen = "+str(k_log_stds_gen))
    # get the means for each k value
    Pk_log_means_real = np.mean(Pk_log_real,
                                axis=0)
    
    print ("\nPk_log_means_real = "+str(Pk_log_means_real))
    Pk_log_means_gen = np.mean(Pk_log_gen,
                                axis=0)
    
    print ("\nPk_log_means_gen = "+str(Pk_log_means_gen))
    
    # get the standard deviations for each x ax value
    Pk_log_stds_real = np.std(Pk_log_real,
                                axis=0)
    
    
    print ("\nPk_log_stds_real = "+str(Pk_log_stds_real))
    
    Pk_log_stds_gen = np.std(Pk_log_gen,
                                axis=0)
    
    print ("\nPk_log_stds_gen = "+str(Pk_log_stds_gen))
    
    for i in range(len(k_log_means_gen)):
        if i % k_steps == 0:
            plt.plot([k_log_means_gen[i],
                  k_log_means_gen[i],
                 k_log_means_gen[i]],
                [Pk_log_means_gen[i]-Pk_log_stds_gen[i],Pk_log_means_gen[i],
                Pk_log_means_gen[i]+Pk_log_stds_gen[i]], color="k", alpha=0.5, marker="_")
            
    # plot power specs
    plt.plot(k_log_means_real, 
             Pk_log_means_real, 
             color="r", 
             alpha = 1,
             label="Real Samples",
            linewidth=2.5)
    plt.plot(k_log_means_gen, 
             Pk_log_means_gen, 
             color="b",
             alpha = 1,
             label="Generated from Noise",
             linewidth=2.5)
        
    plt.rcParams["font.size"] = 12
    plt.title("Power Spectrum Comparison - (Red: Real, Blue: Noise-Generated)")
    plt.xlabel('log10(k)')
    plt.ylabel('log10(Pk.k3D)')
    plt.legend()
    
    if save_plot:
        plt.savefig(redshift_fig_folder + 'powerspectrum_' + str(t) + '.png', 
                    bbox_inches='tight')
    if show_plot:
        plt.show() 
    plt.close()
    
    
    return 