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



def power_spectrum_np(cube, mean_raw_cube):
    """
    taken from: https://astronomy.stackexchange.com/questions/26431/tests-for-code-that-computes-two-point-correlation-function-of-galaxies
    
    cube = should be in shape [. x . x. x . x . ]
    mean_raw_cube = is the mean of the whole cube of that redshift
    """

    print(cube.shape)
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
    









def plot_power_spec(real_cube,        # should be inverse_transformed
                    generated_cube,   # should be inverse_transformed
                    raw_cube_mean,    # mean of the whole raw data cube (fields=z0.0)
                    save_plot,
                     show_plot,
                     redshift_fig_folder,
                     t,
                    threads=1, 
                    MAS="CIC", 
                    axis=0, 
                    BoxSize=75.0/2048*128):
    """Takes as input;
    - Real cube: (batch_size x 1 x n x n x n) torch cuda FloatTensor,
    - Generated copy: (batch_size x 1 x n x n x n) torch cuda FloatTensor,
    - constant assignments: threads, MAS, axis, BoxSize.
    
    Returns;
    - Power spectrum plots of both cubes
    in the same figure.
    """
    real_cube = real_cube.reshape(1,1,real_cube.shape[0],real_cube.shape[0],real_cube.shape[0])
    generated_cube = generated_cube.reshape(1,1,generated_cube.shape[0],generated_cube.shape[0],generated_cube.shape[0])
    
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

        plt.plot(k_real, 
                 np.log10(Pk_real), 
                 color="b", 
                 alpha = 0.2,
                 label="Real Samples")
        plt.plot(k_gen, 
                 np.log10(Pk_gen), 
                 color="r",
                 alpha = 0.2,
                 label="Generated from Noise")
        plt.rcParams["font.size"] = 12
        plt.title("Power Spectrum Comparison - (Red: Real, Blue: Noise-Generated)")
        plt.xlabel('k')
        plt.ylabel('log10(Pk.k3D)')
#         plt.legend()
    
    if save_plot:
        plt.savefig(redshift_fig_folder + 'powerspectrum_' + str(t) + '.png', 
                    bbox_inches='tight')
    if show_plot:
        plt.show() 
    plt.close()
    
    
    return 