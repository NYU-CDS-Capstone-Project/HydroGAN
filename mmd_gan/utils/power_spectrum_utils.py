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


# def power_spectrum_np(cube, mean_raw_cube):
#     """
#     taken from: https://astronomy.stackexchange.com/questions/26431/tests-for-code-that-computes-two-point-correlation-function-of-galaxies
    
#     cube = should be in shape [. x . x. x . x . ]
#     mean_raw_cube = is the mean of the whole cube of that redshift
#     """

# #     print(cube.shape)
#     nc = cube.shape[2]                # define how many cells your box has
#     boxlen = 4.7 # 128.0           # define length of box
#     Lambda = boxlen/4.0     # define an arbitrary wave length of a plane wave
#     dx = boxlen/nc          # get size of a cell

#     # create plane wave density field
# #     density_field = np.zeros((nc, nc, nc), dtype='float')
# #     for x in range(density_field.shape[0]):
# #         density_field[x,:,:] = np.cos(2*np.pi*x*dx/Lambda)
#     density_field = cube

# #    # get overdensity field
# #     delta = density_field/np.mean(density_field) - 1
#     delta = density_field / mean_raw_cube - 1

#     # get P(k) field: explot fft of data that is only real, not complex
#     delta_k = np.abs(np.fft.rfftn(delta).round())
#     Pk_field =  delta_k**2

#     # get 3d array of index integer distances to k = (0, 0, 0)
#     dist = np.minimum(np.arange(nc), np.arange(nc,0,-1))
#     dist_z = np.arange(nc//2+1)
#     dist *= dist
#     dist_z *= dist_z
#     dist_3d = np.sqrt(dist[:, None, None] + dist[:, None] + dist_z)

#     # get unique distances and index which any distance stored in dist_3d 
#     # will have in "distances" array
#     distances, _ = np.unique(dist_3d, return_inverse=True)
    
#     # average P(kx, ky, kz) to P(|k|)
#     Pk = np.bincount(_, weights=Pk_field.ravel())/np.bincount(_)

#     # compute "phyical" values of k
#     dk = 2*np.pi/boxlen
#     k = distances*dk
    
#     # moving averages
#     Pk = moving_average(np.asarray(Pk), 5)
#     k = moving_average(np.asarray(k), 5)

#     # plot results
# #     fig = plt.figure(figsize=(9,6))
# #     ax1 = fig.add_subplot(111)
# # #     ax1.plot(k, Pk, label=r'$P(\mathbf{k})$')
# #     ax1.plot(k, np.log10(Pk), 
# #              alpha = 0.2,
# #              label=r'$log(P(\mathbf{k}))$')
# #     ax1.legend()
# #     plt.show()

#     return Pk, dk, k
    

# def power_spectrum_np(cube, mean_raw_cube, SubBoxSize):
def power_spectrum_np(cube, mean_raw_cube):

#    SubBoxSize = 128
    SubBoxSize = 75.0/2048.0*64
    
    nc = cube.shape[2] # define how many cells your box has
    delta = cube/mean_raw_cube - 1.0

    # get P(k) field: explot fft of data that is only real, not complex
    delta_k = np.abs(np.fft.rfftn(delta)) 
    Pk_field =  delta_k**2

    # get 3d array of index integer distances to k = (0, 0, 0)
    dist = np.minimum(np.arange(nc), np.arange(nc,0,-1))
    dist_z = np.arange(nc//2+1)
    dist *= dist
    dist_z *= dist_z
    dist_3d = np.sqrt(dist[:, None, None] + dist[:, None] + dist_z)

    ################ NEW #################
    dist_3d  = np.ravel(dist_3d)
    Pk_field = np.ravel(Pk_field)
    
    k_bins = np.arange(nc//2+1)
    k      = 0.5*(k_bins[1:] + k_bins[:-1])*2.0*np.pi/SubBoxSize
    
    Pk     = np.histogram(dist_3d, bins=k_bins, weights=Pk_field)[0]
    Nmodes = np.histogram(dist_3d, bins=k_bins)[0]
    Pk     = (Pk/Nmodes)*(SubBoxSize/nc**2)**3
    
    k = k[1:];  Pk = Pk[1:]
    
    return Pk, k 







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
    


    plt.figure(figsize=(16,8))
    
    for cube_no in range(real_cube.shape[0]):
        
        delta_real_cube = real_cube[cube_no][0]
        delta_gen_cube = generated_cube[cube_no][0]
        
#         Pk_real, dk_real, k_real = power_spectrum_np(cube = delta_real_cube, 
#                                                      mean_raw_cube = raw_cube_mean)
#         Pk_gen, dk_gen, k_gen = power_spectrum_np(cube = delta_gen_cube, 
#                                                   mean_raw_cube = raw_cube_mean)
        
        Pk_real, k_real = power_spectrum_np(cube = delta_real_cube, 
                                                     mean_raw_cube = raw_cube_mean)
        Pk_gen, k_gen = power_spectrum_np(cube = delta_gen_cube, 
                                                  mean_raw_cube = raw_cube_mean)
        
        

#         plt.plot(np.log10(k_real), 
#                  np.log10(Pk_real), 
#                  color="r", 
#                  alpha = 0.2,
#                  label="Real Samples")
#         plt.plot(np.log10(k_gen), 
#                  np.log10(Pk_gen), 
#                  color="b",
#                  alpha = 0.2,
#                  label="Generated from Noise")

        plt.loglog(k_real, 
                 Pk_real, 
                 color="r", 
                 alpha = 0.2,
                 linewidth = 0.5,
                 label="Real Samples")
        plt.loglog(k_gen, 
                 Pk_gen, 
                 color="b",
                 alpha = 0.2,
                 linewidth = 0.5,
                 label="Generated from Noise")        
    
        plt.rcParams["font.size"] = 12
        plt.title("Power Spectrum Comparison - (Red: Real, Blue: Noise-Generated)")
        plt.xlabel('k')
        plt.ylabel('Pk.k3D')
#         plt.legend()
    
    if save_plot:
        plt.savefig(redshift_fig_folder + 'powerspectrum_' + str(t) + '.png', 
                    bbox_inches='tight')
    if show_plot:
        plt.show() 
    plt.close()
    
    
    return 



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
        
#         Pk_real, dk_real, k_real = power_spectrum_np(cube = delta_real_cube, 
#                                                      mean_raw_cube = raw_cube_mean)
#         Pk_gen, dk_gen, k_gen = power_spectrum_np(cube = delta_gen_cube, 
#                                                   mean_raw_cube = raw_cube_mean)
        
        Pk_real, k_real = power_spectrum_np(cube = delta_real_cube, 
                                                     mean_raw_cube = raw_cube_mean)
        Pk_gen, k_gen = power_spectrum_np(cube = delta_gen_cube, 
                                                  mean_raw_cube = raw_cube_mean)
        
        
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
#     print ("klog_ shape = "+str(k_log_real.shape))
    k_log_gen = np.array(k_log_gen)
#     print ("klog_gen shape = "+str(k_log_gen.shape))
    Pk_log_real = np.array(Pk_log_real)
#     print ("Pklog_real shape = "+str(Pk_log_real.shape))
    Pk_log_gen = np.array(Pk_log_gen)
#     print ("Pklog_gen shape = "+str(Pk_log_gen.shape))
    # get the means for each k value - among-cubes mean
    
    k_log_means_real = np.mean(k_log_real,
                                axis=0)
    
#     print ("\nk_log_means_real = "+str(k_log_means_real))
#     print ("\nk log gen whole series = "+str(k_log_gen))
    
    k_log_means_gen = np.mean(k_log_gen,
                                axis=0)
    
    # ge the standard deviations for each x ax value
    k_log_stds_real = np.std(k_log_real,
                                axis=0)
    
#     print ("\nk log stds real = "+str(k_log_stds_real))
    
    k_log_stds_gen = np.std(k_log_gen,
                                axis=0)
    
#     print ("\nk_log_stds_gen = "+str(k_log_stds_gen))
    # get the means for each k value
    Pk_log_means_real = np.mean(Pk_log_real,
                                axis=0)
    
#     print ("\nPk_log_means_real = "+str(Pk_log_means_real))
    Pk_log_means_gen = np.mean(Pk_log_gen,
                                axis=0)
    
#     print ("\nPk_log_means_gen = "+str(Pk_log_means_gen))
    
    # get the standard deviations for each x ax value
    Pk_log_stds_real = np.std(Pk_log_real,
                                axis=0)
    
    
#     print ("\nPk_log_stds_real = "+str(Pk_log_stds_real))
    
    Pk_log_stds_gen = np.std(Pk_log_gen,
                                axis=0)
    
#     print ("\nPk_log_stds_gen = "+str(Pk_log_stds_gen))
    
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
        plt.savefig(redshift_fig_folder + 'powerspectrum_std_' + str(t) + '.png', 
                    bbox_inches='tight')
    if show_plot:
        plt.show() 
    plt.close()
    
    
    return 