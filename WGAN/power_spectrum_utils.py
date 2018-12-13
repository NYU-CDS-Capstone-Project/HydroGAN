import numpy as np


def power_spectrum_np(cube, mean_raw_cube, SubBoxSize):

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
    
    return k, Pk


def power_spectrum_np_2d(cube, mean_raw_cube, SubBoxSize):
    #print(cube.shape)
    nc = cube.shape[1] # define how many cells your box has
    delta = cube/mean_raw_cube - 1.0

    # get P(k) field: explot fft of data that is only real, not complex
    delta_k = np.abs(np.fft.rfftn(delta)) 
    Pk_field =  delta_k**2
    #print(Pk_field.shape)

    # get 3d array of index integer distances to k = (0, 0, 0)
    dist = np.minimum(np.arange(nc), np.arange(nc,0,-1))
    dist_z = np.arange(nc//2+1)
    dist *= dist
    dist_z *= dist_z
    dist_2d = np.sqrt(dist[:, None] + dist_z)

    ################ NEW #################
    dist_2d  = np.ravel(dist_2d)
    Pk_field = np.ravel(Pk_field)
    
    k_bins = np.arange(nc//2+1)
    k      = 0.5*(k_bins[1:] + k_bins[:-1])*2.0*np.pi/SubBoxSize
    
    #print(dist_2d.shape)
    Pk     = np.histogram(dist_2d, bins=k_bins, weights=Pk_field)[0]
    Nmodes = np.histogram(dist_2d, bins=k_bins)[0]
    Pk     = (Pk/Nmodes)*(SubBoxSize/nc**2)**3
    
    k = k[1:];  Pk = Pk[1:]
    
    return k, Pk
    








