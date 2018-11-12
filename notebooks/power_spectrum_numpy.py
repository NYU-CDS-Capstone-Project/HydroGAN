#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import h5py


# In[5]:


run_in_jupyter = False
try:
    cfg = get_ipython().config 
    run_in_jupyter = True
except:
    run_in_jupyter = False
    pass


# In[6]:


if run_in_jupyter:
    get_ipython().run_line_magic('run', 'dataset.py')
else:
    from dataset import *


# In[20]:


cube_size = 256 


# In[21]:


test_coords = define_test(s_test = 1024, s_train = cube_size)
test_coords


# In[22]:


data_dir = "../"
redshift_raw_file = "fields_z=0.0.hdf5"


# In[23]:


f = h5py.File(data_dir + redshift_raw_file, 'r')
print("File used for analysis = " + str(f.filename))
f = f['delta_HI']


# In[24]:


sampled_cube = get_samples(f=f,nsamples=1,s_sample=cube_size,test_coords=test_coords)
sampled_cube = sampled_cube[0]
print(sampled_cube.shape)


# In[28]:


root_dir = "./"
experiment = root_dir + "mmd-jupyter/"
redshift_info_folder = experiment + "redshift_info/"
mean_raw_cube = np.load(file = redshift_info_folder + redshift_raw_file + "_mean_cube" + '.npy')
mean_raw_cube


# In[33]:


import numpy as np
import matplotlib.pyplot as plt


#==================================
def main():
#==================================


    nc = cube_size                # define how many cells your box has
    boxlen = 50.0           # define length of box
    Lambda = boxlen/4.0     # define an arbitrary wave length of a plane wave
    dx = boxlen/nc          # get size of a cell

    # create plane wave density field
#     density_field = np.zeros((nc, nc, nc), dtype='float')
#     for x in range(density_field.shape[0]):
#         density_field[x,:,:] = np.cos(2*np.pi*x*dx/Lambda)
    density_field = sampled_cube

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
    fig = plt.figure(figsize=(9,6))
    ax1 = fig.add_subplot(111)
#     ax1.plot(k, Pk, label=r'$P(\mathbf{k})$')
    ax1.plot(k, np.log10(Pk), label=r'$log(P(\mathbf{k}))$')

    # plot expected peak:
    # k_peak = 2*pi/lambda, where we chose lambda for our planar wave earlier
#     ax1.plot([2*np.pi/Lambda]*2, [Pk.min()-1, Pk.max()+1], label='expected peak')
    ax1.legend()
    plt.show()
    
#==================================
if __name__ == "__main__":
#==================================

    main()


# In[ ]:




