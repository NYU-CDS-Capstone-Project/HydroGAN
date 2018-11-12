#!/usr/bin/env python
# coding: utf-8

# https://github.com/OctoberChang/MMD-GAN - accompanying the paper MMD-GAN: Towards Deeper Understanding of Moment Matching Network.
# 
# To check GPU usage, open new terminal inside Jupyter and nvidia-smi

# #### To run with sbatch, 
# * change the jupyter notebook parameters
#     * batch_size
#     * nz (= output channels in the embedding)
#     * lr
#     * optimizer_choice
#     * dist_ae
#     * left_clamp & right_clamp
#     * plot_show_3d = False
#     * plot_save_3d = True
#     * experiment = a folder to output to (drive_output is a good name)
#     * redshift_raw_file = raw data file like fields_z=0.0.hdf5
#     * redshift_file = transformed data file like minmax_scale_neg11_redshift0.h5
#     * inverse_transform = one of minmax11 / minmaxneg11 / std_noshift / std based on the transformed data file
#     * 
# * change the file name inside the run-mmdgan-110919.sbatch file
# * do: sbatch run-mmdgan-110918.sbatch

# #### To transfer files to google drive
# * module load rclone/1.38
# * rclone copy /scratch/jjz289/data/mmd_gan_code/drive_output/ remote1:capstone_cosmo_DATE_X

# In[1]:


import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
import h5py
import timeit
import time
import numpy as np
from scipy import stats
import pickle as pkl
from os import listdir
from os.path import isfile, join


# In[2]:


run_in_jupyter = False
try:
    cfg = get_ipython().config 
    run_in_jupyter = True
except:
    run_in_jupyter = False
    pass

if run_in_jupyter:
    import matplotlib.pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
else: 
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
print("Run in Jupyter = " + str(run_in_jupyter))


# In[3]:


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


# ### Training Options

# In[4]:


batch_size = 16       # BATCH_SIZE: batch size for training
gpu_device = 0        # GPU_DEVICE: gpu id (default 0)
nc = 1                # NC: number of channels in images
nz = 256                # NZ: number of channels, hidden dimension in z and codespace
cube_size = 128       # for our dataset more like one edge of the subcube
lr = 5e-5               # LR: learning rate - default: 5e-5
max_iter = 150         # MAX_ITER: max iteration for training
optimizer_choice = "rmsprop"     # adam or rmsprop
dist_ae = 'L2'                  # "L2" or "L1" or "cos" -> Autoencoder reconstructruced cube loss choice
manual_seed = 1126
sample_size_multiplier = 128
n_samples = batch_size * sample_size_multiplier      # on prince, number of samples to get from the training cube
Diter_1 = 100    # default: 100
Giter_1 = 1      # default: 1
Diter_2 = 5      # default: 5
Giter_2 = 1      # default: 1
gen_iterations_limit = 25   # default = 25


# In[5]:


assert n_samples / batch_size > 100, "The gen_iterations wont work properly!"


# ### Model Options

# In[6]:


if run_in_jupyter:
    get_ipython().run_line_magic('run', 'utils/power_spectrum_utils.py')
else:
    from utils.power_spectrum_utils import *


# In[7]:


"""
for p in netD.encoder.parameters():
    p.data.clamp_(left_clamp, right_clamp)
"""
left_clamp = -0.01     # default: -0.01
right_clamp = 0.01     # default: 0.01


# ### Plotting Options

# In[8]:


viz_multiplier = 1e0           # the norm multiplier in the 3D visualization
plot_show_3d = True            # shows the 3d scatter plot
plot_save_3d = False           # whether to save or not as png
scatter_size_magnitude = False  # change scatter point radius based on the value of the point


# ### Saving Options

# In[9]:


root_dir = "./"  # this goes to 
data_dir = "../"
if run_in_jupyter:
    experiment = root_dir + "drive_output/"       # : output directory of saved models
else:
    experiment = root_dir + "mmd-jupyter/"
model_save_folder = experiment + "model/"
redshift_fig_folder = experiment + "figures/"        # folder to save mmd & related plots
redshift_3dfig_folder = experiment + "/3d_figures/"   # folder to save 3D plots
testing_folder = experiment + "testing/"   # folder to save 3D plots

save_model_every = 10               # (every x epoch) frequency to save the model


# ### Dataset Options

# In[10]:


workers = 2          # WORKERS: number of threads to load data
redshift_info_folder = experiment + "redshift_info/"   # save some info here as pickle to speed up processing
redshift_raw_file = "fields_z=0.0.hdf5"
redshift_file = "minmax_scale_neg11_redshift0.h5"    # redshift cube to be used
    # standardized_no_shift_redshift0.h5
    # minmax_scale_01_redshift0.h5
    # minmax_scale_neg11_redshift0.h5
inverse_transform = "minmaxneg11"    # minmax11 / minmaxneg11 / std_noshift / std


# ### Testing Options

# In[11]:


in_testing = False              # True if doing testing


# ### Debug Utils

# In[12]:


if run_in_jupyter:
    get_ipython().run_line_magic('run', 'utils/debug_utils.py')
else:
    from utils.debug_utils import log


# In[13]:


DEBUG = False


# In[14]:


print("log('asdas') output = " + str(log("asdas")))


# ## Parameter Documentation

# ## Training Parameters

# In[15]:


print("Batch Size = " + str(batch_size))
print("Redshift File Used = " + str(redshift_file))
print("Number of Channels in Input = " + str(nc))
print("Hidden Dimension (codespace) (nz)= " + str(nz))
print("Length of Edge of a Sampled Subcube = " + str(cube_size))
print("Learning Rate = " + str(lr))
print("Number of Epochs = " + str(max_iter))
print("Optimizer = " + str(optimizer_choice))
print("Autoencoder Reconstruction Loss  = " + str(dist_ae))
print("Seed = " + str(manual_seed))
print("Number of Samples = " + str(n_samples))
print("Visualization Multiplier = " + str(viz_multiplier))
print("gen_iterations_limit = " + str(gen_iterations_limit))
print("Diter_1 = " + str(Diter_1))
print("Giter_1 = " + str(Giter_1))
print("Diter_2 = " + str(Diter_2))
print("Giter_2 = " + str(Giter_2))


# In[16]:


edge_sample = cube_size
edge_test = 1024

print("one edge of the test partition of the whole cube = " + str(edge_test))
print("one edge of the sampled subcubes =  " + str(edge_sample))


# ### MMD Parameters

# In[17]:


"""
MMD Parameters

errD = torch.sqrt(mmd2_D) + lambda_rg * one_side_errD 
       - lambda_AE_X * L2_AE_X_D - lambda_AE_Y * L2_AE_Y_D
       
errG = torch.sqrt(mmd2_G) + lambda_rg * one_side_errG

The explanations can be found in Ratio Matching MMD Nets (2018) in 
Equation 3.

"""

lambda_MMD = 1.0   # not used anywhere
lambda_AE_X = 8.0  # used in above calc only 
lambda_AE_Y = 8.0  # used in above calc only
lambda_rg = 16.0   # used in both err calcs

print("lambda_MMD = " + str(lambda_MMD))
print("lambda_AE_X = " + str(lambda_AE_X))
print("lambda_AE_Y = " + str(lambda_AE_Y))
print("lambda_rg = " + str(lambda_rg))


# In[18]:


"""
sigma for MMD
"""
base = 1.0
sigma_list = [1, 2, 4, 8, 16]
sigma_list = [sigma / base for sigma in sigma_list]
print("sigma_list = " + str(sigma_list))


# In[ ]:


"""
used at:
def _mmd2_and_ratio(K_XX, K_XY, K_YY, 
                    const_diagonal=False, 
                    biased=False):
    mmd2, var_est = _mmd2_and_variance(K_XX, K_XY, K_YY, 
                                       const_diagonal=const_diagonal, 
                                       biased=biased)
    loss = mmd2 / torch.sqrt(torch.clamp(var_est, min=min_var_est))
    return loss, mmd2, var_est
    
torch.clamp(input, min, max, out=None) → Tensor
    Clamp all elements in input into the range [ min, max ] 
    and return a resulting tensor
"""

# min_var_est = 1e-8
min_var_est = 1e-30

print("minimum variance estimated = " + str(min_var_est))


# ## Redshift Data Load

# In[ ]:


f = h5py.File(data_dir + redshift_file, 'r')
print("File used for analysis = " + str(f.filename))
f = f['delta_HI']


# ## Redshift Info Load

# In[ ]:


# create trial folder if it doesn't exist
if Path(experiment).exists() == False:
    os.mkdir(experiment)


# In[ ]:


# create redshift info folder if it doesn't exist
if Path(redshift_info_folder).exists() == False:
    os.mkdir(redshift_info_folder)


# In[ ]:


if run_in_jupyter:
    get_ipython().run_line_magic('run', 'utils/data_utils.py')
else:
    from utils.data_utils import *


# In[ ]:


# check if redshift info (min & max exists) as pickle
# if not saved, find the max and min and save them for later use
min_cube_file = Path(redshift_info_folder + redshift_file + "_min_cube" + ".npy")
max_cube_file = Path(redshift_info_folder + redshift_file + "_max_cube" + ".npy")
mean_cube_file = Path(redshift_info_folder + redshift_file + "_mean_cube" + ".npy")
stddev_cube_file = Path(redshift_info_folder + redshift_file + "_stddev_cube" + ".npy")


if not min_cube_file.exists() or not max_cube_file.exists() or not mean_cube_file.exists() or not stddev_cube_file.exists():
    
    f = h5py.File(data_dir + redshift_file, 'r')
    f=f['delta_HI']
    
    # get the min and max
    min_cube = get_min_cube(f=f)
    print(min_cube)
    max_cube = get_max_cube(f=f)
    print(max_cube)
    mean_cube = get_mean_cube(f=f)
    print(mean_cube)
    stddev_cube = get_stddev_cube(f=f, mean_cube=mean_cube)
    print(stddev_cube)
    
    np.save(file = redshift_info_folder + redshift_file + "_min_cube",
        arr = min_cube,
        allow_pickle = True)
    np.save(file = redshift_info_folder + redshift_file + "_max_cube",
        arr = max_cube,
        allow_pickle = True)
    np.save(file = redshift_info_folder + redshift_file + "_mean_cube",
        arr = mean_cube,
        allow_pickle = True)
    np.save(file = redshift_info_folder + redshift_file + "_stddev_cube",
        arr = stddev_cube,
        allow_pickle = True)
    


# In[ ]:


# check if redshift info (min & max exists) as pickle
# if not saved, find the max and min and save them for later use
min_raw_cube_file = Path(redshift_info_folder + redshift_raw_file + "_min_cube" + ".npy")
max_raw_cube_file = Path(redshift_info_folder + redshift_raw_file + "_max_cube" + ".npy")
mean_raw_cube_file = Path(redshift_info_folder + redshift_raw_file + "_mean_cube" + ".npy")
stddev_raw_cube_file = Path(redshift_info_folder + redshift_raw_file + "_stddev_cube" + ".npy")


if not min_raw_cube_file.exists() or not max_raw_cube_file.exists() or not mean_raw_cube_file.exists() or not stddev_raw_cube_file.exists():
    
    f = h5py.File(data_dir + redshift_raw_file, 'r')
    f=f['delta_HI']
    
    # get the min and max
    min_cube = get_min_cube(f=f)
    print(min_cube)
    max_cube = get_max_cube(f=f)
    print(max_cube)
    mean_cube = get_mean_cube(f=f)
    print(mean_cube)
    stddev_cube = get_stddev_cube(f=f, mean_cube=mean_cube)
    print(stddev_cube)
    
    np.save(file = redshift_info_folder + redshift_raw_file + "_min_cube",
        arr = min_cube,
        allow_pickle = True)
    np.save(file = redshift_info_folder + redshift_raw_file + "_max_cube",
        arr = max_cube,
        allow_pickle = True)
    np.save(file = redshift_info_folder + redshift_raw_file + "_mean_cube",
        arr = mean_cube,
        allow_pickle = True)
    np.save(file = redshift_info_folder + redshift_raw_file + "_stddev_cube",
        arr = stddev_cube,
        allow_pickle = True)
    


# In[ ]:


min_cube = np.load(file = redshift_info_folder + redshift_file + "_min_cube" + '.npy')
max_cube = np.load(file = redshift_info_folder + redshift_file + "_max_cube" + '.npy')
print("Min of data = " + str(min_cube))
print("Max of data = " + str(max_cube))
mean_cube = np.load(file = redshift_info_folder + redshift_file + "_mean_cube" + '.npy')
stddev_cube = np.load(file = redshift_info_folder + redshift_file + "_stddev_cube" + '.npy')
print("Mean of data = " + str(mean_cube))
print("Stddev of data = " + str(stddev_cube))


# In[ ]:


min_raw_cube = np.load(file = redshift_info_folder + redshift_raw_file + "_min_cube" + '.npy')
max_raw_cube = np.load(file = redshift_info_folder + redshift_raw_file + "_max_cube" + '.npy')
print("Min of raw data = " + str(min_raw_cube))
print("Max of raw data = " + str(max_raw_cube))
mean_raw_cube = np.load(file = redshift_info_folder + redshift_raw_file + "_mean_cube" + '.npy')
stddev_raw_cube = np.load(file = redshift_info_folder + redshift_raw_file + "_stddev_cube" + '.npy')
print("Mean of raw data = " + str(mean_raw_cube))
print("Stddev of raw data = " + str(stddev_raw_cube))


# ## Figures Handling

# In[ ]:


# create figures folder if it doesn't exist
if Path(redshift_fig_folder).exists() == False:
    os.mkdir(redshift_fig_folder)
if Path(redshift_3dfig_folder).exists() == False:
    os.mkdir(redshift_3dfig_folder)


# ## 3D Plot

# In[ ]:


if run_in_jupyter:
    get_ipython().run_line_magic('run', 'utils/data_utils.py')
else:
    from utils.data_utils import *


# In[ ]:


if run_in_jupyter:
    get_ipython().run_line_magic('run', 'utils/plot_utils.py')
else:
    from utils.plot_utils import *


# ## Data Loader

# In[ ]:


if run_in_jupyter:
    get_ipython().run_line_magic('run', 'dataset.py')
else:
    from dataset import *


# In[ ]:


if run_in_jupyter:
    get_ipython().run_line_magic('run', 'test_3d_plot.py')
else:
    from test_3d_plot import *


# ## Checking Duplicates in Sampled Subcubes

# In[ ]:


s_sample = edge_sample 
nsamples = 10000

testcd = define_test(s_test = edge_test,
                     s_train = edge_sample)
print(testcd)

sample_list=[]
m = 2048 - 128


for n in range(nsamples):
    #print("Sample No = " + str(n + 1) + " / " + str(nsamples))
    sample_valid = False
    while sample_valid == False:
        x = random.randint(0,m)
        y = random.randint(0,m)
        z = random.randint(0,m)
        sample_coords = {'x':[x,x+s_sample], 
                         'y':[y,y+s_sample], 
                         'z':[z,z+s_sample]}

        sample_valid = check_coords(testcd, 
                                    sample_coords)

    sample_list.append(sample_coords)

print(len(sample_list))
# print(len(list(set(sample_list))))
sample_df = pd.DataFrame.from_dict(sample_list)
dropped_sample_df = sample_df.applymap(lambda x: x[0]).drop_duplicates()

sample_df.shape[0] == dropped_sample_df.shape[0]


# ## Dataset & DataLoader

# In[ ]:


# on prince
sampled_subcubes = HydrogenDataset(h5_file=redshift_file,
                                    root_dir = data_dir,
                                    f = f,
                                    s_test = edge_test, 
                                    s_train = edge_sample,
                                    s_sample = edge_sample, 
                                    nsamples = n_samples,
                                   min_cube = min_cube,
                                  max_cube = max_cube,
                                  mean_cube = mean_cube,
                                  stddev_cube = stddev_cube,
                                   min_raw_cube = min_raw_cube,
                                  max_raw_cube = max_raw_cube,
                                  mean_raw_cube = mean_raw_cube,
                                  stddev_raw_cube = stddev_raw_cube)


# In[ ]:


# Get data
trn_loader = torch.utils.data.DataLoader(sampled_subcubes, 
                                         batch_size = batch_size,
                                         shuffle=True, 
                                         num_workers=int(workers))


# ## Checking 3D Plots

# In[ ]:


# dont run this in batch
if run_in_jupyter:
    test_3d_plot(edge_test = edge_test, 
                 edge_sample = edge_sample,
                 f = f, 
                 scatter_size_magnitude = scatter_size_magnitude,
                 viz_multiplier = viz_multiplier,
                 plot_save_3d = plot_save_3d,
                 inverse_transform = inverse_transform,
                 sampled_subcubes = sampled_subcubes)


# ## Model

# In[ ]:


if run_in_jupyter:
    get_ipython().run_line_magic('run', 'utils/mmd_utils.py')
    get_ipython().run_line_magic('run', 'utils/model_utils.py')
    get_ipython().run_line_magic('run', 'utils/conv_utils.py')
    get_ipython().run_line_magic('run', 'decoder.py')
    get_ipython().run_line_magic('run', 'encoder.py')
else:
    from utils.mmd_utils import *
    from utils.model_utils import *
    from utils.conv_utils import *
    from decoder import *
    from encoder import *


# * $ \tilde{K}_XX * e = K_XX * e - diag_X $
# * $ \tilde{K}_YY * e = K_YY * e - diag_Y $ 
# * $ K_{XY}^T * e $
# 
# 
# * $ e^T * \tilde{K}_XX * e $
# * $ e^T * \tilde{K}_YY * e $ 
# * $ e^T * K_{XY} * e $

# In[ ]:


# NetG is a decoder
# input: batch_size * nz * 1 * 1
# output: batch_size * nc * image_size * image_size
class NetG(nn.Module):
    def __init__(self, decoder):
        super(NetG, self).__init__()
        self.decoder = decoder

    def forward(self, input):
        output = self.decoder(input)
        return output


# In[ ]:


# NetD is an encoder + decoder
# input: batch_size * nc * image_size * image_size
# f_enc_X: batch_size * k * 1 * 1
# f_dec_X: batch_size * nc * image_size * image_size
class NetD(nn.Module):
    def __init__(self, encoder, decoder):
        super(NetD, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input):
        f_enc_X = self.encoder(input)
#         print("f_enc_X size = " + str(f_enc_X.size()))
#         print("f_enc_X outputted.")
        
#         # The FC Layers
#         f_enc_X = self.fc1(f_enc_X)
#         f_enc_X = self.fc1(f_enc_X)
        
        
        f_dec_X = self.decoder(f_enc_X)
#         print("f_dec_X size = " + str(f_dec_X.size()))
#         print("f_dec_X outputted.")

        f_enc_X = f_enc_X.view(input.size(0), -1)
        f_dec_X = f_dec_X.view(input.size(0), -1)
        return f_enc_X, f_dec_X


# In[ ]:


if run_in_jupyter:
    get_ipython().run_line_magic('run', 'one_sided.py')
else:
    from one_sided import *


# In[ ]:


# if args.experiment is None:
#     args.experiment = 'samples'
# os.system('mkdir {0}'.format(args.experiment))

if model_save_folder is None:
    model_save_folder = 'samples'
os.system('mkdir {0}'.format(model_save_folder))


# In[ ]:


if torch.cuda.is_available():
#     args.cuda = True
    cuda = True
#     torch.cuda.set_device(args.gpu_device)
    torch.cuda.set_device(gpu_device)
    print("Using GPU device", torch.cuda.current_device())
else:
    raise EnvironmentError("GPU device not available!")


# In[ ]:


# np.random.seed(seed=args.manual_seed)
# random.seed(args.manual_seed)
# torch.manual_seed(args.manual_seed)
# torch.cuda.manual_seed(args.manual_seed)
# cudnn.benchmark = True

np.random.seed(seed=manual_seed)
random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)
cudnn.benchmark = True


# In[ ]:


# construct encoder/decoder modules
hidden_dim = nz


print("\nGenerator")
G_decoder = Decoder(embedded_cube_dimension = 2,
                    fc1_hidden_dim = 0, 
                    fc2_output_dim = 0, 
                    embedding_dim = 0, 
                    leakyrelu_const = 0.2)
print("\nDiscriminator")
D_encoder = Encoder(cube_dimension = 128,
                    fc1_hidden_dim = 0, 
                    fc2_output_dim = 0,
                    embedding_dim = 0, 
                    leakyrelu_const = 0.2,  
                    pool_return_indices = False)
D_decoder = Decoder(embedded_cube_dimension = 2,
                    fc1_hidden_dim = 0, 
                    fc2_output_dim = 0, 
                    embedding_dim = 0, 
                    leakyrelu_const = 0.2)


# In[ ]:


netD = NetD(D_encoder, D_decoder)
print("netD:", netD)


# In[ ]:


netG = NetG(G_decoder)
print("netG:", netG)


# In[ ]:


one_sided = ONE_SIDED()
print("\n \n oneSide:", one_sided)


# In[ ]:


netG.apply(weights_init)
netD.apply(weights_init)
one_sided.apply(weights_init)


# In[ ]:


# put variable into cuda device
fixed_noise = torch.cuda.FloatTensor(64, nz, 1, 1).normal_(0, 1)
one = torch.tensor(1.0).cuda()
#one = torch.cuda.FloatTensor([1])
mone = one * -1
if cuda:
    netG.cuda()
    netD.cuda()
    one_sided.cuda()
fixed_noise = Variable(fixed_noise, 
                       requires_grad=False)


# In[ ]:


if optimizer_choice == "rmsprop":
#     setup optimizer
    optimizerG = torch.optim.RMSprop(netG.parameters(), 
                                     lr=lr)
    optimizerD = torch.optim.RMSprop(netD.parameters(), 
                                     lr=lr)
elif optimizer_choice == "adam":
    # Why not try adam?
    optimizerG = torch.optim.Adam(netG.parameters(), 
                                     lr=lr)
    optimizerD = torch.optim.Adam(netD.parameters(), 
                                     lr=lr)

    


# In[ ]:


time_loop = timeit.default_timer()
print("time = " + str(time_loop))

time_1_list = []
time_2_list = []

gen_iterations = 0  # the code default is = 0

# lists for tracking - Discriminator side
mmd2_D_before_ReLU_list = []
mmd2_D_after_ReLU_list = []
one_side_errD_list = []
L2_AE_X_D_list = []
L2_AE_Y_D_list = []
errD_list = []

# lists for tracking - Generator side
mmd2_G_before_ReLU_list = []
mmd2_G_after_ReLU_list = []
one_side_errG_list = []
errG_list = []
# errG = torch.Tensor(np.array(0.0))
# print(errG.item())

# lists for tracking count of nonzero voxels
log_nonzero_recon_over_real_list = []

# list for tracking gradient norms for generator and discriminator
grad_norm_D = []
grad_norm_G = []

# lists for tracking the sum of all cubes in a minibatch
sum_noise_gen = []
sum_noise_gen_recon = []
sum_real = []
sum_real_recon = []


for t in range(max_iter):
    print("\n-----------------------------------------------")
    print("Epoch = " + str(t+1) + " / " + str(max_iter))
    print("----------------------------------------------- \n")
    
    data_iter = iter(trn_loader)
    print("len(trn_loader) = " + str(len(trn_loader)))
    i = 0
    plotted = 0
    plotted_2 = 0
    plotted_3 = 0
    plotted_4 = 0   # grad norm plotting controller
    
    while (i < len(trn_loader)):
        
        # ---------------------------
        #        Optimize over NetD
        # ---------------------------
        print("Optimize over NetD")
        for p in netD.parameters():
            p.requires_grad = True

            
        """
        What does the below if-else do?
        Trains the discriminator for a lot more when the training
        is starting, then switches to a more frequent generator
        training regime.
        """
        print("gen_iterations = " + str(gen_iterations))
        if gen_iterations < gen_iterations_limit or gen_iterations % 500 == 0:
            Diters = Diter_1
            Giters = Giter_1
        else:
            Diters = Diter_2
            Giters = Giter_2

        for j in range(Diters):
            if i == len(trn_loader):
                break

            time_1 = time.time()
            print("j / Diter = " + str(j+1) + " / " + str(Diters))
            # clamp parameters of NetD encoder to a cube
            # do not clamp parameters of NetD decoder!!!
            # exactly like numpy.clip()
            """
            Given an interval, values outside the interval are clipped to the interval edges. 
            For example, if an interval of [0, 1] is specified, values smaller than 0 become 0, 
            and values larger than 1 become 1.
            
            Below code clamps the encoder parameters of the 
            dsicriminator between -0.01 and 0.01
            """
            for p in netD.encoder.parameters():
                p.data.clamp_(left_clamp, right_clamp)

            data = data_iter.next()
#             print("data shape = " + str(data.shape))
            
            i += 1
            
            netD.zero_grad()

#             x_cpu, _ = data
            x_cpu = data
            x = Variable(x_cpu.cuda().float())
            batch_size = x.size(0)
#             print("batch_size = " + str(batch_size))

            # output of the discriminator with real data input
            """
            2097152^(1/3) = 128 (= one side of our cube so the
            reconstructed cube is the same size as the original one)
            This one just acts like an autoencoder
            """
            f_enc_X_D, f_dec_X_D = netD(x)
#             sum_real.append(x.sum())
#             sum_real_recon.append(f_dec_X_D.sum())
#             print("netD(x) outputs:")
#             print("f_enc_X_D size = " + str(f_enc_X_D.size()))
#             print("f_dec_X_D size = " + str(f_dec_X_D.size()))
#             print("f_dec_X_D min = " + str(f_dec_X_D.min().item()))
#             print("f_dec_X_D max = " + str(f_dec_X_D.max().item()))
#             print("f_dec_X_D mean = " + str(f_dec_X_D.mean().item()))
            

            noise = torch.cuda.FloatTensor(batch_size, 
                                            nz, 
                                            2, 
                                            2,
                                            2).normal_(0, 1)
            with torch.no_grad():
                #noise = Variable(noise, volatile=True)  # total freeze netG
                noise = Variable(noise)
#             print("noise shape = " + str(noise.shape))

            # output of the generator with noise input
#             y = Variable(netG(noise).data)
            y = Variable(netG(noise))
#             sum_noise_gen.append(y.sum())
#             print("y shape = " + str(y.shape))
#             print("y[0] shape = " + str(y[0].shape))
#             print("y[0][0] shape = " + str(y[0][0].shape))
#             sample_cube_viz = y[0][0].cpu().detach().numpy()
#             print("sample_cube_viz shape = " + str(sample_cube_viz.shape))
        
            # output of the discriminator with noise input
            # this tests discriminator 
            f_enc_Y_D, f_dec_Y_D = netD(y)
#             sum_noise_gen_recon.append(f_dec_Y_D.sum())
#             print("netD(y) outputs:")
#             print("f_enc_Y_D size = " + str(f_enc_Y_D.size()))
#             print("f_dec_Y_D size = " + str(f_dec_Y_D.size()))
#             print("f_dec_Y_D min = " + str(f_dec_Y_D.min().item()))
#             print("f_dec_Y_D max = " + str(f_dec_Y_D.max().item()))
#             print("f_dec_Y_D mean = " + str(f_dec_Y_D.mean().item()))

            # compute biased MMD2 and use ReLU to prevent negative value
            mmd2_D = mix_rbf_mmd2(f_enc_X_D, 
                                  f_enc_Y_D, 
                                  sigma_list)
#             mmd2_D = poly_mmd2(f_enc_X_D, f_enc_Y_D)
#             mmd2_D = linear_mmd2(f_enc_X_D, f_enc_Y_D)
            
#             print("mmd2_D before ReLU = " + str(mmd2_D.item()))
            mmd2_D_before_ReLU_list.append(mmd2_D.item())
            mmd2_D = F.relu(mmd2_D)
#             print("mmd2_D after ReLU = " + str(mmd2_D.item()))
            mmd2_D_after_ReLU_list.append(mmd2_D.item())

            # compute rank hinge loss
#             print('f_enc_X_D:', f_enc_X_D.size())
#             print('f_enc_Y_D:', f_enc_Y_D.size())
            one_side_errD = one_sided(f_enc_X_D.mean(0) - f_enc_Y_D.mean(0))
#             print("one_side_errD = " + str(one_side_errD.item()))
            one_side_errD_list.append(one_side_errD.item())
            
            # compute L2-loss of AE
            """
            These L2 losses are decreasing like a standard optimization
            which means that the autoencoder is learning how to encode
            and decode using 3D convolutions.
            x = real cube (x batch_size)
            y = cube generated by the Generator with noise input
            f_dec_X_D = AE reconstructed real cube
            f_dec_Y_D = AE reconstructed noise-input cube
            """
#             print('f_dec_X_D:', f_dec_X_D.size())
#             print('f_dec_Y_D:', f_dec_Y_D.size())
#             print('x:', x.size())
#             print('y:', y.size())
            L2_AE_X_D = match(x.view(batch_size, -1), f_dec_X_D, dist_ae)
            L2_AE_Y_D = match(y.view(batch_size, -1), f_dec_Y_D, dist_ae)
            
#             print("L2-loss of AE, L2_AE_X_D = " + str(L2_AE_X_D.item()))
#             print("L2-loss of AE, L2_AE_Y_D = " + str(L2_AE_Y_D.item()))
            L2_AE_X_D_list.append(L2_AE_X_D.item())
            L2_AE_Y_D_list.append(L2_AE_Y_D.item())
            


#             print("lambda_rg = " + str(lambda_rg))
            errD = torch.sqrt(mmd2_D) + lambda_rg * one_side_errD - lambda_AE_X * L2_AE_X_D - lambda_AE_Y * L2_AE_Y_D
#             print("errD shape = " + str(errD.shape))
#             print("errD = " + str(errD.item()))
            errD_list.append(errD.item())
            errD.backward(mone)
            optimizerD.step()
            
            time_2 = time.time()  
            time_2 = time_2 - time_1
            time_2_list.append(time_2)
            print(np.mean(np.array(time_2_list)))
        

            
            # Plotting Discriminator Plots
            if j % 2 == 0 and plotted < 1:
                if True:
#                 try:
                    """
                    Plotting Different Discriminator Related Values
                    """
                    plot_list = [mmd2_D_before_ReLU_list,mmd2_D_after_ReLU_list,
                                 one_side_errD_list, L2_AE_X_D_list,
                                 L2_AE_Y_D_list, errD_list ]
                    plot_title_list = ["mmd2_D_before_ReLU_list", "mmd2_D_after_ReLU_list",
                                       "one_side_errD_list", "L2_AE_X_D_list",
                                       "L2_AE_Y_D_list", "errD_list - D loss goes to 0: failure mode"]
                    for plot_no in range(len(plot_list)):
                        mmd_loss_plots(fig_id = plot_no, 
                                        fig_title = plot_title_list[plot_no], 
                                        data = plot_list[plot_no], 
                                        show_plot = True, 
                                        save_plot = False, 
                                        redshift_fig_folder = redshift_fig_folder,
                                      t = t)

                    """
                    Plotting the sum of values across a minibatch
                    """
                    plot_minibatch_value_sum(sum_real = sum_real,
                             sum_real_recon = sum_real_recon,
                             sum_noise_gen = sum_noise_gen,
                             sum_noise_gen_recon = sum_noise_gen_recon,
                             save_plot = True,
                             show_plot = True,
                             redshift_fig_folder = redshift_fig_folder,
                             t = t)

                    
                    """
                    Plotting Nominal and Log Histograms
                    """
                    # plot output of the discriminator with real data input
                    # and output of the discriminator with noise input
                    # on the same histogram 
                    # selecting a random cube from the batch
                    random_batch = random.randint(0,batch_size-1)
                    real_ae_cube = f_dec_X_D[random_batch].cpu().view(128,128,128).detach().numpy()
                    noise_ae_cube = f_dec_Y_D[random_batch].cpu().view(128,128,128).detach().numpy()
                    noise_gen_cube = y[random_batch][0].cpu().detach().numpy()
                    real_cube = x[random_batch][0].cpu().detach().numpy()
                                        
                    # inverse transform the real and generated cubes back to normal
                    real_ae_cube = inverse_transform_func(cube = real_ae_cube,
                                                  inverse_type = inverse_transform, 
                                             sampled_dataset = sampled_subcubes)
                    noise_ae_cube = inverse_transform_func(cube = noise_ae_cube,
                                                  inverse_type = inverse_transform, 
                                             sampled_dataset = sampled_subcubes)
                    noise_gen_cube = inverse_transform_func(cube = noise_gen_cube,
                                                  inverse_type = inverse_transform, 
                                             sampled_dataset = sampled_subcubes)
                    real_cube = inverse_transform_func(cube = real_cube,
                                                  inverse_type = inverse_transform, 
                                             sampled_dataset = sampled_subcubes)
                    
                    # using the inverse-transformed randomly selected samples
                    sum_real.append(real_cube.sum())
                    sum_real_recon.append(real_ae_cube.sum())
                    sum_noise_gen.append(noise_gen_cube.sum())
                    sum_noise_gen_recon.append(noise_ae_cube.sum())
                    
                    print("real_ae_cube max = " + str(real_ae_cube.max()) + ", min = " + str(real_ae_cube.min())                      + ", mean = " + str(real_ae_cube.mean()))
                    print("noise_ae_cube max = " + str(noise_ae_cube.max()) + ", min = " + str(noise_ae_cube.min())                         + ", mean = " + str(noise_ae_cube.mean()))
                    print("noise_gen_cube max = " + str(noise_gen_cube.max()) + ", min = " + str(noise_gen_cube.min())                         + ", mean = " + str(noise_gen_cube.mean()))
                    print("real_cube max = " + str(real_cube.max()) + ", min = " + str(real_cube.min())                         + ", mean = " + str(real_cube.mean()))
                    
                    
                    """
                    Power Spectrum Comparisons
                    """
                    plot_power_spec(real_cube = real_cube,        # should be inverse_transformed
                                    generated_cube = noise_gen_cube,   # should be inverse_transformed
                                    raw_cube_mean = sampled_subcubes.mean_val, 
                                    save_plot = True,
                                    show_plot = True,
                                     redshift_fig_folder = redshift_fig_folder,
                                     t = t,
                                    threads=1, 
                                    MAS="CIC", 
                                    axis=0, 
                                    BoxSize=75.0/2048*128)
                    
                    
                    real_ae_cube = real_ae_cube[np.nonzero(real_ae_cube)]
                    noise_ae_cube = noise_ae_cube[np.nonzero(noise_ae_cube)]
                    noise_gen_cube = noise_gen_cube[np.nonzero(noise_gen_cube)]
                    real_cube = real_cube[np.nonzero(real_cube)]
    #                 recon_plot = recon_plot[np.greater(recon_plot, 0)]
                    
#                     print("len(real_plot) - nonzero elements = " + str(len(real_plot)))
#                     print("len(recon_plot) - nonzero elements = " + str(len(recon_plot)))
    #                 log_nonzero_real_list.append(len(real_plot))
    #                 log_nonzero_recon_list.append(len(recon_plot))

#                     log_nonzero_recon_over_real_list.append(len(recon_plot) / len(real_plot))
                    
                    mmd_hist_plot(noise = noise_gen_cube, 
                                  real = real_cube, 
                                  recon_noise = noise_ae_cube, 
                                  recon_real = real_ae_cube,
                                  epoch = t, 
                                  file_name = 'hist_' + str(t) + '.png', 
                                  plot_pdf = False,
                                  log_plot = False,
                                  plot_show = True,
                                  redshift_fig_folder = redshift_fig_folder)
    
                    mmd_hist_plot(noise = noise_gen_cube, 
                                  real = real_cube, 
                                  recon_noise = noise_ae_cube, 
                                  recon_real = real_ae_cube,
                                  epoch = t, 
                                  file_name = 'pdf_' + str(t) + '.png', 
                                  plot_pdf = True,
                                  log_plot = False,
                                  plot_show = True,
                                  redshift_fig_folder = redshift_fig_folder) 
                    
                    """
                    Plotting the log histograms & PDF
                    """
                    mmd_hist_plot(noise = noise_gen_cube, 
                                  real = real_cube, 
                                  recon_noise = noise_ae_cube, 
                                  recon_real = real_ae_cube,
                                  epoch = t, 
                                  file_name = 'hist_log_' + str(t) + '.png', 
                                  plot_pdf = False,
                                  log_plot = True,
                                  plot_show = True,
                                  redshift_fig_folder = redshift_fig_folder)
    
                    mmd_hist_plot(noise = noise_gen_cube, 
                                  real = real_cube, 
                                  recon_noise = noise_ae_cube, 
                                  recon_real = real_ae_cube,
                                  epoch = t, 
                                  file_name = 'pdf_log_' + str(t) + '.png', 
                                  plot_pdf = True,
                                  log_plot = True,
                                  plot_show = True,
                                  redshift_fig_folder = redshift_fig_folder)                     

#                 except:
#                     pass
                
                plotted = plotted + 1
                
                
                
            if plotted_2 < 1 and t % 5 == 0 and t > gen_iterations_limit:
 
                # reshaping DOESNT WORK due to nonzero() -> reshaping 1D to 3D with cube_size edges
                # so just getting them again works.
                real_ae_cube = f_dec_X_D[random_batch].cpu().view(128,128,128).detach().numpy()
                noise_ae_cube = f_dec_Y_D[random_batch].cpu().view(128,128,128).detach().numpy()
                noise_gen_cube = y[random_batch][0].cpu().detach().numpy()
                real_cube = x[random_batch][0].cpu().detach().numpy()

                real_ae_cube = inverse_transform_func(cube = real_ae_cube,
                                                  inverse_type = inverse_transform, 
                                             sampled_dataset = sampled_subcubes)
                noise_ae_cube = inverse_transform_func(cube = noise_ae_cube,
                                                  inverse_type = inverse_transform, 
                                             sampled_dataset = sampled_subcubes)
                noise_gen_cube = inverse_transform_func(cube = noise_gen_cube,
                                                  inverse_type = inverse_transform, 
                                             sampled_dataset = sampled_subcubes)
                real_cube = inverse_transform_func(cube = real_cube,
                                                  inverse_type = inverse_transform, 
                                             sampled_dataset = sampled_subcubes)

                print("real_ae_cube shape = " + str(real_ae_cube.shape))
                print("noise_ae_cube shape = " + str(noise_ae_cube.shape))
                print("noise_gen_cube shape = " + str(noise_gen_cube.shape))
                print("real_cube shape = " + str(real_cube.shape))
            
            
#                 # Plot the 3D Cubes
                print("\nReconstructed, AutoEncoder Generated Real Cube")
#                 recon_real_viz = 
                visualize_cube(cube=real_ae_cube,      ## array name
                                         edge_dim=real_ae_cube.shape[0],        ## edge dimension (128 for 128 x 128 x 128 cube)
                                         start_cube_index_x=0,
                                         start_cube_index_y=0,
                                         start_cube_index_z=0,
                                         fig_size=(10,10),
                                         #stdev_to_white=-2,
                                         norm_multiply=viz_multiplier,
                                            size_magnitude = scatter_size_magnitude,
                                         color_map="Blues",
                                         plot_show = plot_show_3d,
                               plot_save = plot_save_3d,
                                         save_fig = redshift_3dfig_folder + 'recon_ae_real_' + str(t) + '.png')
                
                print("\nReconstructed, AutoEncoder Generated Noise-Input Cube")
#                 recon_fake_viz = 
                visualize_cube(cube=noise_ae_cube,      ## array name
                                         edge_dim=noise_ae_cube.shape[0],        ## edge dimension (128 for 128 x 128 x 128 cube)
                                         start_cube_index_x=0,
                                         start_cube_index_y=0,
                                         start_cube_index_z=0,
                                         fig_size=(10,10),
                                         #stdev_to_white=-2,
                                         norm_multiply=viz_multiplier,
                                            size_magnitude = scatter_size_magnitude,
                                         color_map="Blues",
                                         plot_show = plot_show_3d,
                               plot_save = plot_save_3d,
                                         save_fig = redshift_3dfig_folder + 'recon_ae_noisegen_' + str(t) + '.png')
                
                print("\nNoise-Input Generated Cube")
#                 sample_viz = 
                visualize_cube(cube=noise_gen_cube,      ## array name
                                         edge_dim=noise_gen_cube.shape[0],        ## edge dimension (128 for 128 x 128 x 128 cube)
                                         start_cube_index_x=0,
                                         start_cube_index_y=0,
                                         start_cube_index_z=0,
                                         fig_size=(10,10),
                                         #stdev_to_white=-2,
                                         norm_multiply=viz_multiplier,
                                            size_magnitude = scatter_size_magnitude,
                                         color_map="Blues",
                                         plot_show = plot_show_3d,
                               plot_save = plot_save_3d,
                                         save_fig = redshift_3dfig_folder + 'noisegen_' + str(t) + '.png')
                
                print("\nReal Cube")
#                 real_viz = 
                visualize_cube(cube=real_cube,      ## array name
                                         edge_dim=real_cube.shape[0],        ## edge dimension (128 for 128 x 128 x 128 cube)
                                         start_cube_index_x=0,
                                         start_cube_index_y=0,
                                         start_cube_index_z=0,
                                         fig_size=(10,10),
                                         #stdev_to_white=-2,
                                         norm_multiply=viz_multiplier,
                                            size_magnitude = scatter_size_magnitude,
                                         color_map="Blues",
                                         plot_show = plot_show_3d,
                               plot_save = plot_save_3d,
                                         save_fig = redshift_3dfig_folder +'real_' + str(t) + '.png')

#             sample_viz.show()

                plotted_2 = plotted_2 + 1 # to limit one 3d plotting per epoch
                
        print("\n Finished optimizing over NetD \n")


        # ---------------------------
        #        Optimize over NetG
        # ---------------------------
        """
        Because i is increased in each training loop for the
        discriminitor, the below condition of if i == len(trn_loader)
        is True in every epoch.
        Should an i = 0 be added to the beginning of the netG optimization?
        Look at paper to see how the training method is.
        """
        print("Optimize over NetG")
        for p in netD.parameters():
            p.requires_grad = False

        print("Giters = " + str(Giters))
        for j in range(Giters):
            print("i = " + str(i))
            print("len(trn_loader) = " + str(len(trn_loader)))
            if i == len(trn_loader):
                print("Breaking from the Generator training loop")
                break

            print("j / Giter = " + str(j+1) + " / " + str(Giters))
            data = data_iter.next()
            i += 1
            netG.zero_grad()

#             x_cpu, _ = data
            x_cpu = data
            x = Variable(x_cpu.cuda().float())
            batch_size = x.size(0)

            # output of discriminator with real input
            f_enc_X, f_dec_X = netD(x)

            noise = torch.cuda.FloatTensor(batch_size, 
                                           nz, 
                                           2,
                                           2,
                                           2).normal_(0, 1)
            noise = Variable(noise)
            
            # output of the generator with noise input
            y = netG(noise)

            # output of the discriminator with noise input
            f_enc_Y, f_dec_Y = netD(y)

            # compute biased MMD2 and use ReLU to prevent negative value
            mmd2_G = mix_rbf_mmd2(f_enc_X, f_enc_Y, sigma_list)
#             mmd2_G = poly_mmd2(f_enc_X, f_enc_Y)
#             mmd2_G = linear_mmd2(f_enc_X, f_enc_Y)
    
            mmd2_G_before_ReLU_list.append(mmd2_G)
            mmd2_G = F.relu(mmd2_G)
            mmd2_G_after_ReLU_list.append(mmd2_G)

            # compute rank hinge loss
            one_side_errG = one_sided(f_enc_X.mean(0) - f_enc_Y.mean(0))
            one_side_errG_list.append(one_side_errG)

            errG = torch.sqrt(mmd2_G) + lambda_rg * one_side_errG
            print("errG = " + str(errG.item()))
#             print("one = ") + str(one)
            errG_list.append(errG.item())
            errG.backward(one)
            optimizerG.step()

            gen_iterations += 1
            
            if plotted_3 < 1:
                """
                Plotting Generator Related Values
                """
                plot_list = [mmd2_G_before_ReLU_list,mmd2_G_after_ReLU_list,
                             one_side_errG_list, errG_list ]
                plot_title_list = ["mmd2_G_before_ReLU_list", "mmd2_G_after_ReLU_list",
                                   "one_side_errG_list","errG_list"]
                for plot_no in range(len(plot_list)):
                    mmd_loss_plots(fig_id = plot_no, 
                                    fig_title = plot_title_list[plot_no], 
                                    data = plot_list[plot_no], 
                                    show_plot = True, 
                                    save_plot = False, 
                                    redshift_fig_folder = redshift_fig_folder,
                                  t = t)           
            
                plotted_3 = plotted_3 + 1

        run_time = (timeit.default_timer() - time_loop) / 60.0
        print("run_time = " + str(run_time))
        try:
            print('[%3d/%3d][%3d/%3d] [%5d] (%.2f m) MMD2_D %.6f hinge %.6f L2_AE_X %.6f L2_AE_Y %.6f loss_D %.6f Loss_G %.6f f_X %.6f f_Y %.6f |gD| %.4f |gG| %.4f'
                % (t, max_iter, i, len(trn_loader), gen_iterations, run_time,
                     mmd2_D.item(), one_side_errD.item(),
                     L2_AE_X_D.item(), L2_AE_Y_D.item(),
                     errD.item(), errG.item(),
                     f_enc_X_D.mean().item(), f_enc_Y_D.mean().item(),
                     grad_norm(netD), grad_norm(netG)))
        except:
            pass

        
        # plotting gradient norms for monitoring
        grad_norm_D.append(grad_norm(netD))
        grad_norm_G.append(grad_norm(netG))
        
        if plotted_4 < 1:
            plt.figure(figsize = (10,5))
            plt.title("grad_norms - if they are over 100 things are screwing up")
            plt.plot(grad_norm_D, 
                     color = "red", 
                     label = "grad_norm_D")
            plt.plot(grad_norm_G, 
                     color = "blue", 
                     label = "grad_norm_G")
            plt.legend()
            plt.savefig(redshift_fig_folder + 'grad_norms_' + str(t) + '.png', 
                        bbox_inches='tight')
            plt.show() 
            plt.close()
            #             plt.show()
            
            plotted_4 = plotted_4 + 1


    if t % save_model_every == 0:
        torch.save(netG.state_dict(), 
                   '{0}/netG_iter_{1}.pth'.format(model_save_folder, t))
        torch.save(netD.state_dict(), 
                   '{0}/netD_iter_{1}.pth'.format(model_save_folder, t))
        
        


# # Testing

# In[ ]:


if in_testing == False:
    assert in_testing, "Stopping here, because not in testing..."


# ## Load Optimized Model

# In[ ]:


print("The folder models were saved in: " + str(model_save_folder))
model_files = [f for f in listdir(model_save_folder) if isfile(join(model_save_folder, f))]
model_files

netG_files = [f for f in model_files if "netG" in f]
netG_files

max_iter_netG = max(netG_files, key=lambda x: int(x[10:-4]))
max_iter_netG


# In[ ]:


hidden_dim = nz
G_decoder = Decoder(cube_size, 
                    nc, 
                    k=nz, 
                    ngf=16)

netG_test = NetG(G_decoder)
# print("netG:", netG_test)


# In[ ]:


netG_test.load_state_dict(torch.load(model_save_folder + max_iter_netG))


# In[ ]:


netG_test.eval()
netG_test.cuda()


# ## Generate Cube with Trained Generator

# In[ ]:


noise = torch.cuda.FloatTensor(1, 
                                nz, 
                                1, 
                                1,
                                1).normal_(0, 1)
noise.size()


# In[ ]:


with torch.no_grad():
    noise = Variable(noise)

    # output of the generator with noise input
    y = netG_test(noise)


# In[ ]:


random_batch = random.randint(0,batch_size-1)
noise_gen_cube = y[0][0].cpu().detach().numpy()


# In[ ]:


print("Noise-Input Generated Cube")
#                 sample_viz = 
visualize_cube(cube=noise_gen_cube,      ## array name
                         edge_dim=noise_gen_cube.shape[0],        ## edge dimension (128 for 128 x 128 x 128 cube)
                         start_cube_index_x=0,
                         start_cube_index_y=0,
                         start_cube_index_z=0,
                         fig_size=(10,10),
                         stdev_to_white=-2,
                         norm_multiply=viz_multiplier,
                         color_map="Blues",
                         plot_show = True,
                         save_fig = False)


# In[ ]:


np.save(file = testing_folder + "noise_gen_cube",
    arr = noise_gen_cube,
    allow_pickle = True)


# ## Load a real subcube

# In[ ]:


testcd = define_test(s_test = 1024,
                     s_train = 128)
print(testcd)

trial_sample = get_samples(s_sample = 128, 
                            nsamples = 1, 
#                             h5_filename = redshift_file, 
                            test_coords = testcd,
                            f = f)
trial_sample[0].shape


# In[ ]:


print("Real Sampled Cube")
#                 sample_viz = 
visualize_cube(cube=trial_sample[0],      ## array name
                         edge_dim=trial_sample[0].shape[0],        ## edge dimension (128 for 128 x 128 x 128 cube)
                         start_cube_index_x=0,
                         start_cube_index_y=0,
                         start_cube_index_z=0,
                         fig_size=(10,10),
                         stdev_to_white=-2,
                         norm_multiply=viz_multiplier,
                         color_map="Blues",
                         plot_show = True,
                         save_fig = False)


# In[ ]:


np.save(file = testing_folder + "real_cube",
    arr = trial_sample[0],
    allow_pickle = True)


# ## Compare Generated vs. Real with Power Spectrum

# In[ ]:


# import pyfftw


# In[ ]:


# !cd seda_pylians/Pylians
# import Pk_library as PKL


# In[ ]:





# In[ ]:




