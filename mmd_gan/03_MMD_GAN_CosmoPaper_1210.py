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
#     * redshift_raw_file = raw data file like fields_z=0.0.hdf5
#     * redshift_file = transformed data file like minmax_scale_neg11_redshift0.h5
#     * inverse_transform = one of minmax11 / minmaxneg11 / std_noshift / std based on the transformed data file
#     * gen_iterations_limit
# * change the file name inside the run-mmdgan-110919.sbatch file
# * do: sbatch run-mmdgan-110918.sbatch

# #### To transfer files to google drive
# * module load rclone/1.38
# * rclone copy /scratch/jjz289/data/mmd_gan_code/folderX/ remote1:folderX

# #### Presentation
# 
# Note the following parameters for debug:
# * batch_size
# * nz
# * lr
# * optimizer_choice
# * dist_ae
# * left_clamp
# * right_clamp
# * redshift_raw_file
# * redshift_file
# * inverse_transform
# * gen_iterations_limit
# * encoder architecture
# * decoder architecture
# 
# Following graphs should be included (the last saved ones):
# * 
# 
# 

# In[1]:


import graphviz
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable, grad
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
import re
import shutil


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


run_mode = "continue"   # training OR continue = continue if load another model is loaded and continued to train
continue_train_folder = "jupyter-output-143"    # the folder where the previous run is for the saved model to be trained further
                            # make sure that the below parameters are the same as well
netD_iter_file = "netD_iter_0.pth"         # netD_iter_xx.pth file that contains the state dict under models/
netG_iter_file = "netG_iter_0.pth"         # netG_iter_xx.pth file that contains the state dict under models/


# In[5]:


batch_size = 96       # BATCH_SIZE: batch size for training
gpu_device = 0        # GPU_DEVICE: gpu id (default 0)
nc = 1                # NC: number of channels in images
cube_size = 64       # for our dataset more like one edge of the subcube
lr = 5e-5               # LR: learning rate - default: 5e-5 (rmsprop) , 1e-4:adam
max_iter = 150         # MAX_ITER: max iteration for training
optimizer_choice = "rmsprop"     # adam or rmsprop
dist_ae = 'L2'                  # "L2" or "L1" -> Autoencoder reconstructruced cube loss choice,  "cos" doesnt work
manual_seed = 1
sample_size_multiplier = 128 * 5
n_samples = batch_size * sample_size_multiplier      # on prince, number of samples to get from the training cube
Diter_1 = 100    # default: 100
Giter_1 = 1      # default: 1
Diter_2 = 5      # default: 5
Giter_2 = 1      # default: 1
if run_mode == "continue":
    gen_iterations_limit = 25   # default = 25
else:
    gen_iterations_limit = 25
edge_sample = cube_size
edge_test = 512


# In[6]:


lambda_gradpen = 10  # gradient penalty, 0 for not using


# In[7]:


assert n_samples / batch_size > 100, "The gen_iterations wont work properly!"


# In[8]:


adam_beta1 = 0.5     # default: 0.9    # coefficients used for computing running averages of gradient and its square 
adam_beta2 = 0.9     # default: 0.999


# ### Model Options

# In[9]:


# Which model
model_choice = "conv"     # "conv" or "conv_fc"


# In[10]:


if run_in_jupyter:
    get_ipython().run_line_magic('run', 'utils/power_spectrum_utils.py')
else:
    from utils.power_spectrum_utils import *


# In[11]:


"""
for p in netD.encoder.parameters():
    p.data.clamp_(left_clamp, right_clamp)
"""
left_clamp =  -0.01     # default: -0.01
right_clamp = 0.01    # default: 0.01


# In[12]:


kernel_size = 4
padding = 1
stride = 2

ch_mult = 5        # channel multiplier -> increases this fold in every conv layer

full_conv_limit = 4  # = full deconv limit = number of conv layers with batchnorm
full_deconv_limit = full_conv_limit

just_conv_limit = 0 + full_conv_limit   # + full_conv_limit because of layer counting method in class defnitions
just_deconv_limit = just_conv_limit

leakyrelu_const = 0.2 #  DCGAN = 0.2 or 0.01 # leakyrelu

full_fc_limit = 2    # no of FC layers if available


# In[13]:


# nz = 32                # not used in training, change code in testing - NZ: number of channels, hidden dimension in z and codespace
conv_bias = False
deconv_bias = False
fc_bias = True


# In[14]:


model_param_init = "normal"    # normal OR xavier (doesn't work right now)


# ### Plotting Options

# In[15]:


viz_multiplier = 1e2          # the norm multiplier in the 3D visualization
scatter_size_magnitude = False  # change scatter point radius based on the value of the point 
if run_in_jupyter:
    plot_show_3d = True            # shows the 3d scatter plot
    plot_save_3d = True           # whether to save or not as png 
    plot_save_other = True
    plot_show_other = True
else:
    plot_show_3d = False            # shows the 3d scatter plot
    plot_save_3d = True           # whether to save or not as png 
    plot_save_other = True
    plot_show_other = False


# ### Saving Options

# In[16]:


if run_in_jupyter:
    get_ipython().run_line_magic('run', 'utils/logging_utils.py')
else:
    from utils.logging_utils import *


# In[17]:


root_dir = "./"  # this goes to 
data_dir = "../"
new_output_folder = get_output_folder(run_in_jupyter = run_in_jupyter)
# new_output_folder = "drive-output-XX"   # for batch processing
experiment = root_dir + new_output_folder + "/"       # : output directory of saved models
# print(experiment)

model_save_folder = experiment + "model/"
redshift_fig_folder = experiment + "figures/"        # folder to save mmd & related plots
redshift_3dfig_folder = experiment + "3d_figures/"   # folder to save 3D plots
testing_folder = experiment + "testing/"   # folder to save 3D plots

save_model_every = 2               # (every x epoch) frequency to save the model


# ### Dataset Options

# In[18]:


workers = 2          # WORKERS: number of threads to load data
redshift_info_folder = root_dir + "redshift_info/"   # save some info here as pickle to speed up processing
redshift_raw_file = "fields_z=5.0.hdf5"
# redshift_file = "redshift0_4th_root.h5"    # redshift cube to be used
                                        # standardized_no_shift_redshift0.h5
                                        # minmax_scale_01_redshift0.h5
                                        # minmax_scale_neg11_redshift0.h5
                                        # redshift0_4th_root.h5
                                        # redshift0_6th_root.h5
                                        # redshift0_8th_root.h5
                                        # redshift0_16th_root.h5
                                        # redshift0_4th_root_neg11.h5
root = 8 # should be an integer
inverse_transform = "log_scale_01"    # scale_01 / scale_neg11 / root / 
                                # root_scale_01 / root_scale_neg11
                                # log_scale_01
        


# ### Testing Options

# In[ ]:


in_testing = False              # True if doing testing


# ### Debug Utils

# In[ ]:


if run_in_jupyter:
    get_ipython().run_line_magic('run', 'utils/debug_utils.py')
else:
    from utils.debug_utils import log


# In[ ]:


DEBUG = False


# In[ ]:


print("log('asdas') output = " + str(log("asdas")))


# ## Parameter Documentation

# ### Training Parameters

# In[ ]:


print("\nTraining Parameters:")
print("Batch Size = " + str(batch_size))
print("Number of Samples = " + str(n_samples))

print("Learning Rate = " + str(lr))
print("Number of Epochs = " + str(max_iter))
print("Optimizer = " + str(optimizer_choice))
print("Autoencoder Reconstruction Loss  = " + str(dist_ae))
print("gen_iterations_limit = " + str(gen_iterations_limit))
print("Diter_1 = " + str(Diter_1))
print("Giter_1 = " + str(Giter_1))
print("Diter_2 = " + str(Diter_2))
print("Giter_2 = " + str(Giter_2))

print("Length of Edge of a Sampled Subcube = " + str(cube_size))
print("one edge of the test partition of the whole cube = " + str(edge_test))


# ### Model Parameters

# In[ ]:


print("\nModel Parameters:")
print("Number of Channels in Input = " + str(nc))
# print("Hidden Dimension (codespace) (nz)= " + str(nz))
print("Left clamp = " + str(left_clamp))
print("Right clamp = " + str(right_clamp)) 
print("Parameter Init Method = " + str(model_param_init))
print("Convolution Bias = " + str(conv_bias))
print("Deconvolution Bias = " + str(deconv_bias))
print("FC (Encoder & Decoder) Bias = " + str(fc_bias))
print("Channel Multiplier = " + str(ch_mult))
print("Convolution with BatchNorm count = " + str(full_conv_limit))
print("FC Layer count = " + str(full_fc_limit))
print("LeakyReLU constant = " + str(leakyrelu_const))


# #### MMD Parameters

# In[ ]:


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
lambda_rg = 16.0 #16.0   # used in both err calcs

print("lambda_MMD = " + str(lambda_MMD))
print("lambda_AE_X = " + str(lambda_AE_X))
print("lambda_AE_Y = " + str(lambda_AE_Y))
print("lambda_rg = " + str(lambda_rg))


# In[ ]:


"""
sigma for MMD
"""
base = 1e0
sigma_list = [
#               1, 2, 4, 8, 
              16, 32,64,
              128,256,512,
              1028, 2048, 5096, 5096*2, 5096*4, 5096*8]
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

min_var_est = 1e-30 # 1e-30, default:1e-8
print("minimum variance estimated = " + str(min_var_est))


# ### Plotting Parameters 

# In[ ]:


print("\nPlotting Parameters:")
print("Visualization Multiplier = " + str(viz_multiplier))


# ### Saving Parameters

# In[ ]:


print("\nSaving Parameters:")
print("Output folder = " + str(experiment))
print("model_save_folder folder = " + str(model_save_folder))
print("redshift_fig_folder folder = " + str(redshift_fig_folder))
print("redshift_3dfig_folder folder = " + str(redshift_3dfig_folder))
print("testing_folder folder = " + str(testing_folder))


# ### Dataset Parameters

# In[ ]:


print("\nDataset Parameters:")
# print("Redshift File Used = " + str(redshift_file))
print("redshift_info_folder = " + str(redshift_info_folder))
print("redshift_raw_file = " + str(redshift_raw_file))
print("inverse_transform = " + str(inverse_transform))
print("root = " + str(root))


# ### Testing Parameters

# In[ ]:


print("\nTesting Parameters:")
print("In testing = " + str(in_testing))


# ### Other Parameters

# In[ ]:


print("\nOther Parameters:")
print("Seed = " + str(manual_seed))


# # Code

# ## Redshift Data Load
# 
# Loading the raw data instead of the transformed data because the transformations are going to be done on the fly.

# In[ ]:


# f = h5py.File(data_dir + redshift_file, 'r')
f = h5py.File(data_dir + redshift_raw_file, 'r')
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


# min_cube,max_cube,mean_cube,stddev_cube = get_stats_cube(redshift_info_folder = redshift_info_folder,
#                                            redshift_file = redshift_file,
#                                            data_dir = data_dir)
min_cube,max_cube,mean_cube,stddev_cube = get_stats_cube(redshift_info_folder = redshift_info_folder,
                                           redshift_file = redshift_raw_file,
                                           data_dir = data_dir)

min_raw_cube,max_raw_cube,mean_raw_cube,stddev_raw_cube = get_stats_cube(redshift_info_folder = redshift_info_folder,
                                           redshift_file = redshift_raw_file,
                                           data_dir = data_dir)
# print("\nTransformed  Data Summary Statistics:")
# print("File = " + str(redshift_file))
# print("Min of data = " + str(min_cube))
# print("Max of data = " + str(max_cube))
# print("Mean of data = " + str(mean_cube))
# print("Stddev of data = " + str(stddev_cube))

print("\nRaw Data Summary Statistics:")
print("File = " + str(redshift_raw_file))
print("Min of raw data = " + str(min_raw_cube))
print("Max of raw data = " + str(max_raw_cube))
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


# ## Dataset & DataLoader

# In[ ]:


# on prince
sampled_subcubes = HydrogenDataset(h5_file=redshift_raw_file,
                                    root_dir = data_dir,
                                    f = h5py.File(data_dir + redshift_raw_file, 'r')["delta_HI"],
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
                                  stddev_raw_cube = stddev_raw_cube,
                                  rotate_cubes = True,
                                  transform = inverse_transform,
                                  root = root)


# In[ ]:


# Get data
trn_loader = torch.utils.data.DataLoader(sampled_subcubes, 
                                         batch_size = batch_size,
                                         shuffle=True, 
                                         num_workers=int(workers))


# ## Checking 3D Plots

# In[ ]:


# # dont run this in batch
# if run_in_jupyter:
#     test_3d_plot(edge_test = edge_test, 
#                  edge_sample = edge_sample,
#                  f = h5py.File(data_dir + redshift_file, 'r')["delta_HI"], 
#                  scatter_size_magnitude = scatter_size_magnitude,
#                  viz_multiplier = viz_multiplier,
#                  plot_save_3d = plot_save_3d,
#                  inverse_transform = inverse_transform,
#                  sampled_subcubes = sampled_subcubes)


# ## Model

# In[ ]:


if run_in_jupyter:
    get_ipython().run_line_magic('run', 'utils/mmd_utils.py')
    get_ipython().run_line_magic('run', 'utils/model_utils.py')
    get_ipython().run_line_magic('run', 'utils/conv_utils.py')

else:
    from utils.mmd_utils import *
    from utils.model_utils import *
    from utils.conv_utils import *


# Load & copy the decoder and encoder files to output folder for easier loading of architectures when resuming training:

# In[ ]:


# if run_mode != "continue":
if model_choice == "conv":
    if run_in_jupyter:
        get_ipython().run_line_magic('run', 'models/decoder_v04_UpsampleConv.py')
        get_ipython().run_line_magic('run', 'models/encoder_v04_UpsampleConv.py')
    else:
        from models.decoder_v04_UpsampleConv import *
        from models.encoder_v04_UpsampleConv import *
        
    shutil.copy("models/decoder_v04_UpsampleConv.py",experiment)
    shutil.copy("models/encoder_v04_UpsampleConv.py",experiment)
elif model_choice == "conv_fc":
    if run_in_jupyter:
        get_ipython().run_line_magic('run', 'models/decoder_FC_v03.py')
        get_ipython().run_line_magic('run', 'models/encoder_FC_v03.py')
    else:
        from models.decoder_FC_v03 import *
        from models.encoder_FC_v03 import *

    shutil.copy("models/decoder_FC_v03.py",experiment)
    shutil.copy("models/encoder_FC_v03.py",experiment)
    
# if run_mode == "continue":
#     if model_choice == "conv":
#         if run_in_jupyter:
#             %run continue_train_folder/decoder_v02.py
#             %run continue_train_folder/encoder_v02.py
#         else:
#             from continue_train_folder.decoder_v02 import *
#             from continue_train_folder.encoder_v02 import *
#     elif model_choice == "conv_fc":
#         if run_in_jupyter:
#             %run continue_train_folder/decoder_FC_v03.py
#             %run continue_train_folder/encoder_FC_v03.py
#         else:
#             from continue_train_folder.decoder_FC_v03 import *
#             from continue_train_folder.encoder_FC_v03 import *


# In[ ]:


if run_in_jupyter:
    get_ipython().run_line_magic('run', 'models/NetD.py')
    get_ipython().run_line_magic('run', 'models/NetG.py')
else:
    from models.NetD import *
    from models.NetG import *


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

if model_choice == "conv":
    print("\nDiscriminator")
    D_encoder = Encoder(full_conv_limit = full_conv_limit,
                        just_conv_limit = just_conv_limit,
                         kernel_size = kernel_size,
                         stride = stride,
                         padding = padding,
                         ch_mult = ch_mult,
                         conv_bias = conv_bias,
                         leakyrelu_const = leakyrelu_const)
    D_decoder = Decoder(ch_mult = ch_mult,
                         kernel_size = kernel_size,
                         stride = stride,
                         padding = padding,
                         deconv_bias = deconv_bias,
                         leakyrelu_const = leakyrelu_const,
                       full_deconv_limit = full_deconv_limit,
                        just_deconv_limit = just_deconv_limit,
                       D_encoder = D_encoder)
    print("\nGenerator")
    G_decoder = Decoder(ch_mult = ch_mult,
                         kernel_size = kernel_size,
                         stride = stride,
                         padding = padding,
                         deconv_bias = deconv_bias,
                         leakyrelu_const = leakyrelu_const,
                        full_deconv_limit = full_deconv_limit,
                        just_deconv_limit = just_deconv_limit,
                       D_encoder = D_encoder)

elif model_choice == "conv_fc":
#     print("\nDiscriminator")
    D_encoder = Encoder(full_conv_limit = full_conv_limit,
                         full_fc_limit = full_fc_limit,
                         ch_mult = ch_mult,
                         conv_bias = conv_bias,
                         fc_bias = fc_bias,
                         leakyrelu_const = leakyrelu_const)
    D_decoder = Decoder(ch_mult = ch_mult,
                         deconv_bias = deconv_bias,
                         leakyrelu_const = leakyrelu_const,
                        full_deconv_limit = full_deconv_limit,
                        full_fc_limit = full_fc_limit,
                        fc_bias = fc_bias,
                        D_encoder = D_encoder) 
#     print("\nGenerator")
    G_decoder = Decoder(ch_mult = ch_mult,
                         deconv_bias = deconv_bias,
                         leakyrelu_const = leakyrelu_const,
                        full_deconv_limit = full_deconv_limit,
                        full_fc_limit = full_fc_limit,
                        fc_bias = fc_bias,
                        D_encoder = D_encoder)
   


# In[ ]:


netD = NetD(D_encoder, D_decoder)
# print("type netD: ", type(netD))
print("netD:", netD)


# In[ ]:


netG = NetG(G_decoder)
print("netG:", netG)


# In[ ]:


one_sided = ONE_SIDED()
print("oneSide:", one_sided)


# Save the models to be used when continuing the training:

# In[ ]:


# torch.save(netD, experiment + "netD")
# torch.save(netG, experiment + "netG")


# In[ ]:


# netD_iter = torch.load(continue_train_folder + "/model/" + netD_iter_file)
# netG_iter = torch.load(continue_train_folder + "/model/" + netG_iter_file)
# netD_iter


# In[ ]:


if run_mode == "continue":
    print("Loading saved models and parameters from file...")
#     netD = 
#     netG = 
#     print(type(netD))
#     print(type(netG))
    
    netD.load_state_dict(state_dict = torch.load(f = continue_train_folder + "/model/" + netD_iter_file))
    netG.load_state_dict(state_dict = torch.load(f = continue_train_folder + "/model/" + netG_iter_file))
    


# #### Network Visualization

# In[ ]:


if run_in_jupyter:
    get_ipython().run_line_magic('run', 'utils/network_viz.py')
else:
    from utils.network_viz import *


# In[ ]:


# dict(netD.named_parameters())


# In[ ]:


x = torch.randn(1,1,cube_size,cube_size,cube_size).requires_grad_(True)
y = netD.encoder(Variable(x))
g = make_dot(y,
         params=dict(list(netD.encoder.named_parameters()) + [('x', x)]))
g.view(directory=experiment, filename="netD_encoder_viz")

z = netG.decoder(Variable(y))
g = make_dot(z,
         params=dict(list(netG.decoder.named_parameters()) + [('z', z)]))
g.view(directory=experiment, filename="netG_decoder_viz")


# #### Weights Initialization

# In[ ]:


if run_mode != "continue":
    netG.apply(lambda x: weights_init(x,init_type = model_param_init))
    netD.apply(lambda x: weights_init(x,init_type = model_param_init))
    one_sided.apply(lambda x: weights_init(x,init_type = model_param_init))
    


# In[ ]:



"""
see the parameters of the networks

The convolutional kernels:
torch.Size([2, 1, 4, 4, 4])

What are these for?
torch.Size([4])
"""
print("Discriminator Encoder:")
for p in netD.encoder.parameters():
    print(p.shape)
print("\nDiscriminator Decoder:")  
for p in netD.decoder.parameters():
    print(p.shape)
print("\nGenerator Decoder:")  
for p in netG.decoder.parameters():
    print(p.shape)
    
# for name, param in netD.encoder.named_parameters():
#     if param.requires_grad:
#         print(str(name) + str(param.shape) + str(param.data))


# In[ ]:


# put variable into cuda device


"""
errD.backward(mone)
optimizerD.step()

errG.backward(one)
optimizerG.step()
"""
one = torch.tensor(1.0).cuda()
#one = torch.cuda.FloatTensor([1])
mone = one * -1


# In[ ]:


if cuda:
    netG.cuda()
    netD.cuda()
    one_sided.cuda()


# #### Optimizer Choice

# In[ ]:


if optimizer_choice == "rmsprop":
    optimizerG = torch.optim.RMSprop(netG.parameters(), 
                                     lr=lr)
    optimizerD = torch.optim.RMSprop(netD.parameters(), 
                                     lr=lr)
elif optimizer_choice == "adam":
    optimizerG = torch.optim.Adam(netG.parameters(), 
                                     lr=lr,
                                  betas = (adam_beta1, adam_beta2))
    optimizerD = torch.optim.Adam(netD.parameters(), 
                                     lr=lr,
                                  betas = (adam_beta1, adam_beta2))

    


# In[ ]:


torch.backends.cudnn.benchmark = False


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

fixed_noise_set = 0

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
            
            start = time.time()
#             print("loop start time = " + str(start))
            
            """
            Given an interval, values outside the interval are clipped to the interval edges. 
            For example, if an interval of [0, 1] is specified, values smaller than 0 become 0, 
            and values larger than 1 become 1.
            
            Below code clamps the encoder parameters of the 
            dsicriminator between -0.01 and 0.01
            """
            for p in netD.encoder.parameters():
                p.data.clamp_(left_clamp, right_clamp)
                
            end = time.time()
#             print("part 1a = " + str(end - start))
            start = end

            data = data_iter.next()
#             print("data shape = " + str(data.shape))

            end = time.time()
#             print("part 1b = " + str(end - start))
            start = end
            
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
            f_enc_X_D, f_dec_X_D, f_enc_X_size = netD(x)
#             sum_real.append(x.sum())
#             sum_real_recon.append(f_dec_X_D.sum())
#             print("netD(x) outputs:")
#             print("f_enc_X_D size = " + str(f_enc_X_D.size()))
#             print("f_dec_X_D size = " + str(f_dec_X_D.size()))
#             print("f_dec_X_D min = " + str(f_dec_X_D.min().item()))
#             print("f_dec_X_D max = " + str(f_dec_X_D.max().item()))
#             print("f_dec_X_D mean = " + str(f_dec_X_D.mean().item()))
            
#             print("nz = " + str(nz))
            noise = torch.cuda.FloatTensor(f_enc_X_size).normal_(0, 1)
            
#             noise = torch.cuda.FloatTensor(f_enc_X_size[0], 
#                                             f_enc_X_size[1], 
#                                             f_enc_X_size[2], 
#                                             f_enc_X_size[3],
#                                             f_enc_X_size[4]).normal_(0, 1)
#             noise = Variable(noise)

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
            f_enc_Y_D, f_dec_Y_D, _ = netD(y)
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
                                  sigma_list,
                                  biased=True)
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
#             print("i = " + str(i))
#             print("t = " + str(t))
#             if i <= Diter_1 and t == 0:
            if False:
                errD = torch.sqrt(mmd2_D) + lambda_rg * one_side_errD                         - lambda_AE_X * L2_AE_X_D - lambda_AE_Y * L2_AE_Y_D 
            else:
                # grad_norm_D = grad_norm(netD) down below.
#                 print("grad_normD = " + str(grad_normD))
                gradnorm_D = calc_gradient_penalty(real_data = x, 
                                               generated_data = y, 
                                               gp_weight = lambda_gradpen, 
                                               netD = netD,
                                                  cuda = cuda,
                                                  sigma_list = sigma_list)
                print("calc_gradient_penalty | gradnorm_D = " + str(gradnorm_D.item()))
                errD = torch.sqrt(mmd2_D) + lambda_rg * one_side_errD                         - lambda_AE_X * L2_AE_X_D - lambda_AE_Y * L2_AE_Y_D                         + gradnorm_D
                

                
#             print("errD shape = " + str(errD.shape))
#             print("errD = " + str(errD.item()))
            errD_list.append(errD.item())
            errD.backward(mone)
            optimizerD.step()
            

            
            time_2 = time.time()  
            time_2 = time_2 - time_1
            time_2_list.append(time_2)
            print(np.mean(np.array(time_2_list)))

            """
            fixed_noise was used in the original implementation for 
            generating some image from the same noise input to see
            the evolution
            """
            if fixed_noise_set == 0:
            
                fixed_noise = torch.cuda.FloatTensor(f_enc_X_size).normal_(0, 1)
                if model_choice == "conv_fc":
                    fixed_noise = fixed_noise[0]  # plot just one cube
                    fixed_noise = fixed_noise.view(1,-1)
                print("Fixed Noise size = " + str(fixed_noise.size()))
#                 fixed_noise = torch.cuda.FloatTensor(1, 
#                                                     f_enc_X_size[1], 
#                                                     f_enc_X_size[2], 
#                                                     f_enc_X_size[3],
#                                                     f_enc_X_size[4]).normal_(0, 1)
                fixed_noise = Variable(fixed_noise, 
                                       requires_grad=False)
                fixed_noise_set = fixed_noise_set + 1
        

            
            # Plotting Discriminator Plots
            if j % 2 == 0 and plotted < 1:
                if True:
#                 try:
                    
    
                    """
                    Plotting Different Discriminator Related Values
                    """
                    print("\nPlotting Different Discriminator Related Values")
    
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
                                        show_plot = plot_show_other, 
                                        save_plot = plot_save_other, 
                                        redshift_fig_folder = redshift_fig_folder,
                                      t = t,
                                      dist_ae = dist_ae)


                    # plot output of the discriminator with real data input
                    # and output of the discriminator with noise input
                    # on the same histogram 
                    # selecting a random cube from the batch
                    random_batch = random.randint(0,batch_size-1)
                    real_ae_cube = f_dec_X_D[random_batch].cpu().view(-1,1).detach().numpy()
                    noise_ae_cube = f_dec_Y_D[random_batch].cpu().view(-1,1).detach().numpy()
                    noise_gen_cube = y[random_batch][0].cpu().view(-1,1).detach().numpy()
                    real_cube = x[random_batch][0].cpu().view(-1,1).detach().numpy()
                    
                    """
                    Plotting the Output of the Decoder vs. Real Values
                    for a randomly selected subcube from the minibatch
                    """
#                     print("\nPlotting Nominal Histogram and PDFs")
#                     mmd_hist_plot(noise = noise_gen_cube, 
#                                   real = real_cube, 
#                                   recon_noise = noise_ae_cube, 
#                                   recon_real = real_ae_cube,
#                                   epoch = t, 
#                                   file_name = 'hist_output_' + str(t) + '.png', 
#                                   plot_pdf = True,
#                                   log_plot = False,
#                                   plot_show = plot_show_other,
#                                   redshift_fig_folder = redshift_fig_folder)
                    
                    
                    
                    # full minibatch power spectrum plot
                    real_ae_cube = f_dec_X_D.view(batch_size,1,cube_size,cube_size,cube_size).cpu().detach().numpy()
#                     print(real_ae_cube.shape)
                    noise_ae_cube = f_dec_Y_D.view(batch_size,1,cube_size,cube_size,cube_size).cpu().detach().numpy()
#                     print(noise_ae_cube.shape)
                    noise_gen_cube = y.cpu().detach().numpy()
#                     print(noise_gen_cube.shape)
                    real_cube = x.cpu().detach().numpy()
#                     print(real_cube.shape)

                    """
                    HISTOGRAM OF THE WHOLE BATCH
                    """
                    print("\nPlotting WHOLE BATCH Nominal Histogram and PDFs")
                    hist_plot_2(noise = noise_gen_cube, 
                                real = real_cube, 
                                log_plot = False, 
                                redshift_fig_folder = redshift_fig_folder,
                                t = t)


                    """
                    2D Visualizations
                    NEEDS TRANSFORMED DATA
                    Inverse-transformed is below
                    """
                    visualize2d(real = real_cube, 
                                fake = noise_gen_cube, 
                                raw_cube_mean = sampled_subcubes.mean_raw_val, 
                                redshift_fig_folder = redshift_fig_folder,
                                t = t,
                                save_plot = plot_save_other, 
                                show_plot = plot_show_other)


                    print("\nTransformed Full-Minibatch Subcubes:")
                    print("real_ae_cube max = " + str(real_ae_cube.max()) + ", min = " + str(real_ae_cube.min())                      + ", mean = " + str(real_ae_cube.mean()))
                    print("noise_ae_cube max = " + str(noise_ae_cube.max()) + ", min = " + str(noise_ae_cube.min())                         + ", mean = " + str(noise_ae_cube.mean()))
                    print("noise_gen_cube max = " + str(noise_gen_cube.max()) + ", min = " + str(noise_gen_cube.min())                         + ", mean = " + str(noise_gen_cube.mean()))
                    print("real_cube max = " + str(real_cube.max()) + ", min = " + str(real_cube.min())                         + ", mean = " + str(real_cube.mean()))
                          
                    
                          
                    
                    
                    """
                    INVERSE TRANSFORMATION
                    inverse transform the real and generated cubes back to normal
                    """
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
                    print("\nInverse Transformed Subcubes:")
                    print("real_ae_cube max = " + str(real_ae_cube.max()) + ", min = " + str(real_ae_cube.min())                      + ", mean = " + str(real_ae_cube.mean()))
                    print("noise_ae_cube max = " + str(noise_ae_cube.max()) + ", min = " + str(noise_ae_cube.min())                         + ", mean = " + str(noise_ae_cube.mean()))
                    print("noise_gen_cube max = " + str(noise_gen_cube.max()) + ", min = " + str(noise_gen_cube.min())                         + ", mean = " + str(noise_gen_cube.mean()))
                    print("real_cube max = " + str(real_cube.max()) + ", min = " + str(real_cube.min())                         + ", mean = " + str(real_cube.mean()))
                    
                    

                    """
                    Plotting the sum of values across a minibatch
                    NEEDS INVERSE-TRANSFORMED DATA
                    """
                    sum_real.append(real_cube.sum())
                    sum_real_recon.append(real_ae_cube.sum())
                    sum_noise_gen.append(noise_gen_cube.sum())
                    sum_noise_gen_recon.append(noise_ae_cube.sum())
                    
                    print("\nPlotting the sum of values across a minibatch")
                    plot_minibatch_value_sum(sum_real = sum_real,
                             sum_real_recon = sum_real_recon,
                             sum_noise_gen = sum_noise_gen,
                             sum_noise_gen_recon = sum_noise_gen_recon,
                             save_plot = plot_save_other,
                             show_plot = plot_show_other,
                             redshift_fig_folder = redshift_fig_folder,
                             t = t)
                    

                    
                    
                    """
                    Power Spectrum Comparisons
                    NEEDS INVERSE-TRANSFORMED DATA
                    """
                    print("\nPower Spectrum Comparisons")
#                     plot_power_spec_aggregate(real_cube = real_cube,        # should be inverse_transformed
#                                     generated_cube = noise_gen_cube,   # should be inverse_transformed
#                                     raw_cube_mean = sampled_subcubes.mean_val, 
#                                     save_plot = plot_save_other,
#                                     show_plot = plot_show_other,
#                                      redshift_fig_folder = redshift_fig_folder,
#                                      t = t,
#                                     threads=1, 
#                                     MAS="CIC", 
#                                     axis=0, 
#                                     BoxSize=75.0/2048*128)
                    plot_power_spec(real_cube = real_cube,        # should be inverse_transformed
                                    generated_cube = noise_gen_cube,   # should be inverse_transformed
                                    raw_cube_mean = sampled_subcubes.mean_val, 
                                    save_plot = plot_save_other,
                                    show_plot = plot_show_other,
                                     redshift_fig_folder = redshift_fig_folder,
                                     t = t,
                                    threads=1, 
                                    MAS="CIC", 
                                    axis=0, 
                                    BoxSize=75.0/2048*cube_size)
                    plot_power_spec2(real_cube = real_cube, 
                                     generated_cube = noise_gen_cube,  
                                    raw_cube_mean = sampled_subcubes.mean_val,
                                     redshift_fig_folder = redshift_fig_folder,
                                    s_size = 64,
                                     t = t,
                                    threads=1, 
                                    MAS="CIC", 
                                    axis=0, 
                                    BoxSize=75.0/2048.0*64)
    
    
#                     """
#                     2D Visualizations
#                     NEEDS INVERSE TRANSFORMED DATA
#                     """
#                     visualize2d(real = real_cube, 
#                                 fake = noise_gen_cube, 
#                                 raw_cube_mean = sampled_subcubes.mean_raw_val, 
#                                 redshift_fig_folder = redshift_fig_folder,
#                                 t = t,
#                                 save_plot = plot_save_other, 
#                                 show_plot = plot_show_other)
                    
                    
                    """
                    Select Random Single Cubes
                    Subset them by taking only values greater than 0
                    Even though they are inverse transformed, some values may be negative
                    due to the activation function and output function used
                    """
                    
                    real_ae_cube = real_ae_cube[real_ae_cube > 0.0]
                    noise_ae_cube = noise_ae_cube[noise_ae_cube > 0.0]
                    noise_gen_cube = noise_gen_cube[noise_gen_cube > 0.0]
                    real_cube = real_cube[real_cube > 0.0]
#                     real_ae_cube = real_ae_cube[np.nonzero(real_ae_cube)]
#                     noise_ae_cube = noise_ae_cube[np.nonzero(noise_ae_cube)]
#                     noise_gen_cube = noise_gen_cube[np.nonzero(noise_gen_cube)]
#                     real_cube = real_cube[np.nonzero(real_cube)]
#                     recon_plot = recon_plot[np.greater(recon_plot, 0)]
                    
#                     print("len(real_plot) - nonzero elements = " + str(len(real_plot)))
#                     print("len(recon_plot) - nonzero elements = " + str(len(recon_plot)))
    #                 log_nonzero_real_list.append(len(real_plot))
    #                 log_nonzero_recon_list.append(len(recon_plot))

#                     log_nonzero_recon_over_real_list.append(len(recon_plot) / len(real_plot))
                    
                    """
                    Plotting Nominal and Log Histograms
                    """
                    print("\nPlotting Nominal Histogram and PDFs")
#                     mmd_hist_plot(noise = noise_gen_cube, 
#                                   real = real_cube, 
#                                   recon_noise = noise_ae_cube, 
#                                   recon_real = real_ae_cube,
#                                   epoch = t, 
#                                   file_name = 'hist_' + str(t) + '.png', 
#                                   plot_pdf = False,
#                                   log_plot = False,
#                                   plot_show = plot_show_other,
#                                   redshift_fig_folder = redshift_fig_folder)
    
#                     mmd_hist_plot(noise = noise_gen_cube, 
#                                   real = real_cube, 
#                                   recon_noise = noise_ae_cube, 
#                                   recon_real = real_ae_cube,
#                                   epoch = t, 
#                                   file_name = 'pdf_' + str(t) + '.png', 
#                                   plot_pdf = True,
#                                   log_plot = False,
#                                   plot_show = plot_show_other,
#                                   redshift_fig_folder = redshift_fig_folder) 
                    
#                     """
#                     Plotting the log histograms & PDF
#                     """
#                     print("\nPlotting the log histograms & PDF")
#                     mmd_hist_plot(noise = noise_gen_cube, 
#                                   real = real_cube, 
#                                   recon_noise = noise_ae_cube, 
#                                   recon_real = real_ae_cube,
#                                   epoch = t, 
#                                   file_name = 'hist_log_' + str(t) + '.png', 
#                                   plot_pdf = False,
#                                   log_plot = True,
#                                   plot_show = plot_show_other,
#                                   redshift_fig_folder = redshift_fig_folder)
    
#                     mmd_hist_plot(noise = noise_gen_cube, 
#                                   real = real_cube, 
#                                   recon_noise = noise_ae_cube, 
#                                   recon_real = real_ae_cube,
#                                   epoch = t, 
#                                   file_name = 'pdf_log_' + str(t) + '.png', 
#                                   plot_pdf = True,
#                                   log_plot = True,
#                                   plot_show = plot_show_other,
#                                   redshift_fig_folder = redshift_fig_folder)                     

#                 except:
#                     pass
                
                plotted = plotted + 1
                
                
                
#             if plotted_2 < 1 and t % 5 == 0 and t > gen_iterations_limit:
            if plotted_2 < 1 and t % 1 == 0:
                # reshaping DOESNT WORK due to nonzero() -> reshaping 1D to 3D with cube_size edges
                # so just getting them again works.
                real_ae_cube = f_dec_X_D[random_batch].cpu().view(cube_size,cube_size,cube_size).detach().numpy()
                noise_ae_cube = f_dec_Y_D[random_batch].cpu().view(cube_size,cube_size,cube_size).detach().numpy()
                noise_gen_cube = y[random_batch][0].cpu().detach().numpy()
                real_cube = x[random_batch][0].cpu().detach().numpy()
                y_fixed = netG(fixed_noise)[0][0].cpu().detach().numpy()
                
#                 real_ae_cube = inverse_transform_func(cube = real_ae_cube,
#                                                   inverse_type = inverse_transform, 
#                                              sampled_dataset = sampled_subcubes)
#                 noise_ae_cube = inverse_transform_func(cube = noise_ae_cube,
#                                                   inverse_type = inverse_transform, 
#                                              sampled_dataset = sampled_subcubes)
#                 noise_gen_cube = inverse_transform_func(cube = noise_gen_cube,
#                                                   inverse_type = inverse_transform, 
#                                              sampled_dataset = sampled_subcubes)
#                 real_cube = inverse_transform_func(cube = real_cube,
#                                                   inverse_type = inverse_transform, 
#                                              sampled_dataset = sampled_subcubes)
#                 y_fixed = inverse_transform_func(cube = y_fixed,
#                                                   inverse_type = inverse_transform, 
#                                              sampled_dataset = sampled_subcubes)
                
                print("y_fixed shape = " + str(y_fixed.shape))
                print("real_ae_cube shape = " + str(real_ae_cube.shape))
                print("noise_ae_cube shape = " + str(noise_ae_cube.shape))
                print("noise_gen_cube shape = " + str(noise_gen_cube.shape))
                print("real_cube shape = " + str(real_cube.shape))
                
            
            
#                 # Plot the 3D Cubes
#                 print("\nFixed Noise Input Cube")
#                 visualize_cube(cube=y_fixed,      ## array name
#                                          edge_dim=real_ae_cube.shape[0],        ## edge dimension (128 for 128 x 128 x 128 cube)
#                                          start_cube_index_x=0,
#                                          start_cube_index_y=0,
#                                          start_cube_index_z=0,
#                                          fig_size=(10,10),
#                                          #stdev_to_white=-2,
#                                          norm_multiply=viz_multiplier,
#                                             size_magnitude = scatter_size_magnitude,
# #                                          color_map="Blues",
# #                                          plot_show = plot_show_3d,
# #                                          plot_save = plot_save_3d,
#                                          save_fig = redshift_3dfig_folder + 'fixed_noise_' + str(t) + '.png',
#                       raw_cube_max = sampled_subcubes.max_raw_val)                

#                 print("\nReconstructed, AutoEncoder Generated Real Cube")
# #                 recon_real_viz = 
#                 visualize_cube(cube=real_ae_cube,      ## array name
#                                          edge_dim=real_ae_cube.shape[0],        ## edge dimension (128 for 128 x 128 x 128 cube)
#                                          start_cube_index_x=0,
#                                          start_cube_index_y=0,
#                                          start_cube_index_z=0,
#                                          fig_size=(10,10),
#                                          #stdev_to_white=-2,
#                                          norm_multiply=viz_multiplier,
#                                             size_magnitude = scatter_size_magnitude,
# #                                          color_map="Blues",
# #                                          plot_show = plot_show_3d,
# #                                plot_save = plot_save_3d,
#                                          save_fig = redshift_3dfig_folder + 'recon_ae_real_' + str(t) + '.png',
#                       raw_cube_max = sampled_subcubes.max_raw_val)
                
#                 print("\nReconstructed, AutoEncoder Generated Noise-Input Cube")
# #                 recon_fake_viz = 
#                 visualize_cube(cube=noise_ae_cube,      ## array name
#                                          edge_dim=noise_ae_cube.shape[0],        ## edge dimension (128 for 128 x 128 x 128 cube)
#                                          start_cube_index_x=0,
#                                          start_cube_index_y=0,
#                                          start_cube_index_z=0,
#                                          fig_size=(10,10),
#                                          #stdev_to_white=-2,
#                                          norm_multiply=viz_multiplier,
#                                             size_magnitude = scatter_size_magnitude,
# #                                          color_map="Blues",
# #                                          plot_show = plot_show_3d,
# #                                plot_save = plot_save_3d,
#                                          save_fig = redshift_3dfig_folder + 'recon_ae_noisegen_' + str(t) + '.png',
#                       raw_cube_max = sampled_subcubes.max_raw_val)
                
#                 print("\nNoise-Input Generated Cube")
# #                 sample_viz = 
#                 visualize_cube(cube=noise_gen_cube,      ## array name
#                                          edge_dim=noise_gen_cube.shape[0],        ## edge dimension (128 for 128 x 128 x 128 cube)
#                                          start_cube_index_x=0,
#                                          start_cube_index_y=0,
#                                          start_cube_index_z=0,
#                                          fig_size=(10,10),
#                                          #stdev_to_white=-2,
#                                          norm_multiply=viz_multiplier,
#                                             size_magnitude = scatter_size_magnitude,
# #                                          color_map="Blues",
# #                                          plot_show = plot_show_3d,
# #                                plot_save = plot_save_3d,
#                                          save_fig = redshift_3dfig_folder + 'noisegen_' + str(t) + '.png',
#                       raw_cube_max = sampled_subcubes.max_raw_val)
                
#                 print("\nReal Cube")
# #                 real_viz = 
#                 visualize_cube(cube=real_cube,      ## array name
#                                          edge_dim=real_cube.shape[0],        ## edge dimension (128 for 128 x 128 x 128 cube)
#                                          start_cube_index_x=0,
#                                          start_cube_index_y=0,
#                                          start_cube_index_z=0,
#                                          fig_size=(10,10),
#                                          #stdev_to_white=-2,
#                                          norm_multiply=viz_multiplier,
#                                             size_magnitude = scatter_size_magnitude,
# #                                          color_map="Blues",
# #                                          plot_show = plot_show_3d,
# #                                plot_save = plot_save_3d,
#                                          save_fig = redshift_3dfig_folder +'real_' + str(t) + '.png',
#                       raw_cube_max = sampled_subcubes.max_raw_val)

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
            f_enc_X, f_dec_X, f_enc_X_size = netD(x)

            noise = torch.cuda.FloatTensor(f_enc_X_size).normal_(0, 1)
            
#             noise = torch.cuda.FloatTensor(f_enc_X_size[0], 
#                                             f_enc_X_size[1], 
#                                             f_enc_X_size[2], 
#                                             f_enc_X_size[3],
#                                             f_enc_X_size[4]).normal_(0, 1)
#             noise = Variable(noise)
            
            # output of the generator with noise input
            y = netG(noise)

            # output of the discriminator with noise input
            f_enc_Y, f_dec_Y, _ = netD(y)

            # compute biased MMD2 and use ReLU to prevent negative value
            mmd2_G = mix_rbf_mmd2(f_enc_X, 
                                  f_enc_Y, 
                                  sigma_list, 
                                  biased=True)
#             mmd2_G = poly_mmd2(f_enc_X, f_enc_Y)
#             mmd2_G = linear_mmd2(f_enc_X, f_enc_Y)
    
            mmd2_G_before_ReLU_list.append(mmd2_G)
            mmd2_G = F.relu(mmd2_G)
            mmd2_G_after_ReLU_list.append(mmd2_G)

            # compute rank hinge loss
            one_side_errG = one_sided(f_enc_X.mean(0) - f_enc_Y.mean(0))
            one_side_errG_list.append(one_side_errG)

            errG = torch.sqrt(mmd2_G) + lambda_rg * one_side_errG 
#             errG = torch.sqrt(mmd2_G) + lambda_rg * one_side_errG \
#                     + calc_gradient_penalty(x.data, y.data, lambda_gradpen)
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
                                    show_plot = plot_show_other, 
                                    save_plot = plot_save_other, 
                                    redshift_fig_folder = redshift_fig_folder,
                                  t = t,
                                  dist_ae = dist_ae)           
            
                plotted_3 = plotted_3 + 1

        run_time = (timeit.default_timer() - time_loop) / 60.0
        print("run_time = " + str(run_time))
        try:
            print('[%3d/%3d][%3d/%3d] [%5d] (%.2f m) MMD2_D %.10f hinge %.6f L2_AE_X %.6f L2_AE_Y %.6f loss_D %.6f Loss_G %.6f f_X %.6f f_Y %.6f |gD| %.4f |gG| %.4f'
                % (t, max_iter, i, len(trn_loader), gen_iterations, run_time,
                     mmd2_D.item(), one_side_errD.item(),
                     L2_AE_X_D.item(), L2_AE_Y_D.item(),
                     errD.item(), errG.item(),
                     f_enc_X_D.mean().item(), f_enc_Y_D.mean().item(),
                     grad_norm(netD), grad_norm(netG)))
        except:
            pass

        
        # plotting gradient norms for monitoring
        grad_normD = grad_norm(netD)
        grad_norm_D.append(grad_normD)
        grad_norm_G.append(grad_norm(netG))
        
        if plotted_4 < 1:
            plt.figure(figsize = (10,5))
            plt.title("grad_norms - if they are over 100 things are screwing up")
            plt.yscale('log')
            plt.plot(grad_norm_D, 
                     color = "red", 
                     label = "grad_norm_D",
                     linewidth=0.5)
            plt.plot(grad_norm_G, 
                     color = "blue", 
                     label = "grad_norm_G",
                     linewidth=0.5)
            plt.legend()
            plt.savefig(redshift_fig_folder + 'grad_norms_' + str(t) + '.png', 
                        bbox_inches='tight')
            plt.show() 
            plt.close()
            #             plt.show()
            
            plotted_4 = plotted_4 + 1


    if t % save_model_every == 0:
        print("Saving the model state_dict()")
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
                     s_train = cube_size)
print(testcd)

trial_sample = get_samples(s_sample = cube_size, 
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




