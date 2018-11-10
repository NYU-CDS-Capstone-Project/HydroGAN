#!/usr/bin/env python
# coding: utf-8

# https://github.com/OctoberChang/MMD-GAN - accompanying the paper MMD-GAN: Towards Deeper Understanding of Moment Matching Network.
# 
# To check GPU usage, open new terminal inside Jupyter and nvidia-smi

# In[ ]:


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


# In[ ]:


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


# In[ ]:


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

# In[ ]:


batch_size = 16       # BATCH_SIZE: batch size for training
gpu_device = 0        # GPU_DEVICE: gpu id (default 0)
nc = 1                # NC: number of channels in images
nz = 128                # NZ: hidden dimension in z and codespace
cube_size = 128       # for our dataset more like one edge of the subcube
lr = 5e-5               # LR: learning rate (default 5e-5)
max_iter = 150         # MAX_ITER: max iteration for training
optimizer_choice = "rmsprop"     # adam or rmsprop
dist_ae = 'L2'                  # "L2" or "L1" or "cos" -> Autoencoder reconstructruced cube loss choice
manual_seed = 1126
n_samples = batch_size * 128      # on prince, number of samples to get from the training cube
Diter_1 = 100    # default: 100
Giter_1 = 1      # default: 1
Diter_2 = 5      # default: 5
Giter_2 = 1      # default: 1
gen_iterations_limit = 25   # default = 25


# In[ ]:


assert n_samples / batch_size > 100, "The gen_iterations wont work properly!"


# ### Plotting Options

# In[ ]:


viz_multiplier = 1e0    # the norm multiplier in the 3D visualization
plot_show_3d = False


# ### Saving Options

# In[ ]:


root_dir = "./"  # this goes to 
data_dir = "../"
experiment = root_dir + "cosmo2/"       # : output directory of saved models
model_save_folder = experiment + "model/"
redshift_fig_folder = experiment + "figures/"        # folder to save mmd & related plots
redshift_3dfig_folder = experiment + "/3d_figures/"   # folder to save 3D plots
testing_folder = experiment + "testing/"   # folder to save 3D plots

save_model_every = 10               # (every x epoch) frequency to save the model


# ### Dataset Options

# In[ ]:


workers = 2          # WORKERS: number of threads to load data
redshift_info_folder = experiment + "redshift_info/"   # save some info here as pickle to speed up processing
redshift_file = "minmax_scale_01_redshift0.h5"    # redshift cube to be used
    # standardized_no_shift_redshift0.h5
    # minmax_scale_01_redshift0.h5


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


log("asdas")


# ## Parameter Documentation

# ## Training Parameters

# In[ ]:


print("Batch Size = " + str(batch_size))
print("Redshift File Used = " + str(redshift_file))
print("Number of Channels in Input = " + str(nc))
print("Hidden Dimension (codespace) = " + str(nz))
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


# In[ ]:


edge_sample = 128
edge_test = 1024

print("one edge of the test partition of the whole cube = " + str(edge_test))
print("one edge of the sampled subcubes =  " + str(edge_sample))


# ### MMD Parameters

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
lambda_rg = 16.0   # used in both err calcs

print("lambda_MMD = " + str(lambda_MMD))
print("lambda_AE_X = " + str(lambda_AE_X))
print("lambda_AE_Y = " + str(lambda_AE_Y))
print("lambda_rg = " + str(lambda_rg))


# In[ ]:


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


if not min_cube_file.exists() or not max_cube_file.exists():
    
    f = h5py.File(data_dir + redshift_file, 'r')
    f=f['delta_HI']
    
    # get the min and max
    min_cube = get_min_cube(f=f)
    print(min_cube)
    max_cube = get_max_cube(f=f)
    print(max_cube)
    
    np.save(file = redshift_info_folder + redshift_file + "_min_cube",
        arr = min_cube,
        allow_pickle = True)
    np.save(file = redshift_info_folder + redshift_file + "_max_cube",
        arr = max_cube,
        allow_pickle = True)
    


# In[ ]:


min_cube = np.load(file = redshift_info_folder + redshift_file + "_min_cube" + '.npy')
max_cube = np.load(file = redshift_info_folder + redshift_file + "_max_cube" + '.npy')
print("Min of data = " + str(min_cube))
print("Max of data = " + str(max_cube))


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


testcd = define_test(s_test = edge_test, s_train = edge_sample)
testcd


# In[ ]:


trial_sample = get_samples(s_sample = edge_sample, 
                            nsamples = 1, 
#                             h5_filename = redshift_file, 
                            test_coords = testcd,
                            f = f)
trial_sample


# In[ ]:


trial_sample[0].shape


# In[ ]:


trial_sample[0].reshape(-1,).shape


# In[ ]:


trial_plot = trial_sample[0].reshape(-1,)


# In[ ]:


# [-1,1] -> [0,1] for plotting
# trial_plot = (trial_plot + 1.0) / 2.0


# In[ ]:


trial_plot.min()


# In[ ]:


trial_plot.max()


# In[ ]:


trial_plot.sum()


# In[ ]:


plt.figure(figsize = (16,8))
plt.title("Trial Sample")
plt.xlim((trial_plot.min(),
         trial_plot.max()))
bins = np.linspace(trial_plot.min(),
                   trial_plot.max(), 
                   100)
plt.hist(trial_plot, bins = bins, 
         color = "blue" ,
         alpha = 0.3, 
         label = "Trial")
plt.legend()
plt.show()


# In[ ]:


# from utils.plot_utils import visualize_cube, truncate_colormap


trial_visual = trial_sample[0]
print(trial_visual.shape)
trial_visual_edge = trial_visual.shape[0]
print("edge dim = " + str(trial_visual_edge))

# from [-1,1] to [0,1]
trial_visual = (trial_visual + 1.0) / 2.0
print(trial_visual.shape)

visualize_cube(cube=trial_visual,      ## array name
             edge_dim=trial_visual_edge,        ## edge dimension (128 for 128 x 128 x 128 cube)
             start_cube_index_x=0,
             start_cube_index_y=0,
             start_cube_index_z=0,
             fig_size=(10,10),
             stdev_to_white=-2,
             norm_multiply=viz_multiplier,
             color_map="Blues",
             plot_show = True,
             save_fig = False)


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
                                  max_cube = max_cube)


# In[ ]:


# Get data
trn_loader = torch.utils.data.DataLoader(sampled_subcubes, 
                                         batch_size = batch_size,
                                         shuffle=True, 
                                         num_workers=int(workers))


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
                p.data.clamp_(-0.01, 0.01)

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
#             print("y shape = " + str(y.shape))
#             print("y[0] shape = " + str(y[0].shape))
#             print("y[0][0] shape = " + str(y[0][0].shape))
#             sample_cube_viz = y[0][0].cpu().detach().numpy()
#             print("sample_cube_viz shape = " + str(sample_cube_viz.shape))
        
            # output of the discriminator with noise input
            # this tests discriminator 
            f_enc_Y_D, f_dec_Y_D = netD(y)
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
            if j % 5 == 0 and plotted < 1:
                try:

                    titles_to_plot = ["mmd2_D_before_ReLU_list","mmd2_D_after_ReLU_list","one_side_errD_list","L2_AE_X_D_list","L2_AE_Y_D_list","errD_list - D loss goes to 0: failure mode"] 
                    data_to_plot = [mmd2_D_before_ReLU_list,mmd2_D_after_ReLU_list,one_side_errD_list,L2_AE_X_D_list,L2_AE_Y_D_list,errD_list]

                    for p in range(len(data_to_plot)):
                        mmd_D_loss(p, titles_to_plot[p-1], data_to_plot[p-1], redshift_fig_folder)

                    #                     plt.show() 

                    # plot output of the discriminator with real data input
                    # and output of the discriminator with noise input
                    # on the same histogram 
                    random_batch = random.randint(0,batch_size-1)
                    recon_plot = y[random_batch].cpu().view(-1,1).detach().numpy()
                    real_plot = x[random_batch].cpu().view(-1,1).detach().numpy()
                    print("min(x[random_batch]) = " + str(min(real_plot)))
                    print("max(x[random_batch]) = " + str(max(real_plot)))
                    print("min(y[random_batch]) = " + str(min(recon_plot)))
                    print("max(y[random_batch]) = " + str(max(recon_plot)))
                    
                    recon_plot = recon_plot[np.nonzero(recon_plot)]
    #                 recon_plot = recon_plot[np.greater(recon_plot, 0)]
                    real_plot = real_plot[np.nonzero(real_plot)]
    #                 print("max(x[0] - nonzero) = " + str(max(real_plot)))
    #                 print("max(y[0] - nonzero) = " + str(max(recon_plot)))
    #                 print("min(x[0] - nonzero) = " + str(min(real_plot)))
    #                 print("min(y[0] - nonzero) = " + str(min(recon_plot)))
    #                 recon_plot = recon_plot + 1
    #                 real_plot = real_plot + 1


                    print("len(real_plot) - nonzero elements = " + str(len(real_plot)))
                    print("len(recon_plot) - nonzero elements = " + str(len(recon_plot)))
    #                 log_nonzero_real_list.append(len(real_plot))
    #                 log_nonzero_recon_list.append(len(recon_plot))

                    log_nonzero_recon_over_real_list.append(len(recon_plot) / len(real_plot))
                    print("max(real_plot) = " + str(max(real_plot)))
                    print("max(recon_plot) = " + str(max(recon_plot)))
                    print("min(real_plot) = " + str(min(real_plot)))
                    print("min(recon_plot) = " + str(min(recon_plot)))



                    #histogram
                    2_hist_plot(recon_plot, real_plot, t,'hist_', 0)

                    #pdf
                    2_hist_plot(recon_plot, real_plot, t,'log_', 1)

                    
                    
                    
                    #                     plt.show()

#                     plt.figure(figsize = (16,8))
#                     plt.title("Nonzero in Reconstructed Subcubes / Nonzero in Real Subcubes")
#                     plt.ylim(-0.0001, 10)
#                     plt.plot(log_nonzero_recon_over_real_list, 
#                              color = "blue", 
#                              label = "Nonzero in Reconstructed Subcubes / Nonzero in Real Subcubes")
#                     plt.show()  
                    
                    """
                    Plotting the log histograms & PDF
                    """
                    #normalize between 0-1 before plotting
                    recon_plot_ = cube_scaler_for_plotting(sampled_subcubes,recon_plot)
                    real_plot_ = cube_scaler_for_plotting(sampled_subcubes,real_plot)

                    recon_plot_ = np.log(recon_plot_)
                    real_plot_ = np.log(real_plot_)


                    #histogram
                    2_hist_plot(recon_plot_, real_plot_, t, 'hist_log_', 0)

                    #pdf
                    2_hist_plot(recon_plot_, real_plot_, t,  'pdf_log_', 1)

                    #                     plt.show()

                except:
                    pass
                
                plotted = plotted + 1
                
                
                
            if plotted_2 < 1:
                
                # selecting a random cube from the batch
                random_batch = random.randint(0,batch_size-1)
                real_ae_cube = f_dec_X_D[random_batch].cpu().view(128,128,128).detach().numpy()
                noise_ae_cube = f_dec_Y_D[random_batch].cpu().view(128,128,128).detach().numpy()
                noise_gen_cube = y[random_batch][0].cpu().detach().numpy()
                real_cube = x[random_batch][0].cpu().detach().numpy()
                
                # transforming the inputs for visualization
                # to [0,1] interval for better plotting
                # using the whole cube's min and max!!!
                # not the subcube's!!!
                real_ae_cube = (real_ae_cube - sampled_subcubes.min_val) / sampled_subcubes.max_val
                noise_ae_cube = (noise_ae_cube - sampled_subcubes.min_val) / sampled_subcubes.max_val
                noise_gen_cube = (noise_gen_cube - sampled_subcubes.min_val) / sampled_subcubes.max_val
                real_cube = (real_cube - sampled_subcubes.min_val) / sampled_subcubes.max_val
                
                print("real_ae_cube max = " + str(real_ae_cube.max()) + ", min = " + str(real_ae_cube.min()))
                print("noise_ae_cube max = " + str(noise_ae_cube.max()) + ", min = " + str(noise_ae_cube.min()))
                print("noise_gen_cube max = " + str(noise_gen_cube.max()) + ", min = " + str(noise_gen_cube.min()))
                print("real_cube max = " + str(real_cube.max()) + ", min = " + str(real_cube.min()))

                print("\nReconstructed, AutoEncoder Generated Real Cube")
#                 recon_real_viz = 
                visualize_cube(cube=real_ae_cube,      ## array name
                                         edge_dim=real_ae_cube.shape[0],        ## edge dimension (128 for 128 x 128 x 128 cube)
                                         start_cube_index_x=0,
                                         start_cube_index_y=0,
                                         start_cube_index_z=0,
                                         fig_size=(10,10),
                                         stdev_to_white=-2,
                                         norm_multiply=viz_multiplier,
                                         color_map="Blues",
                                         plot_show = plot_show_3d,
                                         save_fig = redshift_3dfig_folder + 'recon_ae_real_' + str(t) + '.png')
                
                print("\nReconstructed, AutoEncoder Generated Noise-Input Cube")
#                 recon_fake_viz = 
                visualize_cube(cube=noise_ae_cube,      ## array name
                                         edge_dim=noise_ae_cube.shape[0],        ## edge dimension (128 for 128 x 128 x 128 cube)
                                         start_cube_index_x=0,
                                         start_cube_index_y=0,
                                         start_cube_index_z=0,
                                         fig_size=(10,10),
                                         stdev_to_white=-2,
                                         norm_multiply=viz_multiplier,
                                         color_map="Blues",
                                         plot_show = plot_show_3d,
                                         save_fig = redshift_3dfig_folder + 'recon_ae_noisegen_' + str(t) + '.png')
                
                print("\nNoise-Input Generated Cube")
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
                                         plot_show = plot_show_3d,
                                         save_fig = redshift_3dfig_folder + 'noisegen_' + str(t) + '.png')
                
                print("\nReal Cube")
#                 real_viz = 
                visualize_cube(cube=real_cube,      ## array name
                                         edge_dim=real_cube.shape[0],        ## edge dimension (128 for 128 x 128 x 128 cube)
                                         start_cube_index_x=0,
                                         start_cube_index_y=0,
                                         start_cube_index_z=0,
                                         fig_size=(10,10),
                                         stdev_to_white=-2,
                                         norm_multiply=viz_multiplier,
                                         color_map="Blues",
                                         plot_show = plot_show_3d,
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
                # plotting Generator plots
                plt.figure(figsize = (10,5))
                plt.title("mmd2_G_before_ReLU_list")
                plt.plot(mmd2_G_before_ReLU_list)
                plt.savefig(redshift_fig_folder + 'mmd2_G_before_ReLU_list_' + str(t) + '.png', 
                            bbox_inches='tight')
                plt.show() 
                plt.close()
                    #                 plt.show() 
                plt.figure(figsize = (10,5))
                plt.title("mmd2_G_after_ReLU_list")
                plt.plot(mmd2_G_after_ReLU_list)
                plt.savefig(redshift_fig_folder + 'mmd2_G_after_ReLU_list_' + str(t) + '.png', 
                            bbox_inches='tight')
                plt.show() 
                plt.close()
                #                 plt.show() 
                plt.figure(figsize = (10,5))
                plt.title("one_side_errG_list")
                plt.plot(one_side_errG_list)
                plt.savefig(redshift_fig_folder + 'one_side_errG_list_' + str(t) + '.png', 
                            bbox_inches='tight')
                plt.show() 
                plt.close()
                #                 plt.show() 
                plt.figure(figsize = (10,5))
                plt.title("errG_list")
                plt.plot(errG_list)
                plt.savefig(redshift_fig_folder + 'errG_list_' + str(t) + '.png', 
                            bbox_inches='tight')
                plt.show() 
                plt.close()
                #                 plt.show()            
            
                plotted_3 = plotted_3 + 1

        run_time = (timeit.default_timer() - time_loop) / 60.0
        print("run_time = " + str(run_time))
        try:
            print('[%3d/%3d][%3d/%3d] [%5d] (%.2f m) MMD2_D %.6f hinge %.6f L2_AE_X %.6f L2_AE_Y %.6f loss_D %.6f Loss_G %.6f f_X %.6f f_Y %.6f |gD| %.4f |gG| %.4f'
    #                   % (t, max_iter, i, len(trn_loader), gen_iterations, run_time,
    #                      mmd2_D.data[0], one_side_errD.data[0],
    #                      L2_AE_X_D.data[0], L2_AE_Y_D.data[0],
    #                      errD.data[0], errG.data[0],
    #                      f_enc_X_D.mean().data[0], f_enc_Y_D.mean().data[0],
    #                      grad_norm(netD), grad_norm(netG)))
                % (t, max_iter, i, len(trn_loader), gen_iterations, run_time,
                     mmd2_D.item(), one_side_errD.item(),
                     L2_AE_X_D.item(), L2_AE_Y_D.item(),
                     errD.item(), errG.item(),
                     f_enc_X_D.mean().item(), f_enc_Y_D.mean().item(),
                     grad_norm(netD), grad_norm(netG)))
        except:
            pass

        
#         if gen_iterations % 500 == 0:
#             y_fixed = netG(fixed_noise)
#             y_fixed.data = y_fixed.data.mul(0.5).add(0.5)
#             f_dec_X_D = f_dec_X_D.view(f_dec_X_D.size(0), args.nc, args.image_size, args.image_size)
#             f_dec_X_D.data = f_dec_X_D.data.mul(0.5).add(0.5)
#             vutils.save_image(y_fixed.data, '{0}/fake_samples_{1}.png'.format(args.experiment, gen_iterations))
#             vutils.save_image(f_dec_X_D.data, '{0}/decode_samples_{1}.png'.format(args.experiment, gen_iterations))

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
    assert(in_testing, "Stopping here, because not in testing...")


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




