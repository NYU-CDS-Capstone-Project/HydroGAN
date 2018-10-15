import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
import h5py
import matplotlib as mpl
%matplotlib inline
import itertools

f = h5py.File("sample16.h5", mode="r")
f = np.array([*f["sample16"]])

cube_size = 128
random_cube_index = 0
stdev_mask = 1
edge = np.array([*range(cube_size)])

fig = plt.figure(figsize=(20,20)) 
ax = fig.add_subplot(111, projection='3d') 

start = random_cube_index
end = random_cube_index+cube_size
    
data_value = f[0][start:end,
                  start:end,
                  start:end]

x,y,z = edge,edge,edge

product = [*itertools.product(x,y,z)]

X = np.array([product[k][0] for k in [*range(len(product))]])
Y = np.array([product[k][1] for k in [*range(len(product))]])
Z = np.array([product[k][2] for k in [*range(len(product))]])

data_1dim = np.array([data_value[X[i]][Y[i]][Z[i]] for i in [*range(len(product))]])
initial_mean = np.mean(data_1dim) - np.std(data_1dim)
mask = data_1dim > 0
mask = mask.astype(np.int)

data_1dim = np.multiply(mask,data_1dim)
X = X[np.where(data_1dim>0)]
Y = Y[np.where(data_1dim>0)]
Z = Z[np.where(data_1dim>0)]
data_1dim = data_1dim[np.where(data_1dim>0)]
s = 500*data_1dim/np.linalg.norm(data_1dim)
print (s[:5])
array_min = np.min(data_1dim)
array_max = np.max(data_1dim)
ax.scatter(X, Y, Z, 
           c=data_1dim, 
           norm = mpl.colors.Normalize(vmin=array_min,vmax=array_max),
           cmap=plt.get_cmap("Blues"),s=s,alpha=1)
# ax.scatter(X, Y, Z, c=data_1dim,s=50,alpha=0.5)
plt.show()