# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 12:32:43 2018

@author: Juan Jose Zamudio
"""
import random
import numpy as np
#import h5py
random.seed(a=1)

def define_test(s_test, s_train):
    #2048/16=128
    m=2048/s_train
    x=random.randint(0,m)*s_train
    y=random.randint(0,m)*s_train
    z=random.randint(0,m)*s_train
    #print(x,y,z)
    return {'x':[x,x+s_test], 'y':[y,y+s_test], 'z':[z,z+s_test]}

def check_coords(test_coords, train_coords):
    valid=True
    for i in ['x','y','z']:
        r=(max(test_coords[i][0], 
               train_coords[i][0]), 
           min(test_coords[i][1],
               train_coords[i][1]))
        if r[0]<=r[1]:
            valid=False
    return valid

def get_samples(file, s_sample, nsamples, test_coords):
    #n is size of minibatch, get valid samples (not intersecting with test_coords)
    sample_list=[]
    m=2048-s_sample
    for n in range(nsamples):
        #print("Sample No = " + str(n + 1) + " / " + str(nsamples))
        sample_valid=False
        while sample_valid==False:
            x = random.randint(0,m)
            y = random.randint(0,m)
            z = random.randint(0,m)
            sample_coords = {'x':[x,x+s_sample], 
                             'y':[y,y+s_sample], 
                             'z':[z,z+s_sample]}
            
            sample_valid = check_coords(test_coords, sample_coords)
        
        sample_list.append(sample_coords)
    
    #Load cube and get samples and convert them to np.arrays
    sample_array=[]
    #f file has to be opened outisde the function
    for c in sample_list:
        a = file[c['x'][0]:c['x'][1],
              c['y'][0]:c['y'][1],
              c['z'][0]:c['z'][1]]
        
        sample_array.append( np.array(a))
    
    return np.array(sample_array)


def get_max_cube(file):
    
    max_list = []
    for i in range(file.shape[0]):
        #print(np.max(f[i:i+1,:,:]))
        max_list.append(np.max(file[i:i+1,:,:]))
    max_cube = max(max_list)
   
    return max_cube

def get_min_cube(file):
    min_list = [] 
    for i in range(file.shape[0]):
        #print(np.max(f[i:i+1,:,:]))
        min_list.append(np.min(file[i:i+1,:,:]))
    min_cube = min(min_list)
    return min_cube

def get_mean_cube(file):
    mean_list = []
    for i in range(file.shape[0]):
        #print(np.max(f[i:i+1,:,:]))
        mean_list.append(np.mean(file[i:i+1,:,:]))
    mean_cube = np.mean(mean_list)
    return mean_cube

def get_stddev_cube(file, mean_cube):
    variance_list = []
    for i in range(file.shape[0]):
        variance_list.append(np.mean(np.square(file[i:i+1,:,:] - mean_cube)))
    stddev_cube = np.sqrt(np.mean(variance_list))
    return stddev_cube