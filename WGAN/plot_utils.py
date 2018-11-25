# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 12:36:54 2018

@author: Juan Jose Zamudio
"""

import matplotlib.pyplot as plt
import numpy as np

def hist_plot(noise, real, log_plot, redshift_fig_folder) :
    """
    Args:
        real(): real data
        epoch(integer): epoch number
        file_name(string): name of the file
        hd (integer) : if 0 it's a histogram, if 1 it's a pdf
        
    """
    
    plt.figure(figsize = (20,10))
    plot_min = min(float(noise.min()), float(real.min()))
    plot_max = max(float(noise.max()), float(real.max()))
    plt.xlim(plot_min,plot_max)
    
    bins = np.linspace(plot_min,plot_max,400)
    
    real_label = "Real Subcube - Only Nonzero"
    noise_label = "Noise Subcube - Only Nonzero"
    
    for m in range(noise.size()[0]):
        plt.hist(real[m][0].flatten(), bins = bins, color = "b" , log = log_plot, alpha = 0.3, label = real_label, normed=True)
        plt.hist(noise[m][0].flatten(), bins = bins, color = "r" , log = log_plot, alpha= 0.3, label = noise_label, normed=True)

    #plt.legend()
    #plt.savefig(redshift_fig_folder + file_name, bbox_inches='tight')   
    plt.title('Blue: Real/ Red: Generated', fontsize=16)
    plt.show()


def plot_loss(datalist, ylabel, log_):
    plt.figure(figsize=(20,10))
    
    if ylabel=='Wasserstein loss':
        plt.plot([-x for x in datalist], linewidth=2.5, color='b')
        
    else:
        plt.plot([x for x in datalist], linewidth=2.5, color='b')
        
    plt.ylabel(ylabel, fontsize=16)
    plt.yticks(fontsize=14)
    
    if ylabel=='Generator loss':
        plt.xlabel('Epoch', fontsize=16)
    else:
        plt.xlabel('Iterations', fontsize=16)
    plt.show()

def plot_means(real_list, fake_list):
    plt.figure(figsize=(20,10))
    plt.title('Histogram of means', fontsize=18)
    
    bins = np.linspace(0, max(max(real_list), max(fake_list)), 32)
    
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.hist(fake_list, bins, alpha=0.5, label='Generated', color='red', normed=1);
    plt.hist(real_list,bins, alpha=0.5, label='Real', color='blue', normed=1);
    plt.legend(fontsize=16)
    plt.show() 