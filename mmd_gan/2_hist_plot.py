def 2_hist_plot(recon, real, epoch, file_name_ hd ):
	"""
    Args:
        recon(): generated data
        real(): real data
        epoch(integer): epoch number
        file_name(string): name of the file
        hd (integer) : if 0 it's a histogram, if 1 it's a pdf
        

    """
    plt.figure(figsize = (16,8))
    plt.title("Histograms of Hydrogen")
    plt.xlim(min(recon.min(),real.min()),
            max(recon.max(),real.max()))
    bins = np.linspace(min(recon.min(),real.min()),
                       max(recon.max(),real.max()), 
                       100)
    plt.hist(recon, bins = bins, 
             color = "red" ,
             alpha= 0.5, 
             label = "Generator(Noise) Subcube - Only Nonzero")

    if(hd==0):
        plt.hist(real, bins = bins, 
                 color = "blue" ,
                 alpha = 0.3, 
                 label = "Real Sample Subcube - Only Nonzero")
    else:
        plt.hist(real, bins = bins, 
                 color = "blue" ,
                 alpha = 0.3, 
                 label = "Real Sample Subcube - Only Nonzero",
                 density = True)

    plt.legend()
    plt.savefig(redshift_fig_folder + file_name + str(epoch) + '.png', 
                bbox_inches='tight')
    plt.show() 
    plt.close()

    return