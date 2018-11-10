def mmd_D_loss_plot(fig_id, fig_title, data, save_direct ):
	"""
    Args:
        fig_id(int): figure number
        fig_title(string): title of the figure
        data(): data to plot
        save_direct(string): directory to save

    """
    plt.figure(fig_id, figsize = (10,5))
    plt.title(fig_title)
    plt.plot(data)
    plt.savefig(dave_direct + fig_title +'_' + str(t) + '.png', 
                bbox_inches='tight')
    plt.close()