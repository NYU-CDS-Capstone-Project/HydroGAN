from dataset import *
import matplotlib.pyplot as plt
from utils.plot_utils import visualize_cube
from utils.data_utils import inverse_transform_func
import timeit

def test_3d_plot(edge_test, edge_sample, f, scatter_size_magnitude,
                viz_multiplier, plot_save_3d, inverse_transform, sampled_subcubes):

    testcd = define_test(s_test = edge_test, s_train = edge_sample)
    testcd

    trial_sample = get_samples(s_sample = edge_sample, 
                                nsamples = 10, 
    #                             h5_filename = redshift_file, 
                                test_coords = testcd,
                                f = f)
    trial_sample

    trial_sample[0].shape

    trial_sample[0].reshape(-1,).shape

    trial_plot = trial_sample[0].reshape(-1,)

    trial_plot.min()
    trial_plot.max()
    trial_plot.sum()

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

    # from utils.plot_utils import visualize_cube, truncate_colormap

    for i in range(len(trial_sample)):
        trial_visual = trial_sample[i]
        trial_visual = inverse_transform_func(cube = trial_visual,
                                              inverse_type = inverse_transform, 
                                         sampled_dataset = sampled_subcubes)
        print("trial_visual.shape = " + str(trial_visual.shape))
#         print(trial_visual)
        trial_visual_edge = trial_visual.shape[0]
    #     print("edge dim = " + str(trial_visual_edge))


        visualize_cube(cube=trial_visual,      ## array name
                     edge_dim=trial_visual_edge,        ## edge dimension (128 for 128 x 128 x 128 cube)
                     start_cube_index_x=0,
                     start_cube_index_y=0,
                     start_cube_index_z=0,
                     fig_size=(10,10),
                     size_magnitude = scatter_size_magnitude,
                     norm_multiply=viz_multiplier,
                     color_map="Blues",
                     plot_show = True,
                       plot_save = plot_save_3d,
                     save_fig = False)
        
        
