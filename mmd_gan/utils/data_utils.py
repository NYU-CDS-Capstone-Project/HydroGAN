import numpy as np

def get_max_cube(f):
    max_list = [np.max(f[i:i+1,:,:]) for i in range(f.shape[0])]
    max_cube = max(max_list)
    return max_cube

def get_min_cube(f):
    min_list = [np.min(f[i:i+1,:,:]) for i in range(f.shape[0])]
    min_cube = min(min_list)
    return min_cube


def cube_scaler_for_plotting(subcubes_datasets,subcube):
	"""
    Args:
       	sampled_subcubes (HydrogenDataset): main subcube that we sample from
        subcube (?): randomly selected cube - numpy array
    """
	subcube_ = (subcube - subcubes_datasets.min_val)/(sampled_subcubes.max_val - sampled_subcubes.min_val) 

	return subcube_



