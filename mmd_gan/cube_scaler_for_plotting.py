def cube_scaler_for_plotting(subcubes_datasets,subcube):
	"""
    Args:
       	sampled_subcubes (HydrogenDataset): main subcube that we sample from
        subcube (?): randomly selected cube - numpy array
    """
	subcube_ = (subcube - subcubes_datasets.min_val)/(sampled_subcubes.max_val - sampled_subcubes.min_val) 

	return subcube_