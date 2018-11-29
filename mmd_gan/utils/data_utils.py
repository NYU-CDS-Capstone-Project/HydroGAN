import numpy as np
import timeit
from pathlib import Path
import h5py

def get_max_cube(f):
    max_list = [np.max(f[i:i+1,:,:]) for i in range(f.shape[0])]
    max_cube = max(max_list)
    return max_cube

def get_min_cube(f):
    min_list = [np.min(f[i:i+1,:,:]) for i in range(f.shape[0])]
    min_cube = min(min_list)
    return min_cube

def get_mean_cube(f):
    mean_list = [np.mean(f[i:i+1,:,:]) for i in range(f.shape[0])]
    mean_cube = np.mean(mean_list)
    return mean_cube

def get_stddev_cube(f, mean_cube):
    variance_list = [np.mean(np.square(f[i:i+1,:,:] - mean_cube))\
                     for i in range(f.shape[0])]
    stddev_cube = np.sqrt(np.mean(variance_list))
    return stddev_cube

def get_or_calc_stat(stat,
                     redshift_info_folder,
                     redshift_file,
                     data_dir,
                     ):
    """
    stat = either one of ["min","max","mean","stddev"]
    """
    
    if stat == "min":
        file_end = "_min_cube"
    elif stat == "max":
        file_end = "_max_cube"
    elif stat == "mean":
        file_end = "_mean_cube"  
    elif stat == "stddev":
        file_end = "_stddev_cube"
    else:
        raise Exception("Stat wanted not implemented yet!")
        
    stat_cube_file = Path(redshift_info_folder + redshift_file + file_end + ".npy")
    
    
    if not stat_cube_file.exists():
        # if not calculated, calculate, save and return
        f = h5py.File(data_dir + redshift_file, 'r')
        f=f['delta_HI']
        
        if stat == "min":
            stat_cube = get_min_cube(f=f)
        elif stat == "max":
            stat_cube = get_max_cube(f=f)
        elif stat == "mean":
            stat_cube = get_mean_cube(f=f)  
        elif stat == "stddev":
            # this should work because stddev is later than mean in the stat argument 
            mean_cube = np.load(file = Path(redshift_info_folder + redshift_file + "_mean_cube" + ".npy"))
            stat_cube = get_stddev_cube(f=f, mean_cube = mean_cube)
        
#         print(str(stat) + " = " + str(stat_cube))
        
        np.save(file = redshift_info_folder + redshift_file + file_end,
                arr = stat_cube,
                allow_pickle = True)
        return stat_cube
        
    else:
        # if already calculated, just print and return
#         print(str(stat) + " = " + str(np.load(file = stat_cube_file)))
        return np.load(file = stat_cube_file)
        
        

def get_stats_cube(redshift_info_folder,
                   redshift_file,
                   data_dir):
    
    stats_name_list = ["min","max","mean","stddev"] 
    stats_cube_list = []
    
    for stats in stats_name_list:
        stats_cube = get_or_calc_stat(stat = stats,
                                     redshift_info_folder = redshift_info_folder,
                                     redshift_file = redshift_file,
                                     data_dir = data_dir)
        stats_cube_list.append(stats_cube)
        
    return stats_cube_list[0],stats_cube_list[1],stats_cube_list[2],stats_cube_list[3]
        
    
#         # check if redshift info (min & max exists) as pickle
#     # if not saved, find the max and min and save them for later use
#     min_cube_file = Path(redshift_info_folder + redshift_file + "_min_cube" + ".npy")
#     max_cube_file = Path(redshift_info_folder + redshift_file + "_max_cube" + ".npy")
#     mean_cube_file = Path(redshift_info_folder + redshift_file + "_mean_cube" + ".npy")
#     stddev_cube_file = Path(redshift_info_folder + redshift_file + "_stddev_cube" + ".npy")


#     if not min_cube_file.exists() or not max_cube_file.exists() or not mean_cube_file.exists() or not stddev_cube_file.exists():

#         f = h5py.File(data_dir + redshift_file, 'r')
#         f=f['delta_HI']

#         # get the min and max
#         min_cube = get_min_cube(f=f)
#         print(min_cube)
#         max_cube = get_max_cube(f=f)
#         print(max_cube)
#         mean_cube = get_mean_cube(f=f)
#         print(mean_cube)
#         stddev_cube = get_stddev_cube(f=f, mean_cube=mean_cube)
#         print(stddev_cube)

#         np.save(file = redshift_info_folder + redshift_file + "_min_cube",
#             arr = min_cube,
#             allow_pickle = True)
#         np.save(file = redshift_info_folder + redshift_file + "_max_cube",
#             arr = max_cube,
#             allow_pickle = True)
#         np.save(file = redshift_info_folder + redshift_file + "_mean_cube",
#             arr = mean_cube,
#             allow_pickle = True)
#         np.save(file = redshift_info_folder + redshift_file + "_stddev_cube",
#             arr = stddev_cube,
#             allow_pickle = True)
    


def minmax_scale(cube_tensor, 
                 inverse,
                 min_cube, # input raw min when inverse
                 max_cube, # input raw max when inverse
                 redshift, 
                 save_or_return = True):
    """
    save_or_return = True for save, False for return
    """
    whole_new_f = np.empty(shape = (cube_tensor.shape[0],
                                    cube_tensor.shape[1],
                                    cube_tensor.shape[2]),
                       dtype = np.float64)
#     print(whole_new_f.shape)
    
    if inverse == False:
        for i in range(cube_tensor.shape[0]):
            print(str(i + 1) + " / " + str(cube_tensor.shape[0])) if i % 250 == 0 else False
            whole_new_f[i:i+1,:,:] = (cube_tensor[i:i+1,:,:] - min_cube)/(max_cube-min_cube)
    
#         print("New mean = " + str(np.mean(whole_new_f)))
#         print("New median = " + str(np.median(whole_new_f)))
        assert np.amin(whole_new_f) == 0.0, "minimum not = 0!"
        assert np.amax(whole_new_f) == 1.0, "maximum not = 1!"
        
    elif inverse == True:
#         for i in range(cube_tensor.shape[0]):
#             print(str(i + 1) + " / " + str(cube_tensor.shape[0])) if i % 250 == 0 else False
#             whole_new_f[i:i+1,:,:] = cube_tensor[i:i+1,:,:] * (max_cube - min_cube) + min_cube
        whole_new_f = cube_tensor * (max_cube - min_cube) + min_cube
        
    else:
        raise Exception('Please specify whether you want normal or inverse scaling!')
    
    
    if save_or_return and inverse == False:
        hf = h5py.File('minmax_scale_01_redshift'+redshift+'.h5', 'w')
        hf.create_dataset('delta_HI', data=whole_new_f)
        hf.close()
    elif save_or_return and inverse == True:
        raise Exception('Why do you want to save the inverse transformed?\nIts the normal one!')
    else:
        return whole_new_f
    
    
    
def minmax_scale_neg11(cube_tensor, 
                 inverse,
                 min_cube, # input raw min when inverse
                 max_cube, # input raw max when inverse
                 redshift, 
                 save_or_return = True):
    """
    save_or_return = True for save, False for return
    """
    whole_new_f = np.empty(shape = (cube_tensor.shape[0],
                                    cube_tensor.shape[1],
                                    cube_tensor.shape[2]),
                       dtype = np.float64)
    print(whole_new_f.shape)
    
    if inverse == False:
        for i in range(cube_tensor.shape[0]):
            print(str(i + 1) + " / " + str(cube_tensor.shape[0])) if i % 250 == 0 else False
            whole_new_f[i:i+1,:,:] = 2* (cube_tensor[i,:,:] - min_cube)/(max_cube-min_cube) - 1
    
#         print("New mean = " + str(np.mean(whole_new_f)))
#         print("New median = " + str(np.median(whole_new_f)))
        assert np.amin(whole_new_f) == 0.0, "minimum not = 0!"
        assert np.amax(whole_new_f) == 1.0, "maximum not = 1!"
        
    elif inverse == True:
#         for i in range(cube_tensor.shape[0]):
#             print(str(i + 1) + " / " + str(cube_tensor.shape[0])) if i % 250 == 0 else False
#             whole_new_f[i:i+1,:,:] = 0.5*(1 + cube_tensor[i:i+1,:,:])*(max_cube - min_cube) + min_cube
        whole_new_f = 0.5*(1 + cube_tensor)*(max_cube - min_cube) + min_cube
        
    else:
        raise Exception('Please specify whether you want normal or inverse scaling!')
    
    
    if save_or_return and inverse == False:
        hf = h5py.File('minmax_scale_neg11_redshift'+redshift+'.h5', 'w')
        hf.create_dataset('delta_HI', data=whole_new_f)
        hf.close()
    elif save_or_return and inverse == True:
        raise Exception('Why do you want to save the inverse transformed?\nIts the normal one!')
    else:
        return whole_new_f
    
    

    
def standardize(cube_tensor, 
                 inverse,
                 mean_cube, # input raw mean when inverse
                 stddev_cube, # input raw stddev when inverse
                 shift,
                 redshift, 
                 save_or_return):
    """
    save_or_return = True for save, False for return
    """
    whole_new_f = np.empty(shape = (cube_tensor.shape[0],
                                    cube_tensor.shape[1],
                                    cube_tensor.shape[2]),
                       dtype = np.float64)
    print(whole_new_f.shape)
    
    if inverse == False:
        for i in range(cube_tensor.shape[0]):
            print(str(i + 1) + " / " + str(cube_tensor.shape[0])) if i % 250 == 0 else False
            if shift:
                whole_new_f[i:i+1,:,:] = (cube_tensor[i:i+1,:,:] - mean_cube)/ stddev_cube
            else:
                whole_new_f[i:i+1,:,:] = cube_tensor[i:i+1,:,:]/ stddev_cube
            
#         print("New mean = " + str(np.mean(whole_new_f)))
#         print("New median = " + str(np.median(whole_new_f)))
#         print("New min = " + str(np.amin(whole_new_f)))
#         print("New max = " + str(np.amax(whole_new_f)))
        
    elif inverse == True:
#         for i in range(cube_tensor.shape[0]):
#             print(str(i + 1) + " / " + str(cube_tensor.shape[0])) if i % 250 == 0 else False
#             if shift:
#                 whole_new_f[i:i+1,:,:] = cube_tensor[i:i+1,:,:]*stddev_cube + mean_cube
#             else:
#                 whole_new_f[i:i+1,:,:] = cube_tensor[i:i+1,:,:]*stddev_cube
        if shift:
            whole_new_f = cube_tensor*stddev_cube + mean_cube
        else:
            whole_new_f = cube_tensor*stddev_cube

    else:
        raise Exception('Please specify whether you want normal or inverse scaling!')
    
    
    if save_or_return and inverse == False:
        if shift:
            hf = h5py.File('standardize_noshift_redshift'+redshift+'.h5', 'w')
        else:
            hf = h5py.File('standardize_shift_redshift'+redshift+'.h5', 'w')
        hf.create_dataset('delta_HI', data=whole_new_f)
        hf.close()
    elif save_or_return and inverse == True:
        raise Exception('Why do you want to save the inverse transformed?\nIts the normal one!')
    else:
        return whole_new_f
    
def root_transform(cube_tensor, 
                 inverse,
                 root,  # tthe fraction corresponding to the root
                 redshift, 
                 save_or_return):
    """
    save_or_return = True for save, False for return
    """
    whole_new_f = np.empty(shape = (cube_tensor.shape[0],
                                    cube_tensor.shape[1],
                                    cube_tensor.shape[2]),
                       dtype = np.float64)
    print(whole_new_f.shape)
    
    if inverse == False:
        for i in range(cube_tensor.shape[0]):
            print(str(i + 1) + " / " + str(cube_tensor.shape[0])) if i % 250 == 0 else False
            whole_new_f[i:i+1,:,:] = np.power(cube_tensor[i:i+1,:,:],1.0/root)
        
    elif inverse == True:
        whole_new_f = np.power(cube_tensor,root)

    else:
        raise Exception('Please specify whether you want normal or inverse scaling!')
    
    
    if save_or_return and inverse == False:
        hf = h5py.File('redshift'+redshift+'root'+root+'.h5', 'w')
        hf.create_dataset('delta_HI', data=whole_new_f)
        hf.close()
    elif save_or_return and inverse == True:
        raise Exception('Why do you want to save the inverse transformed?\nIts the normal one!')
    else:
        return whole_new_f
    
    
    
    
    
def inverse_transform_func(cube, inverse_type, sampled_dataset):  
    """
    Inverse Transform the Input Cube
    # minmax01 / minmaxneg11 / std_noshift / std / 
    4_root / 6_root / 8_root / 16_root
    """
    if inverse_type == "minmax01":
        cube = minmax_scale(cube_tensor = cube, 
                 inverse = True,
                 min_cube = sampled_dataset.min_raw_val, 
                 max_cube = sampled_dataset.max_raw_val, 
                 redshift = False, 
                 save_or_return = False)
    elif inverse_type == "minmaxneg11":
        cube = minmax_scale_neg11(cube_tensor = cube, 
                 inverse = True,
                 min_cube = sampled_dataset.min_raw_val, 
                 max_cube = sampled_dataset.max_raw_val, 
                 redshift = False, 
                 save_or_return = False)
    elif inverse_type == "std_noshift":
        cube = standardize(cube_tensor = cube, 
                 inverse = True,
                 mean_cube = sampled_dataset.mean_raw_val, 
                 stddev_cube = sampled_dataset.stddev_raw_val, 
                 shift = False,
                 redshift = False, 
                 save_or_return = False)
    elif inverse_type == "std":
        cube = standardize(cube_tensor = cube, 
                 inverse = True,
                 mean_cube = sampled_dataset.mean_raw_val, 
                 stddev_cube = sampled_dataset.stddev_raw_val, 
                shift = True,
                 redshift = False, 
                 save_or_return = False)
    elif inverse_type == "4_root":
        cube = root_transform(cube_tensor = cube, 
                 inverse = True,
                 root = 4,
                 redshift = False, 
                 save_or_return = False) 
    elif inverse_type == "6_root":
        cube = root_transform(cube_tensor = cube, 
                 inverse = True,
                 root = 6,
                 redshift = False, 
                 save_or_return = False) 
    elif inverse_type == "8_root":
        cube = root_transform(cube_tensor = cube, 
                 inverse = True,
                 root = 8,
                 redshift = False, 
                 save_or_return = False)      
    elif inverse_type == "16_root":
        cube = root_transform(cube_tensor = cube, 
                 inverse = True,
                 root = 16,
                 redshift = False, 
                 save_or_return = False)  
    else:
        print("not implemented yet!")
    
#     print("New mean = " + str(np.mean(cube)))
#     print("New median = " + str(np.median(cube)))
#     print("New min = " + str(np.amin(cube)))
#     print("New max = " + str(np.amax(cube)))
    
    return cube


