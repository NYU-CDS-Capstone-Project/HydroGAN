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
    # print("Looking whether " + str(stat_cube_file) + " exits!")
    
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
    
def root_transform_neg11(cube_tensor, 
                         inverse,
                         root,  # the fraction corresponding to the root
                         min_cube,
                         max_cube,
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
            after_root = np.power(cube_tensor[i:i+1,:,:] , 1.0/root)
            whole_new_f[i:i+1,:,:] = 2.0 * (after_root - min_cube) / (max_cube - min_cube) - 1.0

        
    elif inverse == True:
        root_min_cube = np.power(min_cube, 1.0 / root)
        root_max_cube = np.power(max_cube, 1.0 / root)
        cube_tensor = (cube_tensor + 1) * (root_max_cube - root_min_cube) / 2.0 + (root_min_cube)
        whole_new_f = np.power(cube_tensor,root)

    else:
        raise Exception('Please specify whether you want normal or inverse scaling!')
    
    
    if save_or_return and inverse == False:
        hf = h5py.File('redshift' + redshift + 'root' + root + '.h5', 'w')
        hf.create_dataset('delta_HI', data=whole_new_f)
        hf.close()
    elif save_or_return and inverse == True:
        raise Exception('Why do you want to save the inverse transformed?\nIts the normal one!')
    else:
        return whole_new_f    





def root_transform_01(cube_tensor, 
                         inverse,
                         root,  # the fraction corresponding to the root
                         min_cube,
                         max_cube,
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
            after_root = np.power(cube_tensor[i:i+1,:,:] , 1.0/root)
            whole_new_f[i:i+1,:,:] = (after_root - min_cube) / (max_cube - min_cube) 

        
    elif inverse == True:
        root_min_cube = np.power(min_cube, 1.0 / root)
        root_max_cube = np.power(max_cube, 1.0 / root) 
        #inverse
        cube_tensor = (cube_tensor * (root_max_cube - root_min_cube)) + root_min_cube
        whole_new_f = np.power(cube_tensor,root)

    else:
        raise Exception('Please specify whether you want normal or inverse scaling!')
    
    
    if save_or_return and inverse == False:
        hf = h5py.File('redshift' + redshift + 'root' + root + '.h5', 'w')
        hf.create_dataset('delta_HI', data=whole_new_f)
        hf.close()
    elif save_or_return and inverse == True:
        raise Exception('Why do you want to save the inverse transformed?\nIts the normal one!')
    else:
        return whole_new_f    

"""
ON THE FLY TRANSFORMATION FUNCTIONS
""" 
def scale_01(cube, raw_cube_min, raw_cube_max, inverse):  
    if inverse == False :
        output = (cube - raw_cube_min) / (raw_cube_max - raw_cube_min) 
    else :
        output = (cube * (raw_cube_max  - raw_cube_min)) + raw_cube_min
    return output


def scale_neg11(cube, raw_cube_min, raw_cube_max, inverse): 
    if inverse == False :
        output = 2.0 * (cube - raw_cube_min) / (raw_cube_max - raw_cube_min) - 1.0
    else :
        output = (cube + 1.0) * (raw_cube_max - raw_cube_min) / 2.0 + (raw_cube_min)
    return output 

def root_transform(cube, raw_cube_min, raw_cube_max, inverse, root):
    if inverse == False :
        output = np.power(cube , 1.0/root)
    else :
        output = np.power(cube,root)
        
    return output 

def root_transform_neg11(cube, raw_cube_min, raw_cube_max, inverse, root):
    if inverse == False :
        output = 2.0 * (np.power(cube , 1.0/root) - raw_cube_min) / (raw_cube_max - raw_cube_min) - 1.0
        
    else :
        root_min_cube = np.power(raw_cube_min, 1.0 / root)
        root_max_cube = np.power(raw_cube_max, 1.0 / root)
        cube = (cube + 1) * (root_max_cube - root_min_cube) / 2.0 + (root_min_cube)
        output = np.power(cube,root)
        
    return output 

def root_transform_01(cube, raw_cube_min, raw_cube_max, inverse,root):
    
    root_min_cube = np.power(raw_cube_min, 1.0 / root)
    root_max_cube = np.power(raw_cube_max, 1.0 / root)
    
    if inverse == False:
        cube = (np.power(cube , 1.0/root) - root_min_cube) / (root_max_cube - root_min_cube)
    else:
        cube = (cube * (root_max_cube - root_min_cube)) + root_min_cube
        cube = np.power(cube , root)

    return output 

def transform_func(cube,inverse_type,self):
    """
    Transform the Input Cube
    # scale_01 / scale_neg11 / root / root_scale_01 / root_scale_neg11
    """
    if inverse_type == "scale_01":
        cube = scale_01(cube = cube, 
                        raw_cube_min = self.min_raw_val, 
                        raw_cube_max = self.max_raw_val, 
                        inverse = False)
    
    elif inverse_type == "scale_neg11":
        cube = scale_neg11(cube = cube, 
                        raw_cube_min = self.min_raw_val, 
                        raw_cube_max = self.max_raw_val, 
                        inverse = False)
        
    elif inverse_type == "root":
        cube = root_transform(cube = cube, 
                       raw_cube_min = self.min_raw_val, 
                       raw_cube_max = self.max_raw_val, 
                       inverse = False, 
                       root = sampled_dataset.root)
                
    elif inverse_type == "root_scale_01":
        cube = root_transform_01(cube = cube, 
                                    raw_cube_min = self.min_raw_val, 
                                    raw_cube_max = self.max_raw_val, 
                                    inverse = False, 
                                    root = sampled_dataset.root)
                         
    elif inverse_type == "root_scale_neg11":
        cube = root_transform_neg11(cube = cube, 
                                    raw_cube_min = self.min_raw_val, 
                                    raw_cube_max = self.max_raw_val, 
                                    inverse = False, 
                                    root = sampled_dataset.root)
        
    else:
        print("not implemented yet!")   
    
    return cube
    

    
    
def inverse_transform_func(cube, inverse_type, sampled_dataset):  
    """
    Inverse Transform the Input Cube
    # scale_01 / scale_neg11 / root / root_scale_01 / root_scale_neg11
    """
    if inverse_type == "scale_01":
        cube = scale_01(cube = cube, 
                        raw_cube_min = sampled_dataset.min_raw_val, 
                        raw_cube_max = sampled_dataset.max_raw_val, 
                        inverse = True)
    
    elif inverse_type == "scale_neg11":
        cube = scale_neg11(cube = cube, 
                        raw_cube_min = sampled_dataset.min_raw_val, 
                        raw_cube_max = sampled_dataset.max_raw_val, 
                        inverse = True)
        
    elif inverse_type == "root":
        cube = root_transform(cube = cube, 
                       raw_cube_min = sampled_dataset.min_raw_val, 
                       raw_cube_max = sampled_dataset.max_raw_val, 
                       inverse = True, 
                       root = sampled_dataset.root)
                
    elif inverse_type == "root_scale_01":
        cube = root_transform_01(cube = cube, 
                                    raw_cube_min = sampled_dataset.min_raw_val, 
                                    raw_cube_max = sampled_dataset.max_raw_val, 
                                    inverse = True, 
                                    root = sampled_dataset.root)
                         
    elif inverse_type == "root_scale_neg11":
        cube = root_transform_neg11(cube = cube, 
                                    raw_cube_min = sampled_dataset.min_raw_val, 
                                    raw_cube_max = sampled_dataset.max_raw_val, 
                                    inverse = True, 
                                    root = sampled_dataset.root)
        
    else:
        print("not implemented yet!")
    
#     print("New mean = " + str(np.mean(cube)))
#     print("New median = " + str(np.median(cube)))
#     print("New min = " + str(np.amin(cube)))
#     print("New max = " + str(np.amax(cube)))
    
    return cube

