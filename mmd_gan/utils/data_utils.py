import numpy as np
import timeit

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
    print(whole_new_f.shape)
    
    if inverse == False:
        for i in range(cube_tensor.shape[0]):
            print(str(i + 1) + " / " + str(cube_tensor.shape[0])) if i % 250 == 0 else False
            whole_new_f[i:i+1,:,:] = (cube_tensor[i:i+1,:,:] - min_cube)/(max_cube-min_cube)
    
        print("New mean = " + str(np.mean(whole_new_f)))
        print("New median = " + str(np.median(whole_new_f)))
        assert np.amin(whole_new_f) == 0.0, "minimum not = 0!"
        assert np.amax(whole_new_f) == 1.0, "maximum not = 1!"
        
    elif inverse == True:
        for i in range(cube_tensor.shape[0]):
            print(str(i + 1) + " / " + str(cube_tensor.shape[0])) if i % 250 == 0 else False
            whole_new_f[i:i+1,:,:] = cube_tensor[i:i+1,:,:] * (max_cube - min_cube) + min_cube
        
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
            whole_new_f[i:i+1,:,:] = 2* (cube_tensor[i:i+1,:,:] - min_cube)/(max_cube-min_cube) - 1
    
        print("New mean = " + str(np.mean(whole_new_f)))
        print("New median = " + str(np.median(whole_new_f)))
        assert np.amin(whole_new_f) == 0.0, "minimum not = 0!"
        assert np.amax(whole_new_f) == 1.0, "maximum not = 1!"
        
    elif inverse == True:
        for i in range(cube_tensor.shape[0]):
            print(str(i + 1) + " / " + str(cube_tensor.shape[0])) if i % 250 == 0 else False
            whole_new_f[i:i+1,:,:] = 0.5*(1 + cube_tensor[i:i+1,:,:])*(max_cube - min_cube) + min_cube
        
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
            
        print("New mean = " + str(np.mean(whole_new_f)))
        print("New median = " + str(np.median(whole_new_f)))
        print("New min = " + str(np.amin(whole_new_f)))
        print("New max = " + str(np.amax(whole_new_f)))
        
    elif inverse == True:
        for i in range(cube_tensor.shape[0]):
            print(str(i + 1) + " / " + str(cube_tensor.shape[0])) if i % 250 == 0 else False
            if shift:
                whole_new_f[i:i+1,:,:] = cube_tensor[i:i+1,:,:]*stddev_cube + mean_cube
            else:
                whole_new_f[i:i+1,:,:] = cube_tensor[i:i+1,:,:]*stddev_cube
        
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
    
def inverse_transform_func(cube, inverse_type, sampled_dataset):  
    """
    Inverse Transform the Input Cube
    # minmax11 / minmaxneg11 / std_noshift / std
    """
    if inverse_type == "minmax11":
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
                 mean_cube = sampled_dataset.mean_raw_cube, 
                 stddev_cube = sampled_dataset.stddev_raw_cube, 
                shift = False,
                 redshift = False, 
                 save_or_return = True)
    elif inverse_type == "std":
        cube = standardize(cube_tensor = cube, 
                 inverse = True,
                 mean_cube = sampled_dataset.mean_raw_cube, 
                 stddev_cube = sampled_dataset.stddev_raw_cube, 
                shift = True,
                 redshift = False, 
                 save_or_return = True)
    else:
        print("not implemented yet!")
    
    print("New mean = " + str(np.mean(cube)))
    print("New median = " + str(np.median(cube)))
    print("New min = " + str(np.amin(cube)))
    print("New max = " + str(np.amax(cube)))
    
    return cube


