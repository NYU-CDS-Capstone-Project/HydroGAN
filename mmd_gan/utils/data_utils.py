import numpy as np

def get_max_cube(f):
    max_list = [np.max(f[i:i+1,:,:]) for i in range(f.shape[0])]
    max_cube = max(max_list)
    return max_cube

def get_min_cube(f):
    min_list = [np.min(f[i:i+1,:,:]) for i in range(f.shape[0])]
    min_cube = min(min_list)
    return min_cube


def minmax_scale(cube_tensor, 
                 inverse,
                 min_cube, 
                 max_cube, 
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
            whole_new_f[i:i+1,:,:] = cube_tensor[i:i+1,:,:]*(max_cube - min_cube) + min_cube
        
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
    
    
    
    



