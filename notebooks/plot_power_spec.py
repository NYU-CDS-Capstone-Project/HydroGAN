def plot_power_spec(real_cube, generated_cube,
                   threads=1, MAS="CIC", axis=0, BoxSize=75.0/2048*128):
    """Takes as input;
    - Real cube: (n x n x n) torch cuda FloatTensor,
    - Generated copy: (n x n x n) torch cuda FloatTensor,
    - constant assignments: threads, MAS, axis, BoxSize.
    
    Returns;
    - Power spectrum plots of both cubes
    in the same figure.
    """
    
    ## Assert same type
    assert ((real_cube.type() == generated_cube.type())&(real_cube.type()=="torch.FloatTensor")),\
    "Both input cubes should be torch.FloatTensor or torch.cuda().FloatTensor. Got real_cube type " + real_cube.type() + ", generated_cube type " + generated_cube.type() +"."
    ## Assert equal dimensions
    assert (real_cube.size() == generated_cube.size()),\
    "Two input cubes must have the same size. Got real_cube size " + str(real_cube.size()) + ", generated cube size " + str(generated_cube.size())
    
    ## if one or both of the cubes are cuda FloatTensors, detach them
    if real_cube.type() == "torch.cuda.FloatTensor":
        ## convert cuda FloatTensor to numpy array
        real_cube = real_cube.cpu().detach().numpy()
    else:
        real_cube = real_cube.numpy()
    
    if generated_cube.type() == "torch.cuda.FloatTensor":
        ## convert cuda FloatTensor to numpy array
        generated_cube = generated_cube.cpu().detach().numpy()
    else:
        generated_cube = generated_cube.numpy()
    
    # constant assignments
    BoxSize = BoxSize
    axis = axis
    MAS = MAS
    threads = threads

    # CALCULATE POWER SPECTRUM OF THE REAL CUBE
    # SHOULD WE DIVIDE BY WHOLE CUBE MEAN OR JUST THE MEAN OF THIS PORTION
    # Ask the Team
#     delta_real_cube /= mean_cube.astype(np.float64)
    
    delta_real_cube = real_cube
    delta_gen_cube = generated_cube
    
    delta_real_cube /= np.mean(delta_real_cube,
                              dtype=np.float64)
    delta_real_cube -= 1.0
    delta_real_cube = delta_real_cube.astype(np.float32)
    
    Pk_real_cube = PKL.Pk(delta_real_cube, BoxSize, axis, MAS, threads)
    
    
    # CALCULATE POWER SPECTRUM OF THE GENERATED CUBE
    delta_gen_cube /= np.mean(delta_gen_cube,
                             dtype=np.float64)
    delta_gen_cube -= 1.0
    delta_gen_cube = delta_gen_cube.astype(np.float32)
    
    Pk_gen_cube = PKL.Pk(delta_gen_cube, BoxSize, axis, MAS, threads)
    
    plt.figure(figsize=(10,5))
    plt.plot(np.log(Pk_real_cube.k3D), np.log(Pk_real_cube.Pk[:,0]), color="b", label="original cube")
    plt.plot(np.log(Pk_gen_cube.k3D), np.log(Pk_gen_cube.Pk[:,0]), color="r", label="jaas")
    plt.rcParams["font.size"] = 12
    plt.title("Power Spectrum Comparison")
    plt.xlabel('log(Pk.k3D)')
    plt.ylabel('log(Pk.k3D)')
    plt.legend()
    
    plt.show()
    return "Power spectrum plot complete!"