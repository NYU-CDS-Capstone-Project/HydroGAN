def train_plot_hist(epoch):

    model.train()
    train_loss = 0

    plt.figure(figsize=(20,10))

    
    for batch_idx, data in enumerate(train_loader):
        
        data = data.to(device)
        
        optimizer.zero_grad()
        
        recon_batch, first_decode_out_sum, conv_1_out_sum, relu_1_out_sum, max_unpool_1_out_sum,\
                    conv_2_out_sum, relu_2_out_sum, max_unpool_2_out_sum, conv_3_out_sum, \
                    relu_3_out_sum , mu, logvar = model(data,mode="training")
            
        loss = Beta_SSE_KLD(recon_batch, data, mu, logvar, beta=64, epsilon=1e-8)
        print ("Loss = "+str(loss))
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        loss_history.append(loss.item())

        if batch_idx in [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]:
            
            ## REAL DATA            
            real_data = (data[0].cpu().view(128,128,128).detach().numpy())
            real_cube_tr = real_data[np.nonzero(np.nan_to_num(real_data))]
            real_cube_tr = np.log10(real_cube_tr).flatten()
            
            ## RECON DATA        
            recon_data = recon_batch[0].cpu().view(128,128,128).detach().numpy()
            recon_cube_tr = recon_data[np.nonzero(np.nan_to_num(recon_data))]
            recon_cube_tr = np.log10(recon_cube_tr).flatten()

            noise_batch, first_decode_out_sum, conv_1_out_sum, relu_1_out_sum, max_unpool_1_out_sum,\
                     conv_2_out_sum, relu_2_out_sum, max_unpool_2_out_sum, conv_3_out_sum, \
                     relu_3_out_sum , mu, logvar = model(data,mode="inference")
            
            ## NOISE INPUT            
            noise_out_data = noise_batch[0].cpu().view(128,128,128).detach().numpy()
            noise_out_cube_tr = noise_out_data[np.nonzero(noise_out_data)]
            noise_out_cube_tr = np.log10(noise_out_cube_tr).flatten()

            ## HISTOGRAMS
            plt.hist(real_cube_tr, label="real cube", color="g", alpha=0.3, bins=100,
                    density=True)
            plt.hist(recon_cube_tr, label ="reconstructed cube", color="b", alpha=0.3, bins=100,
                    density=True)
            plt.hist(noise_out_cube_tr, label ="noise input cube", color="r", alpha=0.3, bins=100,
                     density=True)
        
    plt.legend()
    print('====> Epoch: {} Average loss: {:.12f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    
def train_plot_power_spec(epoch):

    model.train()
    train_loss = 0

    plt.figure(figsize=(20,10))

    for batch_idx, data in enumerate(train_loader):
        
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, first_decode_out_sum, conv_1_out_sum, relu_1_out_sum, max_unpool_1_out_sum,\
                    conv_2_out_sum, relu_2_out_sum, max_unpool_2_out_sum, conv_3_out_sum, \
                    relu_3_out_sum , mu, logvar = model(data,mode="training")
            
        loss = Beta_SSE_KLD(recon_batch, data, mu, logvar, beta=64, epsilon=1e-8)
        print ("Loss = "+str(loss))
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        loss_history.append(loss.item())

        print("data shape = " + str(data.view(-1,128,128,128).shape))
        print("data shape = " + str(data[0].cpu().view(128,128,128).numpy().shape))
        
        

        if batch_idx in [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]:
            
            real_power_spec = (data[0].cpu().view(128,128,128).detach().numpy() - 1)*max_original_cube
            recon_power_spec = (recon_batch[0].cpu().view(128,128,128).detach().numpy()-1)*max_original_cube

            noise_recon_batch, first_decode_out_sum, conv_1_out_sum, relu_1_out_sum, max_unpool_1_out_sum,\
                     conv_2_out_sum, relu_2_out_sum, max_unpool_2_out_sum, conv_3_out_sum, \
                     relu_3_out_sum , mu, logvar = model(data,mode="inference")
    
            noise_power_spec = (noise_recon_batch[0].cpu().view(128,128,128).detach().numpy()-1)*max_original_cube
        
            BoxSize=75.0/2048*128
            threads, MAS, axis = 1, "CIC", 0
            
            ## assign real data and noise batch
            delta_real_cube = real_power_spec
            delta_gen_cube = noise_power_spec
            
            delta_real_cube /= np.mean(delta_real_cube,
                              dtype=np.float64)
            
            delta_real_cube -= 1.0
            delta_real_cube = delta_real_cube.astype(np.float32)

            Pk_real_cube = PKL.Pk(delta_real_cube, BoxSize, axis, MAS, threads)

            delta_gen_cube /= np.mean(delta_gen_cube,
                                     dtype=np.float64)
            delta_gen_cube -= 1.0
            delta_gen_cube = delta_gen_cube.astype(np.float32)

            Pk_gen_cube = PKL.Pk(delta_gen_cube, BoxSize, axis, MAS, threads)

            plt.plot(np.log(Pk_real_cube.k3D), np.log(Pk_real_cube.Pk[:,0]), color="b", label="original cube")
            plt.plot(np.log(Pk_gen_cube.k3D), np.log(Pk_gen_cube.Pk[:,0]), color="r", label="jaas")
            plt.rcParams["font.size"] = 12
            plt.title("Power Spectrum Comparison")
            plt.xlabel('log(Pk.k3D)')
            plt.ylabel('log(Pk.k3D)')

    plt.legend()
    print('====> Epoch: {} Average loss: {:.12f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

