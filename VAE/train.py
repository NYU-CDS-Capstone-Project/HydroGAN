def ELBO(recon_x, x, mu, logvar):
    """
    Computes ELBO =  Log Likelihood - KL Divergence 
    
    We want to maximize ELBO, so ELBO function returns 
    -1*ELBO, to pass it to the optimizer more conveniently. 
    
    """
    
    # LOG LIKELIHOOD (-CROSS ENTROPY), THE FIRST TERM OF ELBO
    
    print("--------------------------------------")
    print("Calculating Loss...")
    print("recon_x shape = " + str(recon_x.shape))
    
    BCE = F.binary_cross_entropy(recon_x, 
                                 x.view(-1, 1, 128, 128, 128), 
                                 reduction='sum')
    print("BCE Loss = " + str(BCE))
    
    LogLikelihood = (-1)*BCE

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    print ("KLD =" +str(KLD))
    ELBO = (LogLikelihood - KLD)/float(batch_size)
    print ("ELBO = "+str(ELBO))
    
    return (-1)*ELBO

decoder_sum_lists = {}

for out_plot in ["first_decode_out_sum", "conv_1_out_sum", "relu_1_out_sum", "max_unpool_1_out_sum",
            "conv_2_out_sum", "relu_2_out_sum", "max_unpool_2_out_sum", "conv_3_out_sum", \
            "relu_3_out_sum"]:

    decoder_sum_lists[out_plot] = []

def train(epoch):
    model.train()
    train_loss = 0
#     for batch_idx, (data, _) in enumerate(train_loader):
    for batch_idx, data in enumerate(train_loader):
#         print(batch_idx)
#         print(data)
        
        #print("Batch size = " + str(data.shape))
        
        data = data.to(device)
        print("Data transfer to device completed.")
        
        optimizer.zero_grad()
        
        recon_batch, first_decode_out_sum, conv_1_out_sum, relu_1_out_sum, max_unpool_1_out_sum,\
                    conv_2_out_sum, relu_2_out_sum, max_unpool_2_out_sum, conv_3_out_sum, \
                    relu_3_out_sum , mu, logvar = model(data)
            
        # Plotting Input Cube
        print("data shape = " + str(data.view(-1,128,128,128).shape))
        print("data shape = " + str(data[0].cpu().view(128,128,128).numpy().shape))
        
#         if epoch % (epochs / 20) == 0 and batch_idx == 1:
            
            ## plot first_decoder out


        if epoch > 1 and epoch % (epochs / 50) == 0 and batch_idx == 0:
            print ("Plotting the first decoder output sum")
#         print ("first decoder out sum = "+str(first_decode_out_sum))
            plt.figure(figsize=(20,10))
            plt.plot(first_decode_out_sum, linewidth=3.5, alpha=0.6, 
                     color="b")
            plt.rcParams["font.size"] = 16
            plt.xlabel("Batch Iterations")
            plt.title("First Decoder Output Sum")
    #             plt.xticks = range(epochs)
    #             plt.grid()
            plt.show()

            ## plot first convolution out
            print ("Plotting the first convolution output sum")
            plt.figure(figsize=(20,10))
            plt.plot(conv_1_out_sum, linewidth=3.5, alpha=0.6, 
                     color="r")
            plt.rcParams["font.size"] = 16
            plt.xlabel("Batch Iterations")
            plt.title("First Convolution Output Sum")
    #             plt.xticks = range(epochs)
    #             plt.grid()
            plt.show()

            ## plot first relu out
            print ("Plotting the first relu output sum")
            plt.figure(figsize=(20,10))
            plt.plot(relu_1_out_sum, linewidth=3.5, alpha=0.6, 
                     color="m")
            plt.rcParams["font.size"] = 16
            plt.xlabel("Batch Iterations")
            plt.title("First ReLU Output Sum")
    #             plt.xticks = range(epochs)
    #             plt.grid()
            plt.show()

            ## plot first max unpool out
            print ("Plotting the first max unpool output sum")
            plt.figure(figsize=(20,10))
            plt.plot(max_unpool_1_out_sum, linewidth=3.5, alpha=0.6, 
                     color="darkorange")
            plt.rcParams["font.size"] = 16
            plt.xlabel("Batch Iterations")
            plt.title("First Max Unpool Output Sum")
    #             plt.xticks = range(epochs)
    #             plt.grid()
            plt.show()

            ## plot second convolution out
            print ("Plotting the second convolution output sum")
            plt.figure(figsize=(20,10))
            plt.plot(conv_2_out_sum, linewidth=3.5, alpha=0.6, 
                     color="r")
            plt.rcParams["font.size"] = 16
            plt.xlabel("Batch Iterations")
            plt.title("Second Convolution Output Sum")
    #             plt.xticks = range(epochs)
    #             plt.grid()
            plt.show()

            ## plot second relu out
            print ("Plotting the second relu output sum")
            plt.figure(figsize=(20,10))
            plt.plot(relu_2_out_sum, linewidth=3.5, alpha=0.6, 
                     color="m")
            plt.rcParams["font.size"] = 16
            plt.xlabel("Batch Iterations")
            plt.title("Second ReLU Output Sum")
    #             plt.xticks = range(epochs)
    #             plt.grid()
            plt.show()

            ## plot second max unpool out
            print ("Plotting the second max unpool output sum")
            plt.figure(figsize=(20,10))
            plt.plot(max_unpool_2_out_sum, linewidth=3.5,
                     alpha=0.6, 
                     color="darkorange")
            plt.rcParams["font.size"] = 16
            plt.xlabel("Batch Iterations")
            plt.title("Second Max Unpool Output Sum")
    #             plt.xticks = range(epochs)
    #             plt.grid()
            plt.show()

            ## plot third convolution out
            print ("Plotting the third convolution output sum")
            plt.figure(figsize=(20,10))
            plt.plot(conv_3_out_sum, linewidth=3.5, alpha=0.6,
                     color="r")
            plt.rcParams["font.size"] = 16
            plt.xlabel("Batch Iterations")
            plt.title("Third Convolution Output Sum")
    #             plt.xticks = range(epochs)
    #             plt.grid()
            plt.show()

            ## plot third relu out
            print ("Plotting the third relu output sum")
            plt.figure(figsize=(20,10))
            plt.plot(relu_3_out_sum, linewidth=3.5, alpha=0.6, 
                     color="m")
            plt.rcParams["font.size"] = 16
            plt.xlabel("Batch Iterations")
            plt.title("Third ReLU Output Sum")
    #             plt.xticks = range(epochs)
    #             plt.grid()
            plt.show()
            print ("Visualizing original cube")
            visualize_cube(cube=data[0].cpu().view(128,128,128).detach().numpy(),
                           edge_dim = 128,
                           start_cube_index_x = 0,
                          start_cube_index_y = 0,
                          start_cube_index_z = 0,
                          fig_size = (20,20),
                          norm_multiply = 1000,
                          color_map = "Blues",
                          lognormal = False)
            
            print ("Visualizing reconstructed cube")
            visualize_cube(cube=recon_batch[0].cpu().view(128,128,128).detach().numpy(),
                           edge_dim = 128,
                           start_cube_index_x = 0,
                          start_cube_index_y = 0,
                          start_cube_index_z = 0,
                          fig_size = (20,20),
                          norm_multiply = 1000,
                          color_map = "Blues",
                          lognormal = False)
            
            print ("Output mass sum (reconstructed batch): "\
                   + str(np.sum(recon_batch[0].cpu().view(128,128,128).detach().numpy())))
        
        # ACTUALLY MINUS ELBO SO THAT OPTIMIZER CAN MINIMIZE IT
        loss = ELBO(recon_batch, data, mu, logvar)
        
        loss.backward()
        train_loss += loss.item()
        
        loss_history.append(loss.item())
        
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.12f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.12f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    

if __name__ == "__main__":
    
    loss_history = []
    
    decoder_sum_lists = {}
        
    for out_plot in ["first_decode_out_sum", "conv_1_out_sum", "relu_1_out_sum", "max_unpool_1_out_sum",
                    "conv_2_out_sum", "relu_2_out_sum", "max_unpool_2_out_sum", "conv_3_out_sum", \
                    "relu_3_out_sum"]:
            
        decoder_sum_lists[out_plot] = []
            
    for epoch in range(1, epochs + 1):
        print("Epoch = " + str(epoch) + " / " + str(epochs))
        
        train(epoch)
        
        # Plotting Training Losses
        plt.figure(figsize=(20,10))
        plt.plot(loss_history, linewidth=3.5, alpha=0.6,
                color="crimson", marker="o")
        plt.rcParams["font.size"] = 16
        plt.title("Loss History (-1*ELBO)")
        plt.show()