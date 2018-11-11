# Skeleton code taken from: https://chrisorm.github.io/AVB-pyt.html
# Thanks to Rob Fergus: https://cs.nyu.edu/~fergus/teaching/vision/10_unsupervised.pdf

class VAE(nn.Module):
    def __init__(self, mode="training"):
        super(VAE, self).__init__()
        
        """Takes as input;
        - mode: 'training' or 'inference'. 
        If mode == 'training' reparametrizes encoder output and 
        reconstructs the given input, 
        Elif mode == 'inference', a sample (batch_size x hidden_size)
        drawn from a zero-mean, univariance Gaussian is passed to the 
        decoder."""
        
        # pass train or test mode to adjust decoder input
        # as reconstruction or noise
        self.mode = mode
        
        ########################
        #### ENCODER LAYERS ####
        ########################
        
        # Convolutional Layer 1
        self.encode_conv1 = nn.Conv3d(in_channels=1, 
                                      out_channels=8, 
                                      kernel_size=(4,4,4), # == 4
                                      stride = (2,2,2), # == 2
                                      padding=(1,1,1)) # == 1
        nn.init.xavier_uniform_(self.encode_conv1.weight) #Xaviers Initialization
        
        self.encode_relu1 = nn.ReLU()
        
        self.encode_maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2),
                                            return_indices = True)
        
        # Convolutional Layer 2
        self.encode_conv2 = nn.Conv3d(in_channels=8, 
                                      out_channels=16, 
                                      kernel_size=(4,4,4), # == 4 
                                      stride = (2,2,2),
                                      padding=(1,1,1))
        nn.init.xavier_uniform_(self.encode_conv2.weight) #Xaviers Initialization
        
        self.encode_relu2 = nn.ReLU()
        self.encode_maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), 
                                             stride=(2, 2, 2),
                                            return_indices = True)
        
        # Convolutional Layer 3
        self.encode_conv3 = nn.Conv3d(in_channels=16, 
                                      out_channels=32, 
                                      kernel_size=(4,4,4), # == 4 
                                      stride = (2,2,2),
                                      padding=(1,1,1))
        self.encode_relu3 = nn.ReLU()
        nn.init.xavier_uniform_(self.encode_conv3.weight) #Xaviers Initialization
        
        
        """
        Fully Connected Layers after 3D Convolutional Layers
        First FC layer's input should be equal to 
        last convolutional layer's output
        8192 = 8^3 * 16 
            8^3 = (output of 2nd convolutional layer)
            16 = number of out_channels
        """
        
        # First Fully Connected Layer
        self.encode_fc1_linear = nn.Linear(in_features=2048, ## 2048
                                           out_features=128)
        
        self.encode_fc1_relu = nn.ReLU()
        self.encode_fc1_dropout = nn.Dropout(0.5)
        nn.init.xavier_uniform_(self.encode_fc1_linear.weight) #Xaviers Initialization
        
        # Second Fully Connected Layer
        self.encode_fc2_linear = nn.Linear(in_features=128, 
                                           out_features=128)
    

        self.encode_fc2_relu = nn.ReLU()
    
        self.encode_fc2_dropout = nn.Dropout(0.5)
        nn.init.xavier_uniform_(self.encode_fc2_linear.weight) #Xaviers Initialization
        
        """
        The last fully connected layer's output is the dimensions
        of the embeddings?
        
        PyTorch VAE example uses output of 20 dimensions for mu &
        logvariance
        """
        
        # Last Fully Connected Layer
        self.encode_fc31_linear = nn.Linear(in_features=128,
                                            out_features=16) 
        
        self.encode_fc31_relu = nn.ReLU()
        
        self.encode_fc31_dropout = nn.Dropout(0.5)
        nn.init.xavier_uniform_(self.encode_fc31_linear.weight) #Xaviers Initialization

        self.encode_fc32_linear = nn.Linear(in_features=128, 
                                           out_features=16)
        
        self.encode_fc32_relu = nn.ReLU()
    
        self.encode_fc32_dropout = nn.Dropout(0.5)
        nn.init.xavier_uniform_(self.encode_fc32_linear.weight) #Xaviers Initialization
        
        ########################
        #### DECODER LAYERS ####
        ########################
        
        # First Fully Connected Layer
        self.decode_fc1 = nn.Sequential(
            nn.Linear(in_features=16,
                      out_features=128))
        
        # Second Fully Connected Layer
        self.decode_fc2 = nn.Sequential(
            nn.Linear(in_features=128, 
                      out_features=128), 
            nn.ReLU(),
            nn.Dropout(0.5))
        
        # Third Fully Connected Layer
        self.decode_fc3 = nn.Sequential(
            nn.Linear(in_features = 128,
                      out_features = 2048),
            nn.ReLU(),
            nn.Dropout(0.5))
        
        # Convolutional Layer 1
        self.decode_conv1 = nn.ConvTranspose3d(in_channels=32, 
                                              out_channels=16, 
                                              kernel_size=(4,4,4),
                                              stride = (2,2,2),
                                              padding=(1,1,1))
        self.decode_relu1 = nn.ReLU()
        self.decode_maxunpool1 = nn.MaxUnpool3d(kernel_size=(2, 2, 2), 
                                                     stride=(2, 2, 2))
        
        # Convolutional Layer 2
        self.decode_conv2 = nn.ConvTranspose3d(in_channels=16, 
                                              out_channels=8, 
                                              kernel_size=(4,4,4),
                                              stride = (2,2,2),
                                              padding=(1,1,1))
        self.decode_relu2 = nn.ReLU()
        self.decode_maxunpool2 = nn.MaxUnpool3d(kernel_size=(2, 2, 2), 
                                                     stride=(2, 2, 2))
        # Convolutional Layer 3
        self.decode_conv3 = nn.ConvTranspose3d(in_channels=8, 
                                              out_channels=1, 
                                              kernel_size=(4,4,4),
                                              stride = (2,2,2),
                                              padding=(1,1,1))
        self.decode_relu3 = nn.ReLU()
        self.decode_maxunpool3 = nn.MaxUnpool3d(kernel_size=(2, 2, 2), 
                                                     stride=(2, 2, 2))
        
        ## initialize out lists 
        self.first_decode_out_sum = []
        self.conv_1_out_sum = []
        self.relu_1_out_sum = []
        self.max_unpool_1_out_sum = []
        self.conv_2_out_sum = []
        self.relu_2_out_sum = []
        self.max_unpool_2_out_sum = []
        self.conv_3_out_sum = []
        self.relu_3_out_sum = []
        
        
        # Encoding part of VAE
    def encode(self, x):
#         h1 = F.relu(self.fc1(x))
#         return self.fc21(h1), self.fc22(h1)

        print("Starting Encoding")
#         print("----------------------------")
        
        out = self.encode_conv1(x)
#         print("First Conv output shape = " + str(out.shape))
        #print(out.shape)
        out = self.encode_relu1(out)
#         print("First ReLU Layer output shape = " + str(out.shape))
        size1 = out.size()
        out, ind1 = self.encode_maxpool1(out)
#         print("First MaxPooling output shape = " + str(out.shape))
#         print("Ind1 shape = " + str(ind1.shape))
#         #print("Size1 = " + str(size1))
#         print("----------------------------")
        
        out = self.encode_conv2(out)
#         print("Second Conv output shape = " + str(out.shape))
        out = self.encode_relu2(out)
#         print("Second ReLU Layer output shape = " + str(out.shape))
        size2 = out.size()
        out, ind2 = self.encode_maxpool2(out)
#         print("Second MaxPooling output shape = " + str(out.shape))
#         print("Ind2 shape = " + str(ind2.shape))
        #print("Size2 = " + str(size2))
#          print("----------------------------")
        
        out = self.encode_conv3(out)
#         print("Last Conv output shape = " + str(out.shape))
        out = self.encode_relu3(out)
#         print("Last ReLU output shape = " + str(out.shape))
        size3 = out.size()
#         out, ind3 = self.encode_maxpool3(out)
#         print("Last Conv Layer output shape = " + str(out.shape))
#         print("Ind3 shape = " + str(ind3.shape))
#         print("Size3 = " + str(size3))
#         print("----------------------------")

        """
        From here on, the convolutional layers' output is flattened
        into a rank 1 tensor of size x & put into a fully connected 
        network to output ??????
        
        https://github.com/pytorch/examples/blob/master/vae/main.py
        PyTorch's own example uses just 2 fully-connected layers
        to output mu and logvar predictions, below we use 3.
        """
        #out = out.view(out.size(0), -1)
        
        
        # batch_size = 1 - WORKS
#         out = out.view(1, -1)
        # batch_size != 1
        out = out.view(batch_size, -1)
#         print(out.shape)
        
#         print("Last Conv Layer output shape after reshaping \n \
#                 (Input to first FC layer) = " + str(out.shape))
        
#         out = self.encode_fc1(out)
    
        out = self.encode_fc1_linear(out)
        
        ## RELU TO LEAKY RELU
        out = self.encode_fc1_relu(out)
#         out = self.encode_LeakyReLU_1(out)
        
        out = self.encode_fc1_dropout(out)
#         out = self.encode_fc2(out)

        out = self.encode_fc2_linear(out)
    
        ## RELU TO LEAKY RELU
        out = self.encode_fc2_relu(out)
#         out = self.encode_LeakyReLU_2(out)
        out = self.encode_fc2_dropout(out)
        
        
#         out_mu = self.encode_fc31(out)
        
        out_mu = self.encode_fc31_linear(out)
        
        ## RELU TO LEAKY RELU
        out_mu = self.encode_fc31_relu(out_mu)
        out_mu = self.encode_fc31_dropout(out_mu)
        
#         out_logvar = self.encode_fc32(out)

        out_logvar = self.encode_fc32_linear(out)
    
        ## RELU TO LEAKY RELU
        out_logvar = self.encode_fc32_relu(out_logvar)
#         out_logvar = self.encode_LeakyReLU_32(out_logvar)
        
        out_logvar = self.encode_fc32_dropout(out_logvar)
        
        print("Encode - Forward Pass Finished")
#         print(out_mu.shape)
#         print(out_logvar.shape)
#         print("----------------------------")

        return out_mu, out_logvar, [ind1,ind2], [size1,size2]
    
        
#     Reparametrization Trick in training mode,
#     N(0,1) Gaussian sample for inference mode.
    def reparameterize(self, mu, logvar):
        """
        torch.exp = returns a new tensor with the exponential of 
                    the elements of input
        rand_like = returns a tensor with the same size as input
                    that is filled with random numbers from a normal
                    distribution with mean 0 and variance 1
        
        """
        assert self.mode in ["training","inference"], "Mode should be either 'training' or 'inference'."
        
        if self.mode == "training":
            mu = mu
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(mu)
        
        elif self.mode == "inference":
            # input values from 0-mean univariance 
            # Gaussian distribution
            
            # sample from Gaussian
            ## sampling from normal
            
            m = normal.Normal(0,1)
            noise = m.sample(mu.data.shape) # returns torch.Size([x,y,z])
            z = noise
        
        return z
        
        
    def decode(self, z, indices_list, size_list):
        
        print("Starting Decoding")
#         print("z shape = " + str(z.shape))
        
        ### buradaki butun outlarin sumlarina bak 
        out = self.decode_fc1(z)
#         print("1st FC output shape = " + str(out.shape))
        out = self.decode_fc2(out)
#         print("2nd FC output shape = " + str(out.shape))
        out = self.decode_fc3(out)
#         print("Last FC output shape = " + str(out.shape))
        
        # batch_size = 1 - WORKS
#         out = out.view(1, 32, 4, 4, 4)
        # batch_size != 1 
        out = out.view(batch_size, 32, 4, 4, 4)
#         out = out.view(batch_size, 32, )
        self.first_decode_out_sum.append(np.sum(out.cpu().detach().numpy()))
        print ("FIRST DECODE OUT SUM = "+ str(np.sum(out.cpu().detach().numpy())))
        
#         print("First Deconv input shape = " + str(out.shape))
#         print("After last convolution (encoding stage) output shape = " +\
#                   str(indices_list[1].shape))
        out = self.decode_conv1(out)
        self.conv_1_out_sum.append(np.sum(out.cpu().detach().numpy()))
        print ("CONV 1 OUT SUM = "+ str(np.sum(out.cpu().detach().numpy())))
#         print("First Deconv output shape = " + str(out.shape))
        
        ## RELU TO LEAKY RELU
        out = self.decode_relu1(out)
        self.relu_1_out_sum.append(np.sum(out.cpu().detach().numpy()))

        print ("RELU 1 OUT SUM = "+ str(np.sum(out.cpu().detach().numpy())))
#         print("First ReLU output shape = " + str(out.shape))
        # maxunpooling needs indices

#         out = self.decode_maxunpool1(out,
#                              indices = indices_list[1])
        out = self.decode_maxunpool1(out,
                                     indices = indices_list[1],
                                     output_size = size_list[1])
    
        self.max_unpool_1_out_sum.append(np.sum(out.cpu().detach().numpy()))
        print ("MAX UNPOOL 1 OUT SUM = "+ str(np.sum(out.cpu().detach().numpy())))
#         print("2nd MaxUnpool ouput shape = " + str(out.shape))
        
        out = self.decode_conv2(out)
        self.conv_2_out_sum.append(np.sum(out.cpu().detach().numpy()))
        print ("CONV 2 OUT SUM = "+ str(np.sum(out.cpu().detach().numpy())))
#         print("2nd Deconv output shape = " + str(out.shape))

        ## RELU TO LEAKY RELU
        out = self.decode_relu2(out)
        self.relu_2_out_sum.append(np.sum(out.cpu().detach().numpy()))

        print ("RELU 2 OUT SUM = "+ str(np.sum(out.cpu().detach().numpy())))
#         print("2nd ReLU output shape = " + str(out.shape))
        out = self.decode_maxunpool2(out,
                     indices = indices_list[0])
        self.max_unpool_2_out_sum.append(np.sum(out.cpu().detach().numpy()))
        print ("MAX UNPOOL 2 OUT SUM = "+ str(np.sum(out.cpu().detach().numpy())))
#         out = self.decode_maxunpool2(out,
#                                      indices= indices_list[1],
#                                      output_size = size_list[1])
        
        out = self.decode_conv3(out)
        self.conv_3_out_sum.append(np.sum(out.cpu().detach().numpy()))
        print ("CONV 3 OUT SUM = "+ str(np.sum(out.cpu().detach().numpy())))
        
        ## RELU TO LEAKY RELU
        out = self.decode_relu3(out)
        print ("decoder out size = "+str(out.size()))
        self.relu_3_out_sum.append(np.sum(out.cpu().detach().numpy()))

        print ("RELU 3 OUT SUM = "+ str(np.sum(out.cpu().detach().numpy())))
        
        return out, self.first_decode_out_sum, self.conv_1_out_sum, self.relu_1_out_sum, \
                    self.max_unpool_1_out_sum,\
                    self.conv_2_out_sum, self.relu_2_out_sum, self.max_unpool_2_out_sum, \
                    self.conv_3_out_sum, \
                    self.relu_3_out_sum
    
    # Forward Pass
    def forward(self, x):
        
        if self.mode == "inference":
            # will use these for size consistency only,
            # we are not inputting any encoding in inference mode
            # besides N(0,1) - 0-mean, univariance Gaussian noise
            mu, logvar, indices_list, size_list = self.encode(x)
            # input noise in the decoder instead of reparametrized input
            # please check the reparametrize function for further clarification
            z = self.reparameterize(mu, logvar)
            # return noise-input reconstruction (reconstructed_x)
            # returning all these output sums to check if the decoder
            # works correctly
            output_cube,\
            first_decode_out_sum, conv_1_out_sum, relu_1_out_sum, max_unpool_1_out_sum,\
            conv_2_out_sum, relu_2_out_sum, max_unpool_2_out_sum, conv_3_out_sum, \
            relu_3_out_sum = self.decode(z, indices_list, size_list)
            
        elif self.mode == "training":
            # if the VAE network is in training mode,
            # we reparametrize the encoder output, then pass the
            # reparametrized representation into the decoder for 
            # reconstructuion. Here we are only reconstructing whatever 
            # "cube" sample of the space was passed into the encoder.
            mu, logvar, indices_list, size_list = self.encode(x)
            z = self.reparameterize(mu, logvar)
            
            output_cube,\
            first_decode_out_sum, conv_1_out_sum, relu_1_out_sum, max_unpool_1_out_sum,\
            conv_2_out_sum, relu_2_out_sum, max_unpool_2_out_sum, conv_3_out_sum,\
            relu_3_out_sum = self.decode(z, indices_list, size_list)
        
        # changed the output name to output_cube (from reconstructed_x) because it 
        # was confusing when we are running in inference mode. Since, in 
        # the inference/noise input case it's just a cube generation using the
        # trained network rather than "reconstructing" an input.
        return output_cube,\
               first_decode_out_sum, conv_1_out_sum, relu_1_out_sum, max_unpool_1_out_sum,\
               conv_2_out_sum, relu_2_out_sum, max_unpool_2_out_sum, conv_3_out_sum,\
               relu_3_out_sum , mu, logvar