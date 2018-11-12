import torch.nn as nn
from utils.conv_utils import calculate_conv_output_dim, calculate_pool_output_dim

# input: batch_size * nc * 64 * 64
# output: batch_size * k * 1 * 1
class Encoder(nn.Module):
    def __init__(self, cube_dimension, fc1_hidden_dim, fc2_output_dim, 
                embedding_dim, leakyrelu_const, pool_return_indices):        
        super(Encoder, self).__init__()

        self.pool_return_indices = pool_return_indices
      
        # First Convolutional Layer
        self.conv1_in_channels = 1
        self.conv1_out_channels = 4
        self.conv1_kernel = 3
        self.conv1_stride = 1
        self.conv1_padding = 0
        conv1_output_dim = calculate_conv_output_dim(D=cube_dimension,
                                        K=self.conv1_kernel,
                                        P=self.conv1_padding,
                                        S=self.conv1_stride)
        print("Conv1 Output Dimension = " + str(conv1_output_dim))
        self.conv1_encode = nn.Conv3d(in_channels=self.conv1_in_channels, 
                                    out_channels=self.conv1_out_channels, 
                                    kernel_size=self.conv1_kernel, 
                                    stride =self.conv1_stride, 
                                    padding=self.conv1_padding)     
        nn.init.xavier_uniform_(self.conv1_encode.weight)
        self.bn1_encode = nn.BatchNorm3d(num_features = self.conv1_out_channels)
        self.leakyrelu1 = nn.LeakyReLU(leakyrelu_const, inplace=True)

        # First Pooling
        self.pool1_kernel = 2
        self.pool1_stride = 2 
        pool1_output_dim = calculate_pool_output_dim(D=conv1_output_dim,
                                                    K=self.pool1_kernel,
                                                    S=self.pool1_stride)
        print("Pool1 Output Dimension = " + str(pool1_output_dim)) 
#         self.pool1_encode = nn.MaxPool3d(kernel_size=self.pool1_kernel, 
#                                             stride=self.pool1_stride,
#                                             return_indices = self.pool_return_indices)
        self.pool1_encode = nn.AvgPool3d(kernel_size=self.pool1_kernel, 
                                         stride=self.pool1_stride, 
                                         padding=0, 
                                         ceil_mode=False, 
                                         count_include_pad=True)
        

        # Second Convolutional Layer
        self.conv2_in_channels = self.conv1_out_channels
        self.conv2_out_channels = 24
        self.conv2_kernel = 4
        self.conv2_stride = 1
        self.conv2_padding = 0
        conv2_output_dim = calculate_conv_output_dim(D=pool1_output_dim,
                                        K=self.conv2_kernel,
                                        P=self.conv2_padding,
                                        S=self.conv2_stride)
        print("Conv2 Output Dimension= " + str(conv2_output_dim))
        self.conv2_encode = nn.Conv3d(in_channels=self.conv2_in_channels, 
                                    out_channels=self.conv2_out_channels, 
                                    kernel_size=self.conv2_kernel, 
                                    stride =self.conv2_stride, 
                                    padding=self.conv2_padding)     
        nn.init.xavier_uniform_(self.conv2_encode.weight)  
        self.bn2_encode = nn.BatchNorm3d(num_features = self.conv2_out_channels)
        self.leakyrelu2 = nn.LeakyReLU(leakyrelu_const, inplace=True)

        # Second Pooling
        self.pool2_kernel = 2
        self.pool2_stride = 2 
        pool2_output_dim = calculate_pool_output_dim(D=conv2_output_dim,
                                                K=self.pool2_kernel,
                                                S=self.pool2_stride)
        print("Pool2 Output Dimension = " + str(pool2_output_dim)) 
#         self.pool2_encode = nn.MaxPool3d(kernel_size=self.pool2_kernel, 
#                                             stride=self.pool2_stride,
#                                             return_indices = self.pool_return_indices)   
        self.pool2_encode = nn.AvgPool3d(kernel_size=self.pool2_kernel, 
                                         stride=self.pool2_stride, 
                                         padding=0, 
                                         ceil_mode=False, 
                                         count_include_pad=True)

        # Third Convolutional Layer
        self.conv3_in_channels = self.conv2_out_channels
        self.conv3_out_channels = 48
        self.conv3_kernel = 3
        self.conv3_stride = 1
        self.conv3_padding = 0
        conv3_output_dim = calculate_conv_output_dim(D=pool2_output_dim,
                                        K=self.conv3_kernel,
                                        P=self.conv3_padding,
                                        S=self.conv3_stride)
        print("Conv3 Output Dimension= " + str(conv3_output_dim))
        self.conv3_encode = nn.Conv3d(in_channels=self.conv3_in_channels, 
                                    out_channels=self.conv3_out_channels, 
                                    kernel_size=self.conv3_kernel, 
                                    stride =self.conv3_stride, 
                                    padding=self.conv3_padding)     
        nn.init.xavier_uniform_(self.conv3_encode.weight)
        self.bn3_encode = nn.BatchNorm3d(num_features = self.conv3_out_channels)
        self.leakyrelu3 = nn.LeakyReLU(leakyrelu_const, inplace=True)  

        # Third Pooling
        self.pool3_kernel = 2
        self.pool3_stride = 2 
        pool3_output_dim = calculate_pool_output_dim(D=conv3_output_dim,
                                                K=self.pool3_kernel,
                                                S=self.pool3_stride)
        print("Pool3 Output Dimension = " + str(pool3_output_dim)) 
#         self.pool3_encode = nn.MaxPool3d(kernel_size=self.pool3_kernel, 
#                                             stride=self.pool3_stride,
#                                             return_indices = self.pool_return_indices) 
        self.pool3_encode = nn.AvgPool3d(kernel_size=self.pool3_kernel, 
                                         stride=self.pool3_stride, 
                                         padding=0, 
                                         ceil_mode=False, 
                                         count_include_pad=True)

        # Fourth Convolutional Layer
        self.conv4_in_channels = self.conv3_out_channels
        self.conv4_out_channels = 64
        self.conv4_kernel = 4
        self.conv4_stride = 2
        self.conv4_padding = 0
        conv4_output_dim = calculate_conv_output_dim(D=pool3_output_dim,
                                        K=self.conv4_kernel,
                                        P=self.conv4_padding,
                                        S=self.conv4_stride)
        print("Conv4 Output Dimension= " + str(conv4_output_dim))
        self.conv4_encode = nn.Conv3d(in_channels=self.conv4_in_channels, 
                                    out_channels=self.conv4_out_channels, 
                                    kernel_size=self.conv4_kernel, 
                                    stride =self.conv4_stride, 
                                    padding=self.conv4_padding)     
        nn.init.xavier_uniform_(self.conv4_encode.weight) 
        self.bn4_encode = nn.BatchNorm3d(num_features = self.conv4_out_channels)
        self.leakyrelu4 = nn.LeakyReLU(leakyrelu_const, inplace=True)       

        # Fifth Convolutional Layer
        self.conv5_in_channels = self.conv4_out_channels
        self.conv5_out_channels = 128
        self.conv5_kernel = 3
        self.conv5_stride = 1
        self.conv5_padding = 0
        conv5_output_dim = calculate_conv_output_dim(D=conv4_output_dim,
                                        K=self.conv5_kernel,
                                        P=self.conv5_padding,
                                        S=self.conv5_stride)
        print("Conv5 Output Dimension= " + str(conv5_output_dim))
        self.conv5_encode = nn.Conv3d(in_channels=self.conv5_in_channels, 
                                    out_channels=self.conv5_out_channels, 
                                    kernel_size=self.conv5_kernel, 
                                    stride =self.conv5_stride, 
                                    padding=self.conv5_padding)     
        nn.init.xavier_uniform_(self.conv5_encode.weight) 
        self.bn5_encode = nn.BatchNorm3d(num_features = self.conv5_out_channels)
        self.leakyrelu5 = nn.LeakyReLU(leakyrelu_const, inplace=True)

        # Sixth Convolutional Layer
        self.conv6_in_channels = self.conv5_out_channels
        self.conv6_out_channels = 256
        self.conv6_kernel = 2
        self.conv6_stride = 1
        self.conv6_padding = 0
        conv6_output_dim = calculate_conv_output_dim(D=conv5_output_dim,
                                        K=self.conv6_kernel,
                                        P=self.conv6_padding,
                                        S=self.conv6_stride)
        print("Conv6 Output Dimension= " + str(conv6_output_dim))
        self.conv6_encode = nn.Conv3d(in_channels=self.conv6_in_channels, 
                                    out_channels=self.conv6_out_channels, 
                                    kernel_size=self.conv6_kernel, 
                                    stride =self.conv6_stride, 
                                    padding=self.conv6_padding)     
        nn.init.xavier_uniform_(self.conv6_encode.weight) 
        self.bn6_encode = nn.BatchNorm3d(num_features = self.conv6_out_channels)
        self.leakyrelu6 = nn.LeakyReLU(leakyrelu_const, inplace=True)

        # 7th Convolutional Layer
        self.conv7_in_channels = self.conv6_out_channels
        self.conv7_out_channels = 256
        self.conv7_kernel = 2
        self.conv7_stride = 1
        self.conv7_padding = 0
        conv7_output_dim = calculate_conv_output_dim(D=conv6_output_dim,
                                        K=self.conv7_kernel,
                                        P=self.conv7_padding,
                                        S=self.conv7_stride)
        print("Conv7 Output Dimension= " + str(conv7_output_dim))
        self.conv7_encode = nn.Conv3d(in_channels=self.conv7_in_channels, 
                                    out_channels=self.conv7_out_channels, 
                                    kernel_size=self.conv7_kernel, 
                                    stride =self.conv7_stride, 
                                    padding=self.conv7_padding)     
        nn.init.xavier_uniform_(self.conv7_encode.weight) 
        self.bn7_encode = nn.BatchNorm3d(num_features = self.conv7_out_channels)
        self.leakyrelu7 = nn.LeakyReLU(leakyrelu_const, inplace=True)
     
    #         # 1st FC Layer
#         self.fc1_in_features = self.conv7_out_channels * conv7_output_dim**3
#         self.fc1_encode = nn.Linear(in_features=self.fc1_in_features,
#                                     out_features=fc1_hidden_dim)
#         self.leakyrelu8 = nn.LeakyReLU(leakyrelu_const, inplace=True)

#         # 2nd FC Layer
#         self.fc2_encode = nn.Linear(in_features=self.fc1_hidden_dim,
#                                     out_features=embedding_dim)
#         self.relu1 = nn.ReLU(inplace=True)



    def forward(self, input):
        
        ind_list = []
        
        # Convolution Layers
#         print("Input = " +str(input.shape))
        out = self.conv1_encode(input)
#         print("conv1_encode = " + str(out.shape))
        out = self.pool1_encode(out)
#         out, ind = self.pool1_encode(out)
#         ind_list.append(ind)
#         print("pool1_encode = " + str(out.shape))
        out = self.bn1_encode(out) 
        out = self.leakyrelu1(out)

        out = self.conv2_encode(out)
#         print("conv2_encode = " + str(out.shape))
        out = self.pool2_encode(out)
#         out, ind = self.pool2_encode(out)
#         ind_list.append(ind)
#         print("pool2_encode = " + str(out.shape))
        out = self.bn2_encode(out) 
        out = self.leakyrelu2(out)

        out = self.conv3_encode(out)
#         print("conv3_encode = " + str(out.shape))
        out = self.pool3_encode(out)
#         out, ind = self.pool3_encode(out)
#         ind_list.append(ind)
#         print("pool3_encode = " + str(out.shape))
        out = self.bn3_encode(out) 
        out = self.leakyrelu3(out)

        out = self.conv4_encode(out)
#         print("conv4_encode = " + str(out.shape))
        out = self.bn4_encode(out) 
        out = self.leakyrelu4(out)

        out = self.conv5_encode(out)
#         print("conv5_encode = " + str(out.shape))
        out = self.bn5_encode(out) 
        out = self.leakyrelu5(out)

        out = self.conv6_encode(out)
#         print("conv6_encode = " + str(out.shape))
        out = self.bn6_encode(out) 
        out = self.leakyrelu6(out)
        
        out = self.conv7_encode(out)
#         print("conv7_encode = " + str(out.shape))
        out = self.bn7_encode(out) 
        out = self.leakyrelu7(out)
        
#         print("out = " + str(out.shape))

#         # Transform
#         out = out.view(self.batch_size, -1)

#         # FC Layers
#         out = self.fc1_encode(out)
#         out = self.leakyrelu8(out)

#         out = self.fc2_encode(out)
#         out = self.relu1(out)        

#         return out, ind_list
        return out

