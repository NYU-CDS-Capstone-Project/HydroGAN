import torch.nn as nn
from utils.conv_utils import calculate_deconv_output_dim

# input: batch_size * k * 1 * 1
# output: batch_size * nc * image_size * image_size
class Decoder(nn.Module):
    def __init__(self, 
                 embedded_cube_edge,
                 fc1_hidden_dim, 
                 fc2_output_dim, 
                 embedding_dim, 
                 leakyrelu_const,
                 deconv_input_channels):
        super(Decoder, self).__init__()
        
        assert deconv_input_channels % 2 == 0, "Input channels have to be divisible by 2"

        """
        Fully Connected Layers
        
        The first input dimension has to be same with 
        the output dimension of the Encoder
        """
        # 1st FC Layer
        self.fc1_decode = nn.Linear(in_features=embedding_dim,
                                    out_features=fc1_hidden_dim)
        self.leakyrelu1 = nn.LeakyReLU(leakyrelu_const, inplace=True)

        # 2nd FC Layer
        self.fc2_decode = nn.Linear(in_features=fc1_hidden_dim,
                                    out_features=fc2_output_dim)
        self.leakyrelu2 = nn.LeakyReLU(leakyrelu_const, inplace=True)
        
        # 3rd FC Layer
        """
        Output of this layer should be embedded_cube_edge**3 x deconv_input_channels
        """
        self.fc3_output_dim = embedded_cube_edge**3 * deconv_input_channels
        self.fc3_decode = nn.Linear(in_features=fc2_output_dim,
                                    out_features=self.fc3_output_dim)
        self.leakyrelu3 = nn.LeakyReLU(leakyrelu_const, inplace=True)
        
        """
        Deconvolution Layers
        """
        # 1st Deconvolutional Layer
        self.deconv1_in_channels = deconv_input_channels
        self.deconv1_out_channels = deconv_input_channels / 2
        self.deconv1_kernel = 2
        self.deconv1_stride = 1
        self.deconv1_padding = 0
        deconv1_output_dim = calculate_deconv_output_dim(D=embedded_cube_edge,
                                        K=self.deconv1_kernel,
                                        P=self.deconv1_padding,
                                        S=self.deconv1_stride)
        print("Deconv1 Output Dimension = " + str(deconv1_output_dim))
        self.deconv1_decode = nn.ConvTranspose3d(in_channels=self.deconv1_in_channels, 
                                    out_channels=self.deconv1_out_channels, 
                                    kernel_size=self.deconv1_kernel, 
                                    stride =self.deconv1_stride, 
                                    padding=self.deconv1_padding)     
        nn.init.xavier_uniform_(self.deconv1_decode.weight)
        self.bn1_decode = nn.BatchNorm3d(num_features = self.deconv1_out_channels)
        self.leakyrelu1 = nn.LeakyReLU(leakyrelu_const, inplace=True)

        # 2nd Deconvolutional Layer
        self.deconv2_in_channels = self.deconv1_out_channels
        self.deconv2_out_channels = self.deconv2_in_channels / 2
        self.deconv2_kernel = 2
        self.deconv2_stride = 1
        self.deconv2_padding = 0
        deconv2_output_dim = calculate_deconv_output_dim(D=deconv1_output_dim,
                                        K=self.deconv2_kernel,
                                        P=self.deconv2_padding,
                                        S=self.deconv2_stride)
        print("Deconv2 Output Dimension = " + str(deconv2_output_dim))
        self.deconv2_decode = nn.ConvTranspose3d(in_channels=self.deconv2_in_channels, 
                                    out_channels=self.deconv2_out_channels, 
                                    kernel_size=self.deconv2_kernel, 
                                    stride =self.deconv2_stride, 
                                    padding=self.deconv2_padding)     
        nn.init.xavier_uniform_(self.deconv2_decode.weight)
        self.bn2_decode = nn.BatchNorm3d(num_features = self.deconv2_out_channels)
        self.leakyrelu2 = nn.LeakyReLU(leakyrelu_const, inplace=True)
        
        # 3rd Deconvolutional Layer
        self.deconv3_in_channels = self.deconv2_out_channels
        self.deconv3_out_channels = self.deconv3_in_channels / 2
        self.deconv3_kernel = 3
        self.deconv3_stride = 1
        self.deconv3_padding = 0
        deconv3_output_dim = calculate_deconv_output_dim(D=deconv2_output_dim,
                                        K=self.deconv3_kernel,
                                        P=self.deconv3_padding,
                                        S=self.deconv3_stride)
        print("Deconv3 Output Dimension = " + str(deconv3_output_dim))
        self.deconv3_decode = nn.ConvTranspose3d(in_channels=self.deconv3_in_channels, 
                                    out_channels=self.deconv3_out_channels, 
                                    kernel_size=self.deconv3_kernel, 
                                    stride =self.deconv3_stride, 
                                    padding=self.deconv3_padding)     
        nn.init.xavier_uniform_(self.deconv3_decode.weight)
        self.bn3_decode = nn.BatchNorm3d(num_features = self.deconv3_out_channels)
        self.leakyrelu3 = nn.LeakyReLU(leakyrelu_const, inplace=True)
        
        # 4th Deconvolutional Layer
        self.deconv4_in_channels = self.deconv3_out_channels
        self.deconv4_out_channels = self.deconv4_in_channels / 2
        self.deconv4_kernel = 4
        self.deconv4_stride = 2
        self.deconv4_padding = 0
        deconv4_output_dim = calculate_deconv_output_dim(D=deconv3_output_dim,
                                        K=self.deconv4_kernel,
                                        P=self.deconv4_padding,
                                        S=self.deconv4_stride)
        print("Deconv4 Output Dimension = " + str(deconv4_output_dim))
        self.deconv4_decode = nn.ConvTranspose3d(in_channels=self.deconv4_in_channels, 
                                    out_channels=self.deconv4_out_channels, 
                                    kernel_size=self.deconv4_kernel, 
                                    stride =self.deconv4_stride, 
                                    padding=self.deconv4_padding)     
        nn.init.xavier_uniform_(self.deconv4_decode.weight)
        self.bn4_decode = nn.BatchNorm3d(num_features = self.deconv4_out_channels)
        self.leakyrelu4 = nn.LeakyReLU(leakyrelu_const, inplace=True)
        
        # Unpooling 1
        # Avg Unpooling 1
        # Just make 1 voxel to 8 voxels of 2-len edges
        # Implemented in forward pass
        self.unpool1_scale = 2
        self.unpool1 = nn.Upsample(scale_factor = self.unpool1_scale, mode='nearest')
        # Max Unpooling 1
#         self.unpool1 = nn.MaxUnpool3d(kernel_size = self.unpool1_scale, stride=None, padding=0)
        
        # 5th Deconvolutional Layer
        self.deconv5_in_channels = self.deconv4_out_channels
        self.deconv5_out_channels = self.deconv5_in_channels / 2
        self.deconv5_kernel = 3
        self.deconv5_stride = 1
        self.deconv5_padding = 0
        deconv5_output_dim = calculate_deconv_output_dim(D=deconv4_output_dim * self.unpool1_scale,
                                        K=self.deconv5_kernel,
                                        P=self.deconv5_padding,
                                        S=self.deconv5_stride)
        print("Deconv5 Output Dimension = " + str(deconv5_output_dim))
        self.deconv5_decode = nn.ConvTranspose3d(in_channels=self.deconv5_in_channels, 
                                    out_channels=self.deconv5_out_channels, 
                                    kernel_size=self.deconv5_kernel, 
                                    stride =self.deconv5_stride, 
                                    padding=self.deconv5_padding)     
        nn.init.xavier_uniform_(self.deconv5_decode.weight)
        self.bn5_decode = nn.BatchNorm3d(num_features = self.deconv5_out_channels)
        self.leakyrelu5 = nn.LeakyReLU(leakyrelu_const, inplace=True)
        
        # Unpooling 2
        # Avg Unpooling 2
        # Just make 1 voxel to 8 voxels of 2-len edges
        # Implemented in forward pass
        self.unpool2_scale = 2
        self.unpool2 = nn.Upsample(scale_factor = self.unpool2_scale, mode='nearest')
        # Max Unpooling 2
#         self.unpool2 = nn.MaxUnpool3d(kernel_size = self.unpool2_scale, stride=None, padding=0)
        
        
        # 6th Deconvolutional Layer
        self.deconv6_in_channels = self.deconv5_out_channels
        self.deconv6_out_channels = self.deconv6_in_channels / 2
        self.deconv6_kernel = 4
        self.deconv6_stride = 1
        self.deconv6_padding = 0
        deconv6_output_dim = calculate_deconv_output_dim(D=deconv5_output_dim * self.unpool2_scale,
                                        K=self.deconv6_kernel,
                                        P=self.deconv6_padding,
                                        S=self.deconv6_stride)
        print("Deconv6 Output Dimension = " + str(deconv6_output_dim))
        self.deconv6_decode = nn.ConvTranspose3d(in_channels=self.deconv6_in_channels, 
                                    out_channels=self.deconv6_out_channels, 
                                    kernel_size=self.deconv6_kernel, 
                                    stride =self.deconv6_stride, 
                                    padding=self.deconv6_padding)     
        nn.init.xavier_uniform_(self.deconv6_decode.weight)
        self.bn6_decode = nn.BatchNorm3d(num_features = self.deconv6_out_channels)
        self.leakyrelu6 = nn.LeakyReLU(leakyrelu_const, inplace=True)
        
        # Unpooling 3
        # Avg Unpooling 3
        # Just make 1 voxel to 8 voxels of 2-len edges
        # Implemented in forward pass
        self.unpool3_scale = 2
        self.unpool3 = nn.Upsample(scale_factor = self.unpool3_scale, mode='nearest')
        # Max Unpooling 2
#         self.unpool3 = nn.MaxUnpool3d(kernel_size = self.unpool3_scale, stride=None, padding=0)
        
        
        # 7th Deconvolutional Layer
        self.deconv7_in_channels = self.deconv6_out_channels
        self.deconv7_out_channels = self.deconv7_in_channels / 2
        self.deconv7_kernel = 3
        self.deconv7_stride = 1
        self.deconv7_padding = 0
        deconv7_output_dim = calculate_deconv_output_dim(D=deconv6_output_dim * self.unpool3_scale,
                                        K=self.deconv7_kernel,
                                        P=self.deconv7_padding,
                                        S=self.deconv7_stride)
        print("Deconv7 Output Dimension = " + str(deconv7_output_dim))
        self.deconv7_decode = nn.ConvTranspose3d(in_channels=self.deconv7_in_channels, 
                                    out_channels=self.deconv7_out_channels, 
                                    kernel_size=self.deconv7_kernel, 
                                    stride =self.deconv7_stride, 
                                    padding=self.deconv7_padding)     
        nn.init.xavier_uniform_(self.deconv7_decode.weight)
        self.bn7_decode = nn.BatchNorm3d(num_features = self.deconv7_out_channels)
#         self.leakyrelu7 = nn.LeakyReLU(leakyrelu_const, inplace=True)
        
        # For data in [0,1]
        self.relu7 = nn.ReLU(inplace=True)     
        # For data in [-1,1]
#         self.tanh7 = nn.Tanh() 
        


    def forward(self, input):
#         print("\nDecoder - Forward Pass")

        """
        Fully Connected Layers
        """
        out = self.fc1_decode(out)
        out = self.leakyrelu1(out)
        
        out = self.fc2_decode(out)
        out = self.leakyrelu2(out)

        out = self.fc3_decode(out)
        out = self.leakyrelu3(out)
        
        """
        Transformation
        """
        out = out.view(batch_size, 1, embedded_cube_edge,
                                      embedded_cube_edge,
                                      embedded_cube_edge)
        
        
        """
        Deconvolution Layers
        """
#         print("Input = " +str(input.shape))
#         out = self.deconv1_decode(input)
        out = self.deconv1_decode(out)
#         print("deconv1_decode = " + str(out.shape))
        out = self.bn1_decode(out)
        out = self.leakyrelu1(out)

        out = self.deconv2_decode(out)
#         print("deconv2_decode = " + str(out.shape))
        out = self.bn2_decode(out)
        out = self.leakyrelu2(out)
        
        out = self.deconv3_decode(out)
#         print("deconv3_decode = " + str(out.shape))
        out = self.bn3_decode(out)
        out = self.leakyrelu3(out)
        
        out = self.deconv4_decode(out)
#         print("deconv4_decode = " + str(out.shape))
        out = self.bn4_decode(out)
        out = self.leakyrelu4(out)
        out = self.unpool1(out)
#         out = self.unpool1(out,ind_list[0])
#         print("avgunpool1 = " + str(out.shape))

        out = self.deconv5_decode(out)
#         print("deconv5_decode = " + str(out.shape))
        out = self.bn5_decode(out)
        out = self.leakyrelu5(out)
        out = self.unpool2(out)
#         out = self.unpool2(out, ind_list[1])
#         print("avgunpool2 = " + str(out.shape))
        
        out = self.deconv6_decode(out)
#         print("deconv6_decode = " + str(out.shape))
        out = self.bn6_decode(out)
        out = self.leakyrelu6(out)
        out = self.unpool3(out) 
#         out = self.unpool3(out, ind_list[2])
#         print("avgunpool3 = " + str(out.shape))
        
        out = self.deconv7_decode(out)
#         print("deconv7_decode = " + str(out.shape))
        out = self.bn7_decode(out)
#         out = self.leakyrelu7(out)
        out = self.relu7(out) # for [0,1] or standardize no-shift
#         out = self.tanh7(out) # for [-1,1]
        
#         print("decoder out = " + str(out.shape))
#         # Transformation
#         out = out.view(batch_size, nz, 1,1,1)
        
        return out
    
    
    
    
    
    
    
    
    
    
