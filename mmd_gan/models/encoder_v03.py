import torch.nn as nn
from utils.conv_utils import calculate_conv_output_dim, calculate_pool_output_dim

# input: batch_size * nc * 64 * 64
# output: batch_size * k * 1 * 1
class Encoder(nn.Module):
    def __init__(self, 
                 kernel_size = 4,
                 stride = 2,
                 padding = 1,
                 full_conv_limit = 4,
                 just_conv_limit = 1,
                 cube_edge = 128,
                 ch_mult = 2,
                 conv_bias = False,
                 leakyrelu_const = 0.01):        
        super(Encoder, self).__init__()
        print("Encoder = encoder_v02.py")
        
        """
        input: batch_size * channels * cube_edge * cube_edge * cube_edge
        output: batch_size * (channel_multiplier * channels)
        
        BatchNorm is also added.
        
        Conv3D arguments=in_channels,out_channels,kernel_size,stride,padding
        """
       
        """
        Convolutional Layers
        """
        self.embed_cube_edge = cube_edge
        
        conv_net = nn.Sequential()
        channels = 1
        layer = 1
        
        """
        Convolutions with BatchNorm
        """
        while layer <= full_conv_limit:
            conv_net.add_module("Conv_{0}".format(layer), 
                            nn.Conv3d(channels, 
                                      channels * ch_mult,
                                      kernel_size = kernel_size,
                                      stride = stride,
                                      padding = padding, 
                                      bias = conv_bias))
            conv_net.add_module("BatchNorm_{0}".format(layer), 
                            nn.BatchNorm3d(channels * ch_mult)) 
            conv_net.add_module("leakyrelu_{0}".format(layer), 
                            nn.LeakyReLU(leakyrelu_const, inplace = True)) 
            self.embed_cube_edge = calculate_conv_output_dim(D = self.embed_cube_edge,
                                                               K = kernel_size,
                                                               P = padding,
                                                               S = stride)
            channels = channels * ch_mult
            layer = layer + 1
        
        """
        Convolutions without BatchNorm
        """ 
        while layer <= just_conv_limit:
            conv_net.add_module("Conv_{0}".format(layer), 
                            nn.Conv3d(channels, 
                                      channels * ch_mult,
                                      kernel_size = kernel_size,
                                      stride = stride,
                                      padding = padding, 
                                      bias = conv_bias))
            channels = channels * ch_mult
            layer = layer + 1
        
#         conv_net.add_module("Conv_{0}".format(layer), 
#                         nn.Conv3d(channels, 
#                                   channels * ch_mult,
#                                   kernel_size = 4,
#                                   stride = 2,
#                                   padding = 1, 
#                                   bias = conv_bias))
#         channels = channels * ch_mult
        
        
        self.conv_net = conv_net
        self.channels = channels
        self.embed_cube_edge = calculate_conv_output_dim(D = self.embed_cube_edge,
                                           K = kernel_size,
                                           P = padding,
                                           S = stride)
        


    def forward(self, input):
        
        ind_list = []
        
        """
        Convolutional Layers
        """
        out = self.conv_net(input)
#         print("Encoder out shape = " + str(out.shape))
        
        
        """
        Transform
        """
#         out = out.view(batch_size, -1)

        """
        FC Layers
        """   

        return out

