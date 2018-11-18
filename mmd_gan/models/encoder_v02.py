import torch.nn as nn
from utils.conv_utils import calculate_conv_output_dim, calculate_pool_output_dim

# input: batch_size * nc * 64 * 64
# output: batch_size * k * 1 * 1
class Encoder(nn.Module):
    def __init__(self, 
                 full_conv_limit = 4,
                 ch_mult = 2,
                 conv_bias = False,
                 fc1_hidden_dim = False, 
                 fc2_output_dim = False, 
                 leakyrelu_const = 0.01):        
        super(Encoder, self).__init__()
        """
        input: batch_size * channels * cube_edge * cube_edge * cube_edge
        output: batch_size * (channel_multiplier * channels)
        
        BatchNorm is also added.
        
        Conv3D arguments=in_channels,out_channels,kernel_size,stride,padding
        """
       
        """
        Convolutional Layers
        """
        conv_net = nn.Sequential()
        channels = 1
        layer = 1
        while layer <= full_conv_limit:
            conv_net.add_module("Conv_{0}".format(layer), 
                            nn.Conv3d(channels, 
                                      channels*ch_mult,
                                      kernel_size = 4,
                                      stride = 2,
                                      padding = 1, 
                                      bias = conv_bias))
            conv_net.add_module("BatchNorm_{0}".format(layer), 
                            nn.BatchNorm3d(channels*2)) 
            conv_net.add_module("leakyrelu_{0}".format(layer), 
                            nn.LeakyReLU(leakyrelu_const, inplace = True))    
            channels = channels * 2
            layer = layer + 1
        
        conv_net.add_module("Conv_{0}".format(layer), 
                        nn.Conv3d(channels, 
                                  channels*ch_mult,
                                  kernel_size = 4,
                                  stride = 2,
                                  padding = 1, 
                                  bias = conv_bias))
        self.conv_net = conv_net
        
        """
        Fully Connected Layers
        """
        


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

