import torch.nn as nn
from utils.conv_utils import calculate_conv_output_dim, calculate_pool_output_dim

# input: batch_size * nc * 64 * 64
# output: batch_size * k * 1 * 1
class Encoder(nn.Module):
    def __init__(self, 
                 input_cube_edge = 128,
                 full_conv_limit = 4,
                 full_fc_limit = 3,
                 ch_mult = 2,
                 conv_bias = False,
                 fc_bias = False,
                 leakyrelu_const = 0.01):        
        super(Encoder, self).__init__()
        """
        input: batch_size * channels * cube_edge * cube_edge * cube_edge
        output: batch_size * (channel_multiplier * channels)
        BatchNorm is also added.
        Conv3D arguments=in_channels,out_channels,kernel_size,stride,padding
        Make sure that the names of the layers are specified according to
        weight init function.
        
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
                                      channels * ch_mult,
                                      kernel_size = 4,
                                      stride = 2,
                                      padding = 1, 
                                      bias = conv_bias))
            conv_net.add_module("BatchNorm_{0}".format(layer), 
                            nn.BatchNorm3d(channels * ch_mult)) 
            conv_net.add_module("leakyrelu_{0}".format(layer), 
                            nn.LeakyReLU(leakyrelu_const, inplace = True))    
            # update channel count, layer number and convoluted cube edge size
            channels = channels * ch_mult
            layer = layer + 1
            input_cube_edge = calculate_conv_output_dim(D=input_cube_edge,
                                                        K = 4,P = 1,S = 2)
        # final conv layer
        conv_net.add_module("Conv_{0}".format(layer), 
                        nn.Conv3d(channels, 
                                  channels * ch_mult,
                                  kernel_size = 4,
                                  stride = 2,
                                  padding = 1, 
                                  bias = conv_bias))
        input_cube_edge = calculate_conv_output_dim(D=input_cube_edge,
                                                    K = 4,P = 1,S = 2)
        out_channels = channels * ch_mult
        self.conv_net = conv_net
        
        """
        Fully Connected Layers
        """
        total_fc_input = input_cube_edge**3 * out_channels
        fc_net = nn.Sequential()
        layer = 1
        while layer <= full_fc_limit:
            fc_net.add_module("Linear_{0}".format(layer), 
                                nn.Linear(in_features = total_fc_input, 
                                          out_features = total_fc_input // 2, 
                                          bias = fc_bias))
            layer = layer + 1
            total_fc_input = total_fc_input // 2
        self.fc_net = fc_net
        
        # store FC output dimension & embed cube edge size & 
        # channel count for use in Decoders
        self.fc_output_dim = total_fc_input
        self.embed_cube_edge = input_cube_edge
        self.out_channels = out_channels
        


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
        batch_size = out.shape[0]
#         print("batch_size = " + str(batch_size))
        out = out.view(batch_size, -1)

        """
        FC Layers
        """ 
        out = self.fc_net(out)

        return out

