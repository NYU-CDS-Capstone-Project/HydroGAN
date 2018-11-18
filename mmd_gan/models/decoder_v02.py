import torch.nn as nn
from utils.conv_utils import calculate_deconv_output_dim

# input: batch_size * k * 1 * 1
# output: batch_size * nc * image_size * image_size
class Decoder(nn.Module):
    def __init__(self, 
                 embedded_cube_edge,
                 ch_mult,
                 deconv_bias,
                 fc1_hidden_dim, 
                 fc2_output_dim, 
                 embedding_channel, 
                 leakyrelu_const,
                full_deconv_limit):
        super(Decoder, self).__init__()
        
        """
        input: batch_size * embedding_channel * embedded_cube_edge * embedded_cube_edge * embedded_cube_edge
        output: batch_size * 1 * cube_edge * cube_edge * cube_edge
        
        BatchNorm is also added.
        
        ConvTranspose3D arguments=in_channels,out_channels,kernel_size,stride,padding
        """
        
      
        """
        Convolutional Layers
        """
        deconv_net = nn.Sequential()
        channels = embedding_channel
        layer = 1
        while layer <= full_deconv_limit:
            deconv_net.add_module("DeConv_{0}".format(layer), 
                            nn.ConvTranspose3d(channels, 
                                              channels // ch_mult,
                                              kernel_size = 4,
                                              stride = 2,
                                              padding = 1, 
                                              bias = deconv_bias))
            deconv_net.add_module("BatchNorm_{0}".format(layer), 
                            nn.BatchNorm3d(channels // ch_mult)) 
            deconv_net.add_module("leakyrelu_{0}".format(layer), 
                            nn.LeakyReLU(leakyrelu_const, inplace = True))    
            channels = channels // ch_mult
            layer = layer + 1
        
        deconv_net.add_module("DeConv_{0}".format(layer), 
                        nn.ConvTranspose3d(channels, 
                                  channels // ch_mult,
                                  kernel_size = 4,
                                  stride = 2,
                                  padding = 1, 
                                  bias = deconv_bias))
        deconv_net.add_module("relu_{0}".format(layer), 
                            nn.ReLU(inplace = True))    
        self.deconv_net = deconv_net
        
        """
        Fully Connected Layers
        """        


    def forward(self, input):
#         print("\nDecoder - Forward Pass")

        """
        Fully Connected Layers
        """
        
        """
        Transformation
        """
#         out = out.view(batch_size, 1, embedded_cube_edge,
#                                       embedded_cube_edge,
#                                       embedded_cube_edge)
        
        
        """
        Deconvolution Layers
        """
        out = self.deconv_net(input)
#         print("deconv out shape = " + str(out.shape))
        
        return out
    
    
    
    
    
    
    
    
    
    
