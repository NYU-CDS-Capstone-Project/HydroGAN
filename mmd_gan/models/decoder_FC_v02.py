import torch.nn as nn
from utils.conv_utils import calculate_deconv_output_dim


# input: batch_size * k * 1 * 1
# output: batch_size * nc * image_size * image_size
class Decoder(nn.Module):
    def __init__(self, 
                 embedded_cube_edge,
                 ch_mult,
                 deconv_bias,
                 fc_bias,
                 embedding_channel, 
                 leakyrelu_const,
                full_deconv_limit,
                 full_fc_limit,
                  D_encoder):
        super(Decoder, self).__init__()
        
        """
        input: batch_size * embedding_channel * embedded_cube_edge * embedded_cube_edge * embedded_cube_edge
        output: batch_size * 1 * cube_edge * cube_edge * cube_edge
        
        BatchNorm is also added.
        
        ConvTranspose3D arguments=in_channels,out_channels,kernel_size,stride,padding
        """
        
        """
        Fully Connected Layers
        """
        total_fc_input = D_encoder.fc_output_dim

        fc_net = nn.Sequential()
        layer = 1
        while layer <= full_fc_limit:
            fc_net.add_module("Linear_{0}".format(layer), 
                                nn.Linear(in_features = total_fc_input, 
                                          out_features = total_fc_input * 2, 
                                          bias = fc_bias))
            layer = layer + 1
            total_fc_input = total_fc_input * 2
        self.fc_net = fc_net
        
        # store output dimension for later use
        self.fc_output_dim = total_fc_input
        
        
      
        """
        Convolutional Layers
        """
        
        self.channels = D_encoder.out_channels
        channels = D_encoder.out_channels
        self.embed_cube_edge = D_encoder.embed_cube_edge
        
        deconv_net = nn.Sequential()
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


    def forward(self, input):
#         print("\nDecoder - Forward Pass")

        """
        Fully Connected Layers
        """
#         print("decoder fc_net input = " + str(input.size()))
        out = self.fc_net(input)
#         print("decoder fc_net out = " + str(out.size()))
        
        """
        Transformation
        """
        batch_size = out.shape[0]
#         print("batch_size = " + str(batch_size))
#         print("self.embed_cube_edge = " + str(self.embed_cube_edge))
        out = out.view(batch_size, self.channels, self.embed_cube_edge,
                                                  self.embed_cube_edge,
                                                  self.embed_cube_edge)
        
        
        """
        Deconvolution Layers
        """
        out = self.deconv_net(out)
#         print("deconv out shape = " + str(out.shape))
        
        return out
    
    
    
    
    
    
    
    
    
    
