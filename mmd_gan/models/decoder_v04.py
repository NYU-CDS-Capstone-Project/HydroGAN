import torch.nn as nn
from utils.conv_utils import calculate_deconv_output_dim
import torch

# input: batch_size * k * 1 * 1
# output: batch_size * nc * image_size * image_size
class Decoder(nn.Module):
    def __init__(self, 
                 kernel_size,
                 stride,
                 padding,
                 ch_mult,
                 deconv_bias,
                 leakyrelu_const,
                full_deconv_limit,
                 just_deconv_limit,
                 D_encoder):
        super(Decoder, self).__init__()
        print("Decoder = decoder_v03.py")
        
        """
        input: batch_size * embedding_channel * embedded_cube_edge * embedded_cube_edge * embedded_cube_edge
        output: batch_size * 1 * cube_edge * cube_edge * cube_edge
        
        BatchNorm is also added.
        
        ConvTranspose3D arguments=in_channels,out_channels,kernel_size,stride,padding
        """
        
      
        """
        Deconvolutional Layers
        """
        deconv_net = nn.Sequential()
        channels = D_encoder.channels
#         embed_cube_edge = D_encoder.embed_cube_edge
        layer = 1
        
        # 1
        deconv_net, channels, layer = add_deconv_module(
                    deconv_net = deconv_net,
                    channels = channels,
                    ch_mult = ch_mult,
                    kernel_size = kernel_size,
                    stride = stride,
                    padding = padding,
                    batch_norm = True,
                    deconv_bias = False,
                    activation = "relu",
                    leakyrelu_const = False,
                    layer = layer)
        # 2
        deconv_net, channels, layer = add_deconv_module(
                    deconv_net = deconv_net,
                    channels = channels,
                    ch_mult = ch_mult,
                    kernel_size = kernel_size,
                    stride = stride,
                    padding = padding,
                    batch_norm = True,
                    deconv_bias = False,
                    activation = "relu",
                    leakyrelu_const = False,
                    layer = layer)
        # 3
        deconv_net, channels, layer = add_deconv_module(
                    deconv_net = deconv_net,
                    channels = channels,
                    ch_mult = ch_mult,
                    kernel_size = kernel_size,
                    stride = stride,
                    padding = padding,
                    batch_norm = True,
                    deconv_bias = False,
                    activation = "relu",
                    leakyrelu_const = False,
                    layer = layer)
        # 4
        deconv_net, channels, layer = add_deconv_module(
                    deconv_net = deconv_net,
                    channels = channels,
                    ch_mult = ch_mult,
                    kernel_size = kernel_size,
                    stride = stride,
                    padding = padding,
                    batch_norm = True,
                    deconv_bias = False,
                    activation = "relu",
                    leakyrelu_const = False,
                    layer = layer)
        # 5
        deconv_net, channels, layer = add_deconv_module(
                    deconv_net = deconv_net,
                    channels = channels,
                    ch_mult = ch_mult,
                    kernel_size = kernel_size,
                    stride = stride,
                    padding = padding,
                    batch_norm = False,
                    deconv_bias = False,
                    activation = "tanh",
                    leakyrelu_const = False,
                    layer = layer)
        
        
        self.deconv_net = deconv_net  
        
        


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

        """
        Output Transformation
        """
#        out = torch.exp(out)
#         print(out)
        
        return out
    
   
"""
Adding a 3D convolutional module
"""
def add_deconv_module(deconv_net,
                    channels,
                    ch_mult,
                    kernel_size,
                    stride,
                    padding,
                    batch_norm,
                    deconv_bias,
                    activation,
                    leakyrelu_const,
                    layer):
    
    # Deconvolution Module
    deconv_net.add_module("DeConv_{0}".format(layer), 
                    nn.Conv3d(channels, 
                              channels // ch_mult,
                              kernel_size = kernel_size,
                              stride = stride,
                              padding = padding, 
                              bias = deconv_bias))
    
    # Batch Norm Module
    if batch_norm == True:
        deconv_net.add_module("BatchNorm_{0}".format(layer), 
                        nn.BatchNorm3d(channels // ch_mult)) 
    
    # Activation Module
    if activation == "leakyrelu":
        deconv_net.add_module("leakyrelu_{0}".format(layer), 
                        nn.LeakyReLU(leakyrelu_const, inplace = True))
    elif activation == "relu":
        deconv_net.add_module("relu_{0}".format(layer), 
                        nn.ReLU(inplace = True)) 
    elif activation == "tanh":
        deconv_net.add_module("tanh_{0}".format(layer), 
                        nn.Tanh())
    elif activation == "sigmoid":
        deconv_net.add_module("sigmoid_{0}".format(layer), 
                            nn.Sigmoid()) 
        
        
    channels = channels // ch_mult
    layer = layer + 1
    
    return deconv_net, channels, layer
    
    
    
    
    
    
    
    
    
