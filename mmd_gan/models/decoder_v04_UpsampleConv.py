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
        UpsampleConv Layers
        """
        deconv_net = nn.Sequential()
        channels = D_encoder.channels
#         embed_cube_edge = D_encoder.embed_cube_edge
        layer = 1
        
        # 1
        deconv_net, channels, layer = add_upsampleconv_module(deconv_net,
                    channels = channels,
                    ch_mult = ch_mult,
                    kernel_size = 3,
                    stride = 1,
                    conv_padding = 1,
                    batch_norm = True,
                    deconv_bias = False,
                    activation = "relu",
                    leakyrelu_const = False,
                    layer = layer,
                    scale_factor = 2,
                    mode = "nearest",
                    reflection_padding = 1)
        # 2
        deconv_net, channels, layer = add_upsampleconv_module(deconv_net,
                    channels = channels,
                    ch_mult = ch_mult,
                    kernel_size = 3,
                    stride = 1,
                    conv_padding = 1,
                    batch_norm = True,
                    deconv_bias = False,
                    activation = "relu",
                    leakyrelu_const = False,
                    layer = layer,
                    scale_factor = 2,
                    mode = "nearest",
                    reflection_padding = 1)
        # 3
        deconv_net, channels, layer = add_upsampleconv_module(deconv_net,
                    channels = channels,
                    ch_mult = ch_mult,
                    kernel_size = 3,
                    stride = 1,
                    conv_padding = 1,
                    batch_norm = True,
                    deconv_bias = False,
                    activation = "relu",
                    leakyrelu_const = False,
                    layer = layer,
                    scale_factor = 2,
                    mode = "nearest",
                    reflection_padding = 1)
        # 4
        deconv_net, channels, layer = add_upsampleconv_module(deconv_net,
                    channels = channels,
                    ch_mult = ch_mult,
                    kernel_size = 3,
                    stride = 1,
                    conv_padding = 1,
                    batch_norm = True,
                    deconv_bias = False,
                    activation = "relu",
                    leakyrelu_const = False,
                    layer = layer,
                    scale_factor = 2,
                    mode = "nearest",
                    reflection_padding = 1)
        # 5
        deconv_net, channels, layer = add_upsampleconv_module(deconv_net,
                    channels = channels,
                    ch_mult = ch_mult,
                    kernel_size = 3,
                    stride = 1,
                    conv_padding = 1,
                    batch_norm = False,
                    deconv_bias = False,
                    activation = "sigmoid",
                    leakyrelu_const = False,
                    layer = layer,
                    scale_factor = 2,
                    mode = "nearest",
                    reflection_padding = 0)
        
        
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
Adding a 3D Deconvolutional module
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
                    nn.ConvTranspose3d(channels, 
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
    
    
"""
Adding a 3D Upsample-Conv module
This should alleviate the checkerboard problem.
"""
class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, 
                        scale_factor=self.scale_factor, 
                        mode=self.mode)
        return x   
    

class UpsampleChecker(torch.nn.Module):
    def __init__(self, 
                 scale_factor, 
                 mode,
                 reflection_padding,
                 channels,
                 ch_mult,
                 kernel_size,
                 conv_padding,
                 stride,
                 deconv_bias):
        """
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/190
          nn.Upsample(scale_factor = 2, mode='bilinear'),
          nn.ReflectionPad2d(1),
          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                             kernel_size=3, stride=1, padding=0)
                             
        nn.Upsample
        mode (string, optional) – the upsampling algorithm: 
        one of nearest, linear, bilinear and trilinear. Default: nearest
        
        For this module to preserve the dimensions of the input cube,
        the kernel = 3, stride = 1 and padding = 0 for the conv
        the padding = 1 for reflection padding
        """
        super(UpsampleChecker, self).__init__()
        self.upsample = nn.Upsample(scale_factor = scale_factor, 
                                    mode = mode)
        self.reflection_pad = reflection_padding
        self.conv = nn.Conv3d(channels, 
                              channels // ch_mult,
                              kernel_size = kernel_size,
                              stride = stride,
                              padding = conv_padding, 
                              bias = deconv_bias)

    def forward(self, x):
        """
        pad (tuple) – m-elem tuple, where m/2 ≤ input dimensions and m is even
        mode – ‘constant’, ‘reflect’ or ‘replicate’. 
        value – fill value for ‘constant’ padding.
        """
        print("upsample")
        print("out size = " + str(x.size()))
        out = self.upsample(x)
        
#         print("padded")
#         print("out size = " + str(out.size()))
#         out = nn.functional.pad(input = out, 
# #                                 pad = self.reflection_pad, 
#                                 pad = (1,1,1,1,1,1),
#                                 mode='constant', 
#                                 value = 0)
        
        
        print("conv")
        print("out size = " + str(out.size()))
        out = self.conv(out)
        
        print("out size = " + str(out.size()))
        return out


    
    
def add_upsampleconv_module(deconv_net,
                    channels,
                    ch_mult,
                    kernel_size,
                    stride,
                    conv_padding,
                    batch_norm,
                    deconv_bias,
                    activation,
                    leakyrelu_const,
                    layer,
                    scale_factor,
                    mode,
                    reflection_padding):
    
#     deconv_net.add_module("UpsampleChecker_{0}".format(layer), 
#                            UpsampleChecker(scale_factor = scale_factor, 
#                                              mode = mode,
#                                              reflection_padding = reflection_padding,
#                                              channels = channels,
#                                              ch_mult = ch_mult,
#                                              kernel_size = kernel_size,
#                                              conv_padding = conv_padding,
#                                              stride = stride,
#                                              deconv_bias = deconv_bias))

    deconv_net.add_module("InterpolateUpsample_{0}".format(layer),
                          Interpolate(scale_factor = scale_factor, 
                                      mode = mode))
                          
    
    deconv_net.add_module("Conv_{0}".format(layer),
                          nn.Conv3d(channels, 
                              channels // ch_mult,
                              kernel_size = kernel_size,
                              stride = stride,
                              padding = conv_padding, 
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
    
    

    
    
    
