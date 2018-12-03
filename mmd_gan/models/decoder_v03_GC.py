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
        print("Decoder = decoder_v02.py")
        
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
        embed_cube_edge = D_encoder.embed_cube_edge
        layer = 1
        
        """
        Deconvolutional Layers with BatchNorm
        """        
        while layer <= full_deconv_limit:
            deconv_net.add_module("DeConv_{0}".format(layer), 
                            nn.ConvTranspose3d(channels, 
                                              channels // ch_mult,
                                              kernel_size = kernel_size,
                                              stride = stride,
                                              padding = padding, 
                                              bias = deconv_bias))
            deconv_net.add_module("BatchNorm_{0}".format(layer), 
                            nn.BatchNorm3d(channels // ch_mult)) 
            deconv_net.add_module("leakyrelu_{0}".format(layer), 
                            nn.LeakyReLU(leakyrelu_const, inplace = True))    
            channels = channels // ch_mult
            layer = layer + 1

        """
        Deconvolutional Layers without BatchNorm
        """        
        while layer <= just_deconv_limit:
            deconv_net.add_module("DeConv_{0}".format(layer), 
                            nn.ConvTranspose3d(channels, 
                                      channels // ch_mult,
                                      kernel_size = kernel_size,
                                      stride = stride,
                                      padding = padding, 
                                      bias = deconv_bias))
            channels = channels // ch_mult
            layer = layer + 1
        
#         deconv_net.add_module("DeConv_{0}".format(layer), 
#                         nn.ConvTranspose3d(channels, 
#                                   channels // ch_mult,
#                                   kernel_size = 4,
#                                   stride = 2,
#                                   padding = 1, 
#                                   bias = deconv_bias))
#         channels = channels // ch_mult
        
        
#         deconv_net.add_module("relu_{0}".format(layer), 
#                             nn.ReLU(inplace = True))
        print("Output = torch.exp(out)")
        
        
        self.deconv_net = deconv_net  
        
        


    def forward(self, input):
#         print("\nDecoder - Forward Pass")
        ngpu = torch.cuda.device_count()

        if isinstance(input.data, torch.cuda.FloatTensor) and ngpu > 1:
            #print("Multiple GPU forward Decoder pass")
            #print("deconv net")
            out = nn.parallel.data_parallel(self.deconv_net, input, range(ngpu))
            #print("torch.exp")
            #out = nn.parallel.data_parallel(torch.exp, out, range(ngpu))
#            
        else: 
            #print("single GPU forward decoder pass")
            out = self.deconv_net(input)
            #out = torch.exp(out)

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
#        out = self.deconv_net(input)
#         print("deconv out shape = " + str(out.shape))

        """
        Output Transformation
        """
        out = torch.exp(out)
#         print(out)
        
        return out
    
    
    
    
    
    
    
    
    
    
