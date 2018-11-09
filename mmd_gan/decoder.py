

# input: batch_size * k * 1 * 1
# output: batch_size * nc * image_size * image_size
class Decoder(nn.Module):
    def __init__(self, cube_dimension,fc1_hidden_dim, fc2_output_dim, 
                embedding_dim, leakyrelu_const, batch_size):
        super(Decoder, self).__init__()

#         # 1st FC Layer
#         self.embedding_dim = embedding_dim
#         self.fc1_in_features = self.embedding_dim 
#         self.fc1_hidden_dim = fc1_hidden_dim
#         self.fc1_decode = nn.Linear(in_features=self.fc1_in_features,
#                                     out_features=self.fc1_hidden_dim)
#         self.leakyrelu1 = nn.LeakyReLU(leakyrelu_const, inplace=True)

#         # 2nd FC Layer
#         self.fc2_output_dim = fc2_output_dim
#         self.fc2_decode = nn.Linear(in_features=self.fc1_hidden_dim,
#                                     out_features=self.fc2_output_dim )
#         self.leakyrelu2 = nn.LeakyReLU(leakyrelu_const, inplace=True)

        # 1st Deconvolutional Layer
        self.deconv1_in_channels = 128
        self.deconv1_out_channels = 128
        self.deconv1_kernel = 2
        self.deconv1_stride = 1
        self.deconv1_padding = 0
        deconv1_output_dim = calculate_deconv_output_dim(D=cube_dimension,
                                        K=self.deconv1_kernel,
                                        P=self.deconv1_padding,
                                        S=self.deconv1_stride)
        print("Deconv1 Output Dimension = " + str(deconv1_output_dim))
        self.deconv1_decode = nn.Conv3d(in_channels=self.deconv1_in_channels, 
                                    out_channels=self.deconv1_out_channels, 
                                    kernel_size=self.deconv1_kernel, 
                                    stride =self.deconv1_stride, 
                                    padding=self.deconv1_padding)     
        nn.init.xavier_uniform_(self.deconv1_encode.weight)
        self.bn1_decode = nn.BatchNorm3d(num_features = self.deconv1_out_channels)
        self.leakyrelu1 = nn.LeakyReLU(leakyrelu_const, inplace=True)

        # 2nd Deconvolutional Layer
        self.deconv2_in_channels = 128
        self.deconv2_out_channels = 64
        self.deconv2_kernel = 2
        self.deconv2_stride = 1
        self.deconv2_padding = 0
        deconv2_output_dim = calculate_deconv_output_dim(D=deconv1_output_dim,
                                        K=self.conv1_kernel,
                                        P=self.conv1_padding,
                                        S=self.conv1_stride)
        print("Deconv1 Output Dimension = " + str(deconv1_output_dim))
        self.deconv1_decode = nn.ConvTranspose3d(in_channels=self.deconv1_in_channels, 
                                    out_channels=self.deconv1_out_channels, 
                                    kernel_size=self.deconv1_kernel, 
                                    stride =self.deconv1_stride, 
                                    padding=self.deconv1_padding)     
        nn.init.xavier_uniform_(self.deconv1_encode.weight)
        self.bn1_decode = nn.BatchNorm3d(num_features = self.deconv1_out_channels)
        self.leakyrelu1 = nn.LeakyReLU(leakyrelu_const, inplace=True)


    def forward(self, input):

        out = out.view(batch_size, nz, 1,1,1)
        
        return output
