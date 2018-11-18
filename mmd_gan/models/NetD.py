import torch.nn.functional as F
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

# NetD is an encoder + decoder
# input: batch_size * nc * image_size * image_size
# f_enc_X: batch_size * k * 1 * 1
# f_dec_X: batch_size * nc * image_size * image_size
class NetD(nn.Module):
    def __init__(self, encoder, decoder):
        super(NetD, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input):
        f_enc_X = self.encoder(input)
        f_enc_X_size = f_enc_X.size()
#         print("f_enc_X size = " + str(f_enc_X.size()))
#         print("f_enc_X outputted.")
        
        f_dec_X = self.decoder(f_enc_X)
#         print("f_dec_X size = " + str(f_dec_X.size()))
#         print("f_dec_X outputted.")

        f_enc_X = f_enc_X.view(input.size(0), -1)
        f_dec_X = f_dec_X.view(input.size(0), -1)
        return f_enc_X, f_dec_X, f_enc_X_size