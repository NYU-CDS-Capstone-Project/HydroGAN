import torch.nn as nn
from torch.autograd import Variable, grad
import torch
from utils.mmd_utils import *


def grad_norm(m, norm_type=2):
    total_norm = 0.0
    for p in m.parameters():
#         try:
#             print("p = " + str(p))
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm ** norm_type
#         except:
#             pass
    total_norm = total_norm ** (1. / norm_type)
#     print("Gradient norm  = " + str(total_norm.item()))
    
    return total_norm

def weights_init(m, init_type = "normal"):
    """
    Arguments:
    init_type = either "normal" or "xavier"
    
    
    Implementation Details:
    Be sure to name the layers accordingly!
    https://pytorch.org/docs/stable/_modules/torch/nn/init.html
    normal_ = mean: the mean of the normal distribution
              std: the standard deviation of the normal distribution
    """
    
    if init_type == "normal":
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.1)
    #         m.bias.data.fill_(0)
    elif init_type == "xavier":
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.xavier_uniform_(m.weight.data, gain = 1)
#             m.weight.data.xavier_uniform_(gain = 1)
        elif classname.find('BatchNorm') != -1:
            nn.init.xavier_uniform_(m.weight.data, gain = 1)
#             m.weight.data.xavier_uniform_(gain = 1)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            nn.init.xavier_uniform_(m.weight.data, gain = 1)
#             m.weight.data.xavier_uniform_(gain = 1)
    #         m.bias.data.fill_(0)
    
    
##https://github.com/EmilienDupont/wgan-gp/blob/master/training.py
def calc_gradient_penalty(real_data, 
                          generated_data, 
                          gp_weight, 
                          netD,
                          cuda,
                          sigma_list):
    
    batch_size = real_data.size()[0]
    


    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1, 1)
    alpha = alpha.expand_as(real_data)
#     alpha = alpha.expand_as(f_enc_X_D)
    
    if torch.cuda.is_available():
        alpha = alpha.cuda()
    
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    
    interpolated = Variable(interpolated, requires_grad=True)
    if torch.cuda.is_available():
        interpolated = interpolated.cuda()
        
    f_enc_X_D, f_dec_X_D, _ = netD(real_data)
#     f_enc_Y_D, f_dec_Y_D, _ = netD(generated_data)
    f_enc_int_D, f_dec_int_D, _ = netD(interpolated)
        

    # Calculate probability of interpolated examples
    # f_enc_X_D, f_dec_X_D, f_enc_X_size = netD(x) -> thats why it returns a tuple
#     prob_interpolated = netD(interpolated)
    
#     f_enc_Y_D, f_dec_Y_D, _ = netD(interpolated)
    mmd2_D = mix_rbf_mmd2(f_enc_X_D, 
                          f_enc_int_D, 
                          sigma_list,
                          biased = True)
    
#     print("mmd2_D = " + str(mmd2_D))
#     print("len(mmd2_D) = " + str(len(mmd2_D)))
#     print("len(prob_interpolated) = " + str(len(prob_interpolated)))
#     prob_interpolated = prob_interpolated[0]
    prob_interpolated = mmd2_D

    # Calculate gradients of probabilities with respect to examples
    gradients = grad(outputs=prob_interpolated, 
                     inputs=interpolated,
                     grad_outputs=torch.ones(prob_interpolated.size()).cuda() if cuda else torch.ones(
                           prob_interpolated.size()),
                           create_graph=True, 
                     retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # Gradients have shape (batch_size,num_channels,edge_size,edge_size,edge_size),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)
    #self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data[0])

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return gp_weight * ((gradients_norm - 1) ** 2).mean()


