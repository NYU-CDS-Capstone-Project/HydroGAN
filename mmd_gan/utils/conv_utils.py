

def calculate_conv_output_dim(D,K,P,S):

    out_dim = (D-K+2*P)/S + 1
    return int(out_dim)

def calculate_pool_output_dim(D,K,S):

    out_dim = (D-K)/S + 1
    return out_dim

def calculate_deconv_output_dim(D,K,P,S):

    out_dim = S * (D-1) - 2*P + K
    return int(out_dim)

def calculate_unpool_output_dim(D,K,P,S):
    """
    Source: https://pytorch.org/docs/stable/nn.html # MaxUnpool1d
    """

    out_dim = S * (D-1) - 2*P + K 
    return out_dim