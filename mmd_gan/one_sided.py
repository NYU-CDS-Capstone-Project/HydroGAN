class ONE_SIDED(nn.Module):
    """
    rank hinge loss
    one_side_errD = one_sided(f_enc_X_D.mean(0) - f_enc_Y_D.mean(0))
        always 0!
    one_side_errG = one_sided(f_enc_X.mean(0) - f_enc_Y.mean(0))
        always 0!
    
    torch.mean(input, dim, keepdim=False, out=None) → Tensor
        Returns the mean value of each row of the input tensor 
        in the given dimension dim
        0 = dim -> rows
        
    
    
    """
    def __init__(self):
        super(ONE_SIDED, self).__init__()

        main = nn.ReLU()
        self.main = main

    def forward(self, input):
        output = self.main(-input)
        output = -output.mean()
        return output