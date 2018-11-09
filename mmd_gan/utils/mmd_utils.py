# Consider linear time MMD with a linear kernel:
# K(f(x), f(y)) = f(x)^Tf(y)
# h(z_i, z_j) = k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(x_j, y_i)
#             = [f(x_i) - f(y_i)]^T[f(x_j) - f(y_j)]
#
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def linear_mmd2(f_of_X, f_of_Y):
    loss = 0.0
    delta = f_of_X - f_of_Y
    loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
    return loss

# Consider linear time MMD with a polynomial kernel:
# K(f(x), f(y)) = (alpha*f(x)^Tf(y) + c)^d
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def poly_mmd2(f_of_X, f_of_Y, d=2, alpha=1.0, c=2.0):
    K_XX = (alpha * (f_of_X[:-1] * f_of_X[1:]).sum(1) + c)
    K_XX_mean = torch.mean(K_XX.pow(d))

    K_YY = (alpha * (f_of_Y[:-1] * f_of_Y[1:]).sum(1) + c)
    K_YY_mean = torch.mean(K_YY.pow(d))

    K_XY = (alpha * (f_of_X[:-1] * f_of_Y[1:]).sum(1) + c)
    K_XY_mean = torch.mean(K_XY.pow(d))

    K_YX = (alpha * (f_of_Y[:-1] * f_of_X[1:]).sum(1) + c)
    K_YX_mean = torch.mean(K_YX.pow(d))

    return K_XX_mean + K_YY_mean - K_XY_mean - K_YX_mean

def _mix_rbf_kernel(X, Y, sigma_list):
    """
    Inputs:
        X -> f_enc_X_D ->
            size = batch_size x nz 
                nz = hidden dimension of z
        Y -> f_enc_Y_D -> 
            size = batch_size x nz 
                nz = hidden dimension of z
        sigma_list -> 
            base = 1.0
            sigma_list = [1, 2, 4, 8, 16]
            sigma_list = [sigma / base for sigma in sigma_list] 
            
    m = batch_size
    torch.cat(seq, dim=0, out=None) → Tensor
        Concatenates the given sequence of seq tensors 
        in the given dimension
    Z size = [2 x batch_size, nz]
    
    torch.mm(mat1, mat2, out=None) → Tensor
        Performs a matrix multiplication of the matrices mat1 and mat2
    ZZT size = [2 x batch_size, 2 x batch_size]
    
    torch.diag(input, diagonal=0, out=None) → Tensor
        If input is a matrix (2-D tensor), then returns a 1-D tensor 
        with the diagonal elements of input
    torch.unsqueeze(input, dim, out=None) → Tensor
        Returns a new tensor with a dimension of size 
        one inserted at the specified position
    diag_ZZT = [2 x batch_size, 1]
    
    expand_as(other) → Tensor
        Expand this tensor to the same size as other
    Z_norm_sqr = [2 x batch_size, 2 x batch_size]
    
    torch.exp(tensor, out=None) → Tensor
        Returns a new tensor with the exponential of the elements of input
        y_i = e^(x_i)
        
    exponent size = [2 x batch_size, 2 x batch_size]
    K size = [2 x batch_size, 2 x batch_size]
    """
    
    assert(X.size(0) == Y.size(0))
    m = X.size(0)

    Z = torch.cat((X, Y), dim = 0)
#     print("Z size = " + str(Z.size()))
    
    ZZT = torch.mm(Z, Z.t())
#     print("ZZT size = " + str(ZZT.size()))
    
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
#     print("diag_ZZT size = " + str(diag_ZZT.size()))
    
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
#     print("Z_norm_sqr size = " + str(Z_norm_sqr.size()))
    
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()
#     print("exponent size = " + str(exponent.size()))
#     print("exponent = " + str(exponent))

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        K += torch.exp(-gamma * exponent)
#     print("K size = " + str(K.size()))

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)


def mix_rbf_mmd2(X, Y, sigma_list, biased=True):
    """
    How it is used in the training loop:
        mmd2_D = mix_rbf_mmd2(f_enc_X_D,  f_enc_Y_D,  sigma_list)
        X -> f_enc_X_D ->
            size = batch_size x nz 
                nz = hidden dimension of z
        Y -> f_enc_Y_D -> 
            size = batch_size x nz 
                nz = hidden dimension of z
        sigma_list -> 
            base = 1.0
            sigma_list = [1, 2, 4, 8, 16]
            sigma_list = [sigma / base for sigma in sigma_list]
        
    _mix_rbf_kernel's internal K has [2 x batch_size, 2 x batch_size] size
    K_XX = K[:m, :m] (left upper quadrant) -> size = [batch_size, batch_size]
    K_XY = K[:m, m:] (right upper and left lower quadrant) -> size = [batch_size, batch_size]
    K_YY = K[m:, m:] (right lower quadrant) -> size = [batch_size, batch_size]
    d = len(sigma_list)
        
    """
    
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
#     print("K_XX size = " + str(K_XX.size()))
#     print("K_XY size = " + str(K_XY.size()))
#     print("K_YY size = " + str(K_YY.size()))
    # return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


def mix_rbf_mmd2_and_ratio(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    """
    Inputs:
        K_XX = K[:m, :m] size = [batch_size, batch_size]
        K_XY = K[:m, m:] size = [batch_size, batch_size]
        K_YY = K[m:, m:] size = [batch_size, batch_size]
        
    m = batch_size

    """
    
    m = K_XX.size(0)    # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)                       # (m,)
        diag_Y = torch.diag(K_YY)                       # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e

    Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
            + Kt_YY_sum / (m * (m - 1))
            - 2.0 * K_XY_sum / (m * m))

    return mmd2



def _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    mmd2, var_est = _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=const_diagonal, biased=biased)
    loss = mmd2 / torch.sqrt(torch.clamp(var_est, min=min_var_est))
    return loss, mmd2, var_est



def _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)    # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
        sum_diag2_X = sum_diag2_Y = m * const_diagonal**2
    else:
        diag_X = torch.diag(K_XX)                       # (m,)
        diag_Y = torch.diag(K_YY)                       # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)
        sum_diag2_X = diag_X.dot(diag_X)
        sum_diag2_Y = diag_Y.dot(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e
    K_XY_sums_1 = K_XY.sum(dim=1)                     # K_{XY} * e

    Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

    Kt_XX_2_sum = (K_XX ** 2).sum() - sum_diag2_X      # \| \tilde{K}_XX \|_F^2
    Kt_YY_2_sum = (K_YY ** 2).sum() - sum_diag2_Y      # \| \tilde{K}_YY \|_F^2
    K_XY_2_sum  = (K_XY ** 2).sum()                    # \| K_{XY} \|_F^2

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
            + Kt_YY_sum / (m * (m - 1))
            - 2.0 * K_XY_sum / (m * m))

    var_est = (
        2.0 / (m**2 * (m - 1.0)**2) * (2 * Kt_XX_sums.dot(Kt_XX_sums) - Kt_XX_2_sum + 2 * Kt_YY_sums.dot(Kt_YY_sums) - Kt_YY_2_sum)
        - (4.0*m - 6.0) / (m**3 * (m - 1.0)**3) * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 4.0*(m - 2.0) / (m**3 * (m - 1.0)**2) * (K_XY_sums_1.dot(K_XY_sums_1) + K_XY_sums_0.dot(K_XY_sums_0))
        - 4.0*(m - 3.0) / (m**3 * (m - 1.0)**2) * (K_XY_2_sum) - (8 * m - 12) / (m**5 * (m - 1)) * K_XY_sum**2
        + 8.0 / (m**3 * (m - 1.0)) * (
            1.0 / m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
            - Kt_XX_sums.dot(K_XY_sums_1)
            - Kt_YY_sums.dot(K_XY_sums_0))
        )
    return mmd2, var_est



def normalize(x, dim=1):
    """
    used only in match() when dist == cos
    """
    return x.div(x.norm(2, dim=dim).expand_as(x))


def match(x, y, dist):
    '''
    Computes distance between corresponding points points in `x` and `y`
    using distance `dist`.
    
    # compute L2-loss of AE
    L2_AE_X_D = match(x.view(batch_size, -1), f_dec_X_D, 'L2')
    L2_AE_Y_D = match(y.view(batch_size, -1), f_dec_Y_D, 'L2')
    
    '''
    if dist == 'L2':
        return (x - y).pow(2).mean()
    elif dist == 'L1':
        return (x - y).abs().mean()
    elif dist == 'cos':
        x_n = normalize(x)
        y_n = normalize(y)
        return 2 - (x_n).mul(y_n).mean()
    else:
        assert dist == 'none', 'wtf ?'
        
        
        
