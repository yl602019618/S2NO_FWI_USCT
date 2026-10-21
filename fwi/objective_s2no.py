"""Original source-estimated objective and reciprocal-field adjoint gradient.

Plotting/timing side effects have been removed; numerical calculations are retained.
"""

import torch
import numpy as np
import torch.nn.functional as F
import math
from .source_s2no import source_estimation

def cal_fcost_grad(x_sp,dobs,param,fscale,gscale,u_allsource,model,batchsize, sigma_grad = 2,name = None, save_file = None):
    '''
    x_sp: input square slownes, 230400
    x_ro: input intensity? 230400 1000*torch.ones(230400)
    dobs: observation 256,256, complex
    param
    fscale: 1
    gscale:1
    verb:1
    u_allsource: homo_field 256,480,480
    model:FNO model
    '''
    print('cal grad '+str(name))
    J = 0
    G = torch.zeros(param['NX']*param['NY'],1).cuda()
    # irow = np.arange(param['n_rec'])
    # icol = param['i_rec']-1#[irow - 1]  # MATLAB索引从1开始，Python索引从0开始
    # Rval=torch.ones(np.size(irow))
    # R = coo_matrix((Rval, (irow , icol )), shape=(param['n_rec'],param['NYX']))
    # R = R.tocsr() # 256,230400
    transducer_irow = np.squeeze(param['x_rec']).astype(int)-1
    transducer_icol = np.squeeze(param['y_rec']).astype(int)-1
    # Checked  R(i, param['i_rec'][i]-1)

    vp = 1/x_sp.detach().cpu().numpy().reshape(param['NX'],param['NY'],order = 'F') # return to 480,480
    omg = 2* np.pi* param['freq'] #10pi
    # select_mat = param['select_mat'] # 256,256
    # Checked
    # select_mat[abs(dobs)==0] = 0

    u = torch.zeros(param['NX'],param['NY'],param['n_src'],dtype = torch.complex64).cuda() # 480,480,256
    # ajs = torch.zeros(1,1,param['n_src'],dtype = torch.complex64).cuda()    # 1,1,256
    sval_all = torch.zeros(param['n_src'],dtype = torch.complex64).cuda()  # 256,1

    module1 = model
    u, sval_all, dsim, dobs = source_estimation(dobs, param, vp, transducer_irow, transducer_icol, u_allsource, module1, batchsize) #u:(256,480,480) sval_all(256,) dsim,dobs:(src, rec)

    # calculate the forward wavefield
    u_forward = u * sval_all.view(-1, 1, 1) # src, 480,480

    # calculate the adjoint wavefield
    residual = dsim * sval_all.view(-1, 1) - dobs # src, rec
    u_adjoint = torch.matmul(residual.conj(), u.view(param['n_src'], -1)) # src, 480*480
    u_adjoint = u_adjoint.view(param['n_src'], 480, 480)

    vp = torch.tensor(vp).cuda()
    # calculate the gradient
    grad_term = 2 * omg**2 / vp * torch.real(u_forward * u_adjoint)
    grad = -torch.sum(grad_term, dim=0)

    # calculate the objective function
    obj = torch.sum(torch.norm(residual, dim=1)**2)

    G = grad.permute(1,0).reshape(-1,1)
    J = obj

    G = G.reshape(param['NY'],param['NX']).permute(1,0)/1e10

    def create_gaussian_kernel(kernel_size=3, sigma=1.0):
        x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
        x = torch.exp(-x**2 / (2 * sigma**2))
        kernel_1d = x / x.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        return kernel_2d

    if sigma_grad>0:
        kernel_size = 2*math.ceil(sigma_grad)+1  # 核大小

        gaussian_kernel = create_gaussian_kernel(kernel_size, sigma_grad)
        gaussian_kernel = gaussian_kernel.expand(1, 1, -1, -1)  # 使其适用于2D卷积

        # 将张量转换为具有批处理维度和通道维度的格式
        G = G.unsqueeze(0).unsqueeze(0).float()

        # 使用F.conv2d进行高斯模糊
        G_blurred = F.conv2d(G, gaussian_kernel.cuda(), padding=kernel_size//2).squeeze(0).squeeze(0)

        # 移除多余的维度
        G = G_blurred
    # 差一个转制G
    G = G.transpose(1,0) / gscale
    mask = torch.tensor(param['roi']).to(G.device)
    G = G * mask

    # bone_mask = param['bone_mask']
    # G[bone_mask==1] = G[bone_mask==1]*2

    J = J / fscale / 1e9

    return J,torch.tensor(G.detach().cpu().numpy().reshape(-1,order = 'F')).cuda()
