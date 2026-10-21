"""Original S2NO wavefield conversion and complex source estimation."""

import torch
import numpy as np

def check_condition_or_error(condition):
    assert condition

def batch_cal_wvfields(vp,index,field_source,module1):
    '''
    dobs:observation 256,256 complex
    param:
    vp: input sos 480,480
    index: batchsize
    R: restrict operator: 256, 230400 sparse
    field_source: u_homo of field  batchsize, 480,480
    module1: FNO model
    '''
    field_source_real = torch.real(field_source) #batch,480,480 real
    field_source_imag = torch.imag(field_source) #batch,480,480 real

    u = matpy_prediction(module1, vp, field_source_real,field_source_imag,index)

    u = u/(1.9514e09 - 2.7546e08j)*1e10
    u = torch.conj(u.transpose(2,1)) # b,480,480

    return u #(batch,480,480), complex

def source_estimation(dobs,param,vp,transducer_irow,transducer_icol,u_allsource,module1,batchsize):
    '''
    dobs:observation 256,256 complex
    param:
    vp: input sos 480,480
    index: batchsize
    R: restrict operator: 256, 230400 sparse
    field_source: u_homo of field  batchsize, 480,480
    module1: FNO model
    '''
    if batchsize <= 0 or 256 % batchsize:
        raise ValueError("S2NO FWI batch size must be a positive divisor of 256")
    num_iter = 256//batchsize
    u = torch.zeros(256, 480, 480, dtype=torch.complex64, device='cuda')
    with torch.no_grad():
        for it in range(num_iter):
            index = np.arange(it*batchsize,(it+1)*batchsize)
            field_source = u_allsource[index,:,:]
            u_batch = batch_cal_wvfields(vp,index,field_source,module1)
            u[index,:,:] = u_batch

    dsim = u[:,transducer_irow,transducer_icol].clone()
    dobs = dobs.transpose(0, 1).cuda() # (rec, src) -> (src, rec)

    data_mask = param['select_mat'].T.astype(bool)
    dsim[~data_mask] = 0
    dobs[~data_mask] = 0

    num = torch.sum(torch.conj(dsim) * dobs, dim=1)
    den = torch.sum(torch.abs(dsim)**2, dim=1)
    s_intensity = num / den

    return u, s_intensity, dsim, dobs

def matpy_prediction(module, v0, field_real, field_imag,index):
    '''
    module: FNO model
    v0: sos 480,480
    field_real: homo_field: batch,480,480
    field_imag: homo_field: batch,480,480
    isrc: source index
    batchflag: single
    '''

    pred = prediction( v0, field_real, field_imag,index,module)
    pred = pred*500
    pred = pred[...,0] + 1j*pred[...,1]

    return pred



def prediction(data, field_real, field_imag, batch_indices, model):
    """Original preprocessing; output is normalized real/imaginary channels."""
    data = (1500 / data - 1) * 30
    data = torch.tensor(data, dtype=torch.float).cuda()
    batchsize = field_real.shape[0]
    sos = data.reshape(1, 480, 480, 1).repeat(batchsize, 1, 1, 1).cuda()
    field = torch.concat([field_real.unsqueeze(-1), field_imag.unsqueeze(-1)], dim=-1).cuda() * 2e-3
    return model(sos, field)
