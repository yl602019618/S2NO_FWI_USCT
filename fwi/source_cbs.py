"""Original CBS wavefield conversion and complex source estimation."""

import torch
import numpy as np
from .cbs import CBS

def check_condition_or_error(condition):
    assert condition

def source_estimation(dobs,param,vp,transducer_irow,transducer_icol,config):
    '''
    dobs:observation 256,256 complex
    param:
    vp: input sos 480,480
    index: batchsize
    R: restrict operator: 256, 230400 sparse
    field_source: u_homo of field  batchsize, 480,480
    module1: FNO model
    '''
    batchsize = 256
    ix = param['x_src']
    iy = param['y_src']
    src_list = []
    for index_src in range(len(ix)):
         src_list.append(
            # the matlab index begin with 1 but python index begin with 0
            {"ix": ix[index_src] - 1, "iy": iy[index_src] - 1, "iz": 0, "value": (1.9514e+09 + 2.7546e+08*1j)/1e10}
        )
    speed = torch.tensor(vp, dtype=torch.float32).cuda().transpose(0,1)
    config.CBSConfig.freq = param['freq']
    cbs_solver = CBS(speed, config.CBSConfig)
    u = cbs_solver.solve(src_list, Batch=batchsize).transpose(0,1) #480,480,256

    u = u.permute(2,0,1)
    u = u/(1.9514e09 - 2.7546e08j)*1e10
    u = u.transpose(2,1) # 256,480,480

    dsim = u[:,transducer_irow,transducer_icol].clone()
    dobs = dobs.transpose(0, 1).cuda() # (rec, src) -> (src, rec)

    data_mask = param['select_mat'].T.astype(bool)
    dsim[~data_mask] = 0
    dobs[~data_mask] = 0

    num = torch.sum(torch.conj(dsim) * dobs, dim=1)
    den = torch.sum(torch.abs(dsim)**2, dim=1)
    s_intensity = num / den

    return u, s_intensity, dsim, dobs
