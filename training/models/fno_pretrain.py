import pytorch_lightning as pl
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from ..utils.loss import LpLoss,RRMSE
from ..models.basics_model import get_grid2D,FC_nn
from scipy.io import loadmat
from ..utils.visual_loss_curve_pretrain import visual_train_breast, visual_train_arm,visual_train_limb
import os
##################
#fourier convolution 2d block
class fourier_conv_2d(nn.Module):
    def __init__(self, in_, out_, wavenumber1, wavenumber2):
        super(fourier_conv_2d, self).__init__()
        self.out_ = out_
        self.wavenumber1 = wavenumber1
        self.wavenumber2 = wavenumber2
        scale = (1 / (in_ * out_))
        self.weights1 = nn.Parameter(scale * torch.rand(in_, out_, wavenumber1, wavenumber2, 2 , dtype=torch.float32))
        self.weights2 = nn.Parameter(scale * torch.rand(in_, out_, wavenumber1, wavenumber2, 2 , dtype=torch.float32))
        # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ,2), (in_channel, out_channel, x,y,2) -> (batch, out_channel, x,y)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)
    def forward(self, x):
        #input: batch,channel,x,y
        #out: batch,channel,x,y
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.view_as_real(torch.fft.rfft2(x))#input: batch,channel,x,y->batch,channel,x,y,2
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_,  x.size(-2), x.size(-1)//2 + 1,2, dtype=torch.float32, device=x.device)
        out_ft[:, :, :self.wavenumber1, :self.wavenumber2,:] = \
            self.compl_mul2d(x_ft[:, :, :self.wavenumber1, :self.wavenumber2,:], self.weights1)
        out_ft[:, :, -self.wavenumber1:, :self.wavenumber2,:] = \
            self.compl_mul2d(x_ft[:, :, -self.wavenumber1:, :self.wavenumber2,:], self.weights2)
        #Return to physical space
        x = torch.fft.irfft2(torch.view_as_complex(out_ft), s=(x.size(-2), x.size(-1)))
        return x


######################################################
#fourier convolution layer using fourier conv block and conv2d block
class Fourier_layer(nn.Module):
    def __init__(self,  features_, wavenumber, activation = 'relu', is_last = False):
        super(Fourier_layer, self).__init__()
        self.W =  nn.Conv2d(features_, features_, 1)
        self.fourier_conv = fourier_conv_2d(features_, features_ , *wavenumber)
        if is_last== False:
            self.act = F.relu
        else:
            self.act = nn.Identity()
    def forward(self, x):
        x1 = self.fourier_conv(x)
        x2 = self.W(x)
        return self.act(x1 + x2)

######################################################
#fourier neural operator implementation
class FNO_pretrain(pl.LightningModule):
    def __init__(self,
                    wavenumber, features_,
                    padding = 9,
                    activation= 'relu',
                    lifting = None,
                    proj =  None,
                    dim_input = 1,
                    source_type = 'theta',
                    add_term = True,
                    loss = "rel_l2",
                    learning_rate = 1e-3,
                    step_size= 100,
                    gamma= 0.5,
                    weight_decay= 1e-5,
                    eta_min = 5e-4,
                    val_exp = True,
                    src_path_breast = '',
                    gt_path_breast = '',
                    src_path_arm = '',
                    gt_path_arm = '',
                    src_path_limb = '',
                    gt_path_limb = '',
                    speed_path_breast = '',
                    speed_path_arm = '',
                    speed_path_limb = '',
                    valid_num_breast = 320,
                    valid_num_arm = 373,
                    ):
        super(FNO_pretrain, self).__init__()
        self.with_grid = True
        if self.with_grid == True:
            dim_input +=2
        self.source_type = source_type
        if self.source_type == 'source':
            dim_input +=2
        elif self.source_type == 'theta':
            dim_input +=2
        self.padding = padding
        self.layers = len(wavenumber)
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.eta_min = eta_min
        self.add_term = add_term
        if loss == 'l1':
            self.criterion = nn.L1Loss()
            self.criterion_val = LpLoss()
        elif loss == 'l2':
            self.criterion = nn.MSELoss()
            self.criterion_val = LpLoss()
        elif loss == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()
            self.criterion_val = LpLoss()
        elif loss == "rel_l2":
            self.criterion = LpLoss()
            self.criterion_val = RRMSE()

        if lifting is None:
            self.lifting = FC_nn([dim_input, features_//2, features_],
                                    activation = "relu",
                                    outermost_norm=False
                                    )
        else:
            self.lifting = lifting
        if  proj is None:
            self.proj =  FC_nn([features_, features_//2, 2],
                                activation = "relu",
                                outermost_norm=False
                                    )
        else:
            self.proj = proj
        self.fno = []
        for l in range(self.layers-1):
            self.fno.append(Fourier_layer(features_ = features_,
                                        wavenumber=[wavenumber[l]]*2,
                                        activation = activation))

        self.fno.append(Fourier_layer(features_=features_,
                                        wavenumber=[wavenumber[-1]]*2,
                                        activation = activation,
                                        is_last= True))
        self.fno =nn.Sequential(*self.fno)
        self.val_iter = 0

        self.val_exp = val_exp
        self.valid_num_breast = valid_num_breast
        self.valid_num_arm = valid_num_arm
        self.save_path = None
        self.exp_name = None
        self.src_path_breast = src_path_breast
        self.gt_path_breast = gt_path_breast
        self.src_path_arm = src_path_arm
        self.src_path_limb = src_path_limb
        self.gt_path_arm = gt_path_arm
        self.gt_path_limb = gt_path_limb



        if self.val_exp:
            data = loadmat(speed_path_breast)['image']
            data = (1500/data-1)*30
            sos = torch.tensor(data,dtype=torch.float).view(1,480,480,1)
            sos = sos.repeat(64,1,1,1)
            self.sos_exp_breast = sos
            index_2 = np.arange(0,64).reshape(64,1,1,1)
            src = np.load(self.src_path_breast)[:,:,:]*2e-3
            src = np.concatenate((np.real(src)[:,:,:,np.newaxis],np.imag(src)[:,:,:,np.newaxis]),axis = -1)
            self.src_exp_breast =  torch.tensor(src,dtype=torch.float).view(64,480,480,2)
            gt_np = loadmat(self.gt_path_breast)['u_all_single']
            gt = torch.view_as_real(torch.tensor(gt_np))
            self.gt_breast = torch.tensor(gt)

            data = loadmat(speed_path_arm)['speed']
            data = (1500/data-1)*30
            sos = torch.tensor(data,dtype=torch.float).view(1,480,480,1)
            sos = sos.repeat(64,1,1,1)
            self.sos_exp_arm = sos
            index_2 = np.arange(0,64).reshape(64,1,1,1)
            src = np.load(self.src_path_arm)[:,:,:]*2e-3
            #theta = (index_2/64*2*np.pi)*np.ones((1,480,480,1))
            src = np.concatenate((np.real(src)[:,:,:,np.newaxis],np.imag(src)[:,:,:,np.newaxis]),axis = -1)
            self.src_exp_arm =  torch.tensor(src,dtype=torch.float).view(64,480,480,2)
            gt_np = loadmat(self.gt_path_arm)['u_all_single']
            gt = torch.view_as_real(torch.tensor(gt_np))
            self.gt_arm = torch.tensor(gt)

            data = loadmat(speed_path_limb)['image']
            data = (1500/data-1)*30
            sos = torch.tensor(data,dtype=torch.float).view(1,480,480,1)
            sos = sos.repeat(64,1,1,1)
            self.sos_exp_limb = sos
            index_2 = np.arange(0,64).reshape(64,1,1,1)
            src = np.load(self.src_path_limb)[:,:,:]*2e-3
            #theta = (index_2/64*2*np.pi)*np.ones((1,480,480,1))
            src = np.concatenate((np.real(src)[:,:,:,np.newaxis],np.imag(src)[:,:,:,np.newaxis]),axis = -1)
            self.src_exp_limb =  torch.tensor(src,dtype=torch.float).view(64,480,480,2)
            gt_np = loadmat(self.gt_path_limb)['u_all_single']
            gt = torch.view_as_real(torch.tensor(gt_np))
            self.gt_limb = torch.tensor(gt)


    def forward(self, sos,src):
        # 100,960,960,2
        #x = torch.cat((sos, src), dim=-1)
        if self.source_type == 'theta':
            src_input = src[:,:,:,:]
        elif self.source_type == 'source':
            src_input = src[:,:,:,0:2]
        x = sos
        field = src[...,:2].clone()
        if self.with_grid == True:
          grid = get_grid2D(x.shape, x.device)
        x = torch.cat((x,grid,src_input), dim=-1) # 100,960,960,4
        x = self.lifting(x)  # 100,960,960,feature_
        x = x.permute(0, 3, 1, 2)# batch,feature,x,y:   100,feature_,960,960
        x = nn.functional.pad(x, [0,self.padding, 0,self.padding])
        x = self.fno(x)# batch,feature,x,y:   100,feature_,960+pad,960+pad
        x = x[..., :-self.padding, :-self.padding] # batch,feature,x,y:   100,feature_,960,960
        x = x.permute(0, 2, 3, 1 )
        x =self.proj(x)  # batch,x,y,2:   100,960,960 ,2
        if self.add_term == True:

            x = torch.view_as_real(torch.view_as_complex(field.to(x.device))*(1+torch.view_as_complex(x)))
        return x

    def training_step(self, batch: torch.Tensor, batch_idx):
        sos,src,y,index = batch
        batch_size = sos.shape[0]
        out = self(sos,src)
        loss = self.criterion(out, y)
        self.log("loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, val_batch: torch.Tensor, batch_idx):
        sos, src, y, index = val_batch
        split_index = (index[-1] + index[0])//2
        batch_size = sos.shape[0]
        out = self(sos, src)
        val_loss = self.criterion_val(out.view(batch_size, -1), y.view(batch_size, -1))

        # 定义数据集的区间
        breast_end = self.valid_num_breast*64
        arm_end = self.valid_num_arm*64


        # 判断当前 batch_idx 属于哪个数据集
        if split_index < breast_end:
            dataset = 'breast'
        elif breast_end <= split_index < breast_end + arm_end:
            dataset = 'arm'
        else:
            dataset = 'limb'

        # 按数据集分类记录 val_loss
        if dataset == 'breast':
            self.log('val_loss_breast', val_loss, on_epoch=True, prog_bar=True, logger=True)
        elif dataset == 'arm':
            self.log('val_loss_arm', val_loss, on_epoch=True, prog_bar=True, logger=True)
        elif dataset == 'limb':
            self.log('val_loss_limb', val_loss, on_epoch=True, prog_bar=True, logger=True)

        # 记录总的 val_loss
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)

        if self.val_exp and self.val_iter % 1280 ==0: ##1450
            error = self.validation_exp_breast(device = out.device)
            #self.log('exp_loss_breast', error.item(), on_epoch=True, prog_bar=True, logger=True)
            error = self.validation_exp_arm(device = out.device)
            #self.log('exp_loss_arm', error.item(), on_epoch=True, prog_bar=True, logger=True)
            error = self.validation_exp_limb(device = out.device)
            #self.log('exp_loss_limb', error.item(), on_epoch=True, prog_bar=True, logger=True)

        self.val_iter += 1
        return val_loss

    # def on_validation_epoch_end(self):
    #     error = self.validation_exp_breast(device = self.device)
    #     #self.log('exp_loss_breast', error.item(), on_epoch=True, prog_bar=True, logger=True)
    #     error = self.validation_exp_arm(device = self.device)
    #     #self.log('exp_loss_arm', error.item(), on_epoch=True, prog_bar=True, logger=True)
    #     error = self.validation_exp_limb(device = self.device)
    #     #self.log('exp_loss_limb', error.item(), on_epoch=True, prog_bar=True, logger=True)

    def validation_exp_breast(self,device):
        pred = self(self.sos_exp_breast.to(device), self.src_exp_breast.to(device))
        y = self.gt_breast.to(device)
        pred = pred / 2e-3
        error = self.criterion_val(y[::4,:].reshape(64,-1),pred[:,:].reshape(64,-1))
        self.log('exp_loss_breast', error.item(), on_epoch=True, prog_bar=True, logger=True)###newline
        self.plot_exp(y, pred, path = self.save_path,error = error,name = 'breast')
        visual_train_breast(csv_file_path = os.path.join(self.save_path,self.exp_name,'version_0/metrics.csv'),save_path = self.save_path)
        return error
    def validation_exp_arm(self,device):
        pred = self(self.sos_exp_arm.to(device), self.src_exp_arm.to(device))
        y = self.gt_arm.to(device)
        pred = pred / 2e-3
        error = self.criterion_val(y[::4,:].reshape(64,-1),pred[:,:].reshape(64,-1))
        self.log('exp_loss_arm', error.item(), on_epoch=True, prog_bar=True, logger=True)###newline
        self.plot_exp(y, pred, path = self.save_path,error = error,name = 'arm')
        visual_train_arm(csv_file_path = os.path.join(self.save_path,self.exp_name,'version_0/metrics.csv'),save_path = self.save_path)
        return error

    def validation_exp_limb(self,device):
        pred = self(self.sos_exp_limb.to(device), self.src_exp_limb.to(device))
        y = self.gt_limb.to(device)
        pred = pred / 2e-3
        error = self.criterion_val(y[::4,:].reshape(64,-1),pred[:,:].reshape(64,-1))
        self.log('exp_loss_limb', error.item(), on_epoch=True, prog_bar=True, logger=True)###newline
        self.plot_exp(y, pred, path = self.save_path,error = error,name = 'limb')
        visual_train_limb(csv_file_path = os.path.join(self.save_path,self.exp_name,'version_0/metrics.csv'),save_path = self.save_path)
        return error

    def plot_exp(self,field, pred_field,path,error,name):
        field = field.detach().cpu().numpy()
        pred_field = pred_field.detach().cpu().numpy()
        index  = 0
        plt.figure(figsize=(10,10))
        plt.subplot(2,2,1)
        plt.imshow(np.log(field[index,:,:,0]**2+field[index,:,:,1]**2)/2,cmap='seismic')
        plt.colorbar()
        #plt.clim(-4000,4000)
        plt.clim(0,12)
        plt.axis('off')
        plt.subplot(2,2,2)
        plt.imshow(np.log(pred_field[index,:,:,0]**2+pred_field[index,:,:,1]**2)/2,cmap='seismic')
        plt.colorbar()
        #plt.clim(-4000,4000)
        plt.clim(0,12)
        plt.axis('off')

        plt.subplot(2,2,3)
        plt.imshow(field[index,:,:,0],cmap='seismic')
        plt.colorbar()
        plt.clim(-4000,4000)
        #plt.clim(0,12)
        plt.axis('off')

        plt.subplot(2,2,4)
        plt.imshow(pred_field[index,:,:,0],cmap='seismic')
        plt.colorbar()
        plt.clim(-4000,4000)
        #plt.clim(0,12)
        plt.axis('off')
        save_path = os.path.join(path, f'{name}_test_epoch{self.val_iter}_error{error.item()}.png')
        plt.savefig(save_path)
        plt.close()

    def log_wandb_image(self,wandb,  sos, field, pred_field):
        fig, ax = plt.subplots(2, 3, figsize=(12, 8))
        ax = ax.flatten()
        ax0 = ax[0].imshow(sos[...,0], cmap="inferno")
        ax[0].set_title("Sound speed")
        ax[1].imshow(field[...,0],vmin=-1, vmax=1, cmap="RdBu_r")
        ax[1].set_title("Field")
        ax[2].imshow(pred_field[...,0],vmin=-1, vmax=1, cmap="RdBu_r")
        ax[2].set_title("Predicted field")
        ax[3].imshow(np.log(field[...,0]**2+field[...,1]**2), cmap="RdBu_r")
        ax[3].set_title("Field")
        ax[4].imshow(np.log(pred_field[...,0]**2+pred_field[...,1]**2), cmap="RdBu_r")
        ax[4].set_title("Predicted field")
        img = wandb.Image(plt)
        plt.close()
    def configure_optimizers(self, optimizer=None, scheduler=None):
        if optimizer is None:
            optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if  scheduler is None:
            #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = self.step_size, eta_min= self.eta_min)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 6, gamma = 0.1)# 6 0.5

        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler
        },
    }
