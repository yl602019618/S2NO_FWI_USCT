import numpy as np
import torch
import pytorch_lightning as pl
from torch import optim, nn
import torch.nn.functional as F
from ..utils.loss import LpLoss,RRMSE
import matplotlib.pyplot as plt
import math
import os
from scipy.io import loadmat
from ..utils.visual_loss_curve_pretrain import visual_train_breast, visual_train_arm,visual_train_limb
class SpectralConv2d_born(nn.Module):
    '''
    Input:
    x: batch,in, x,y
    x_eps: batch,in,x,y
    Output: batch,out,x,y
    '''
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_born, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2,2, dtype=torch.float32)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2,2, dtype=torch.float32)
        )
    def compl_mul2d(self,input, weights):
        #print(input.shape,weights.shape)
        real = torch.einsum('bixy,ioxy->boxy',input[...,0],weights[...,0])-torch.einsum('bixy,ioxy->boxy',input[...,1],weights[...,1])
        comp = torch.einsum('bixy,ioxy->boxy',input[...,0],weights[...,1])+torch.einsum('bixy,ioxy->boxy',input[...,1],weights[...,0])
        output = torch.cat((real.unsqueeze(-1),comp.unsqueeze(-1)),dim = -1)

        return output
    # Complex multiplication
    # def compl_mul2d(self, input, weights):
    #     # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
    #     return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x, x_eps):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.view_as_real(torch.fft.rfft2(x * x_eps))
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,2, dtype=torch.float32, device=x.device
        )

        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2,:], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2,:], self.weights2
        )
        # Return to physical space
        x = torch.fft.irfft2(torch.view_as_complex(out_ft), s=(x.size(-2), x.size(-1)))
        return x


class FNO_step(nn.Module):
    def __init__(self, width, modes1, modes2 ):
        super(FNO_step, self).__init__()
        self.modes1 = modes1 # Truncate mode of FT in x
        self.modes2 = modes2 # Truncate mode of FT in y
        self.width = width #image size
        self.conv0 = SpectralConv2d_born(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.bn = nn.BatchNorm2d(self.width, eps=0, momentum=0.5, affine=True)
        #self.coef = nn.Parameter(torch.ones(1)).cuda()
    def forward(self,input):
        # Qin Jiu Shao version
        x,v_v,v_q,x_0, v_mq= input
        x_n = x.clone()
        x = self.conv0(x, v_v)
        x = x * v_mq+ v_q * x_n
        x = self.w1(F.leaky_relu(self.w0(x)))
        x = F.leaky_relu(x)
        x = x + x_0
        x = self.bn(x)
        return [x,v_v,v_q,x_0,v_mq]

class S2NO_step_wrap(nn.Module):
    def __init__(self, model,width):
        super(S2NO_step_wrap, self).__init__()
        self.model  = model
        #self.DC = DC1(self.width, 8, self.width, 3)
        self.bn = nn.BatchNorm2d(width, eps=0, momentum=0.5, affine=True)
        #self.ln = torch.nn.LayerNorm(489, eps=1e-05, elementwise_affine=True)

    def forward(self,input):
        # Qin Jiu Shao version
        #x = self.DC(x)
        [x,v_v,v_q,x_0] = self.model(input)
        x = self.bn(x)
        return [x,v_v,v_q,x_0]


class S2NO_pretrain(pl.LightningModule):
    def __init__(self,width=32,
                    modes1=12,
                    modes2=12,
                    layer_num=4,
                    padding = 0,
                    dim_input = 1,
                    source_type = 'theta',
                    loss = "rel_l2",
                    learning_rate = 1e-2,
                    step_size= 100,
                    gamma= 0.5,
                    weight_decay= 1e-5,
                    F_feature = False,
                    add_term = True,
                    eta_min = 2e-4,
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
        super().__init__()
        self.with_grid = True
        if self.with_grid == True:
            dim_input +=2
        self.source_type = source_type
        if self.source_type == 'source':
            dim_input +=2
        elif self.source_type == 'theta':
            dim_input +=2
        self.padding = padding
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.eta_min = eta_min
        if loss == 'l1':
            self.criterion = nn.L1Loss()
        elif loss == 'l2':
            self.criterion = nn.MSELoss()
            self.criterion_val = LpLoss()
        elif loss == "rel_l2":
            self.criterion =LpLoss()
            self.criterion_val = RRMSE()



        self.modes1 = modes1 # Truncate mode of FT in x
        self.modes2 = modes2 # Truncate mode of FT in y
        self.width = width #image size
        self.padding = padding  # pad the domain if input is non-periodic
        self.fc_c1 = nn.Linear(3, self.width)  # input channel is 3: (c(x, y), x, y)
        self.fc_c2 = nn.Linear(self.width, self.width)
        self.fc_c3 = nn.Linear(3, self.width)  # input channel is 3: (c(x, y), x, y)
        self.fc_c4 = nn.Linear(self.width, self.width)
        self.fc_0 = nn.Linear(dim_input, self.width)  # input channel is 4: (s(x,y),c(x, y), x, y)
        self.layer_num = layer_num
        self.fno_step = []

        for i in range(layer_num):
            self.fno_step.append(FNO_step(self.width,self.modes1,self.modes2).to(self.device))
        self.net =nn.Sequential(*self.fno_step)
        self.F_feature = F_feature
        self.add_term = add_term
        self.fc1 = nn.Linear(self.width, 256)
        self.fc2 = nn.Linear(256, 2)
        self.val_iter = 0
        self.bn = nn.BatchNorm2d(self.width, eps=0, momentum=0.5, affine=True, track_running_stats=True)
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



    def forward(self,sos,src):
        '''
        x: batch,x,y,channel=3 (c(x),s(x))
        '''
        if self.source_type == 'theta':
            src_input = src[:,:,:,:]
        elif self.source_type == 'source':
            src_input = src[:,:,:,0:2]
        # x = torch.cat((sos, src_input), dim=-1)
        x_1 = sos
        field = src[...,0:2].clone()

        grid = self.get_grid(x_1.shape, x_1.device, field)

        x_0 = torch.cat((x_1, grid,src_input), dim=-1)
        x_c = torch.cat((x_1, grid), dim=-1)

        v_0 = self.fc_0(x_0)
        v_0 = v_0.permute(0, 3, 1, 2)
        if self.padding >=1:
            v_0 = F.pad(v_0, [0, self.padding, 0, self.padding])

        v_v = self.fc_c2(F.tanh(self.fc_c1(x_c)))
        v_v = v_v.permute(0, 3, 1, 2)
        if self.padding >=1:
            v_v = F.pad(v_v, [0, self.padding, 0, self.padding])
        v_q = self.fc_c4(F.tanh(self.fc_c3(x_c)))
        v_q = v_q.permute(0, 3, 1, 2)
        if self.padding >=1:
            v_q = F.pad(v_q, [0, self.padding, 0, self.padding])
        v_mq = 1 - v_q
        x_1 = v_0
        x_out = x_1.clone()
        #for i in range(self.layer_num):
        [x_out,v_v,v_q,x_1,v_mq] = self.net([x_out,v_v,v_q,x_1, v_mq])
        if self.padding >=1:
            x_out = x_out[..., : -self.padding, : -self.padding]
        x_out = x_out.permute(0, 2, 3, 1)
        x_out = self.fc1(x_out)
        x_out = F.leaky_relu(x_out)
        x_out = self.fc2(x_out)#* self.field.to(x_out.device)
        if self.add_term == True:
            #x_out = x_out + field
            x_out = torch.view_as_real(torch.view_as_complex(field.to(x_out.device))*(1+torch.view_as_complex(x_out)))
        return x_out

    def get_grid(self, shape, device,field):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        #feature = []
        gridxy = torch.cat((gridx,gridy), dim=-1).to(device)
        #feature.append(gridxy)
        # if self.F_feature == True:
        #     for i in range(-3,4):
        #         feature.append(torch.sin(2**(-i)*gridxy))
        #         feature.append(torch.cos(2**(-i)*gridxy))
        #feature.append(field)
        return gridxy

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
