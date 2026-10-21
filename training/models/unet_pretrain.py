import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch import optim
import numpy as np
from ..utils.loss import LpLoss,RRMSE
import matplotlib.pyplot as plt
from ..models.basics_model import get_grid2D,FC_nn
from scipy.io import loadmat
from ..utils.visual_loss_curve_pretrain import visual_train_breast, visual_train_arm,visual_train_limb
import os


def get_unet_model(in_ch=1, out_ch=1, scales=5, skip=4,
                   channels=(32, 32, 64, 64, 128, 128), use_sigmoid=True,
                   use_norm=True):
    #assert (1 <= scales <= 6)
    skip_channels = [skip] * (scales)
    return UNet_module(in_ch=in_ch, out_ch=out_ch, channels=channels[:scales],
                skip_channels=skip_channels, use_sigmoid=use_sigmoid,
                use_norm=use_norm)


class DownBlock(nn.Module):
    '''
    Down sampling
    '''
    def __init__(self, in_ch, out_ch, kernel_size=3, num_groups=4, use_norm=True):
        super(DownBlock, self).__init__()
        to_pad = int((kernel_size - 1) / 2)
        if use_norm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size,
                          stride=2, padding=to_pad),
                nn.GroupNorm(num_channels=out_ch, num_groups=num_groups),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.GroupNorm(num_channels=out_ch, num_groups=num_groups),
                nn.LeakyReLU(0.2, inplace=True))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size,
                          stride=2, padding=to_pad),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class InBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, num_groups=2, use_norm=True):
        super(InBlock, self).__init__()
        to_pad = int((kernel_size - 1) / 2)
        if use_norm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.GroupNorm(num_channels=out_ch, num_groups=num_groups),
                nn.LeakyReLU(0.2, inplace=True))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch=4, kernel_size=3, num_groups=2, use_norm=True):
        super(UpBlock, self).__init__()
        to_pad = int((kernel_size - 1) / 2)
        self.skip = skip_ch > 0
        if skip_ch == 0:
            skip_ch = 1
        if use_norm:
            self.conv = nn.Sequential(
                nn.GroupNorm(num_channels=in_ch + skip_ch, num_groups=1), #LayerNorm
                nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size, stride=1,
                          padding=to_pad),
                nn.GroupNorm(num_channels=out_ch, num_groups=num_groups),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.GroupNorm(num_channels=out_ch, num_groups=num_groups),
                nn.LeakyReLU(0.2, inplace=True))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size, stride=1,
                          padding=to_pad),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.LeakyReLU(0.2, inplace=True))

        if use_norm:
            self.skip_conv = nn.Sequential(
                nn.Conv2d(out_ch, skip_ch, kernel_size=1, stride=1),
                nn.GroupNorm(num_channels=skip_ch, num_groups=1), #LayerNorm
                nn.LeakyReLU(0.2, inplace=True))
        else:
            self.skip_conv = nn.Sequential(
                nn.Conv2d(out_ch, skip_ch, kernel_size=1, stride=1),
                nn.LeakyReLU(0.2, inplace=True))

        self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True)
        self.concat = Concat()

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.skip_conv(x2)
        if not self.skip:
            x2 = x2 * 0
        x = self.concat(x1, x2)
        x = self.conv(x)
        return x


class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, *inputs):
        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]

        if (np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and
                np.all(np.array(inputs_shapes3) == min(inputs_shapes3))):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2,
                                   diff3:diff3 + target_shape3])
        return torch.cat(inputs_, dim=1)


class OutBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv(x)
        return x

    def __len__(self):
        return len(self._modules)




class UNet_module(nn.Module):
    def __init__(self, in_ch, out_ch, channels, skip_channels,
                 use_sigmoid=True, use_norm=True):
        super(UNet_module, self).__init__()

        assert (len(channels) == len(skip_channels))
        self.scales = len(channels)
        self.use_sigmoid = use_sigmoid
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.inc = InBlock(in_ch, channels[0], use_norm=use_norm)
        for i in range(1, self.scales):
            self.down.append(DownBlock(in_ch=channels[i - 1],
                                       out_ch=channels[i],
                                       use_norm=use_norm))
        for i in range(1, self.scales):
            self.up.append(UpBlock(in_ch=channels[-i],
                                   out_ch=channels[-i - 1],
                                   skip_ch=skip_channels[-i],
                                   use_norm=use_norm))
        self.outc = OutBlock(in_ch=channels[0],
                             out_ch=out_ch)

    def forward(self, x0):
        xs = [self.inc(x0), ]
        for i in range(self.scales - 1):
            xs.append(self.down[i](xs[-1]))
        x = xs[-1]
        for i in range(self.scales - 1):
            x = self.up[i](x, xs[-2 - i])

        return torch.sigmoid(self.outc(x)) if self.use_sigmoid else self.outc(x)



class UNet_pretrain(pl.LightningModule):
    """
    """
    def __init__(self,
                in_ch=1,
                out_ch=1,
                scales=5,
                skip=4,
                source_type = 'theta',
                channels=(32, 32, 64, 64, 128, 128),
                use_sigmoid=False,
                use_norm=True,
                learning_rate=0.001,
                step_size = 5,
                gamma = 0.5,
                weight_decay = 0.00001,
                eta_min = 5e-4,
                loss = 'rel_l2',
                F_feature = False,
                add_term = False,
                val_exp = False,
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
        self.in_ch = in_ch
        self.with_grid = True
        if self.with_grid == True:
            self.in_ch +=2
        self.source_type = source_type
        if self.source_type == 'source':
            self.in_ch +=2
        elif self.source_type == 'theta':
            self.in_ch +=2
        self.out_ch = out_ch
        self.channels = channels
        self.skip = skip
        self.use_sigmoid = use_sigmoid
        self.use_norm = use_norm
        self.scales = scales
        self.unet = self.build_unet()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.F_feature = F_feature
        self.add_term = add_term
        self.eta_min = eta_min
        if loss == 'l1':
            self.criterion = nn.L1Loss()
            self.criterion_val = RRMSE()
        elif loss == 'l2':
            self.criterion = nn.MSELoss()
            self.criterion_val = RRMSE()
            # self.criterion1 = LpLoss()
        elif loss == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()
            self.criterion_val = RRMSE()
        elif loss == "rel_l2":
            self.criterion =LpLoss()
            self.criterion_val = RRMSE()
        self.F_feature = F_feature
        self.add_term = add_term
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
        if self.source_type == 'theta':
            src_input = src[:,:,:,:]
        elif self.source_type == 'source':
            src_input = src[:,:,:,0:2]
        x = sos
        field = src[...,:2].clone()
        if self.with_grid == True:
          grid = get_grid2D(x.shape, x.device)
        x = torch.cat((x,grid,src_input), dim=-1) # 100,960,960,4


        x = self.unet(x.permute(0,3,1,2)).contiguous()
        x = x.permute(0,2,3,1).contiguous()
        #print(x.shape,field.shape)
        if self.add_term == True:
            x = torch.view_as_real(torch.view_as_complex(field)*(1+torch.view_as_complex(x)))
        return x
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        feature = []
        gridxy = torch.cat((gridx,gridy), dim=-1).to(device)
        feature.append(gridxy)
        if self.F_feature == True:
            for i in range(1,3):
                feature.append(torch.sin(10**(-i)*gridxy))
        return torch.cat(feature, dim=-1).to(device)

    def build_unet(self):
        """
            Build UNet with group normalization
            Parameters
            ----------
            Returns
            -------
            torch.nn module
                UNet model
            """
        return get_unet_model(in_ch=self.in_ch, out_ch=self.out_ch, scales=self.scales, skip=self.skip,
                   channels=self.channels, use_sigmoid=self.use_sigmoid,
                   use_norm=self.use_norm)

    def training_step(self, batch: torch.Tensor, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        sos,src,y, index = batch
        batch_size = sos.shape[0]
        out = self(sos,src)
        loss = self.criterion(out.view(batch_size,-1), y.view(batch_size,-1))
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
