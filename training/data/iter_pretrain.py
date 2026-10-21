

from bisect import bisect
import os
import torch

import numpy as np
from torch.utils.data import Dataset, DataLoader



class File_Loader(Dataset):
    #data loader file
    def __init__(self, data_paths, target_paths,u_homo_path_breast = None, u_homo_path_arm = None , u_homo_path_limb = None , size =128, exp = 'single', split_num = None):
        if exp != "multi":
            raise ValueError("Released datasets use exp=multi with 64 source fields per sample.")
        self.size = size
        self.exp = exp
        self.u_homo_path_breast = u_homo_path_breast
        self.u_homo_path_arm = u_homo_path_arm
        self.src_breast = torch.tensor(np.load(self.u_homo_path_breast))
        self.src_arm = torch.tensor(np.load(self.u_homo_path_arm))
        self.u_homo_path_limb = u_homo_path_limb
        self.src_limb = torch.tensor(np.load(self.u_homo_path_limb))
        print('breast source shape',self.src_breast.shape)
        print('arm source shape',self.src_arm.shape)
        print('limb source shape',self.src_limb.shape)
        self.data_paths = data_paths
        self.target_paths = target_paths
        self.start_indices = [0] * len(data_paths) #4600
        self.data_count = 0
        self.split_num = split_num
        for index, memmap in enumerate(self.data_paths):
            self.start_indices[index] = self.data_count
            if exp == 'single':
                self.data_count += 1
            elif exp == 'multi':
                self.data_count += 1
        self.start_indices_target = [0] * len(target_paths)
        self.data_count_target = 0
        for index, memmap in enumerate(self.target_paths):
            self.start_indices_target[index] = self.data_count_target
            if exp == 'single':
                self.data_count_target += 1
            elif exp == 'multi':
                self.data_count_target += 64
        print({'data_count':self.data_count, 'self.data_count_target': self.data_count_target})
    def __len__(self):
        return self.data_count_target #2400
    def __getitem__(self, index):
        if self.exp == 'single':
            memmap_index = bisect(self.start_indices_target, index) - 1 #0-4600*8-1
            index_in_memmap = index - self.start_indices_target[memmap_index] #0-8
            data = (1500/np.load(self.data_paths[memmap_index], mmap_mode='r')-1)*30
            #data = self.data_norm.encode(np.copy(self.data_memmaps[memmap_index][:,:]))
            target = np.load(self.target_paths[memmap_index], mmap_mode='r')*2e-3
            #target = self.target_norm.encode(np.copy(self.target_memmaps[memmap_index][:,:]))
            target = np.concatenate((np.real(target)[:,:,np.newaxis],np.imag(target)[:,:,np.newaxis]),axis = -1)


            theta = np.zeros((self.size,self.size,1))
            #src = self.target_norm.encode(torch.view_as_complex(self.src[0,:,:,:]))
            src = torch.view_as_complex(self.src[0,:,:])*2e-3 #torch.cat((self.src[0,:,:],torch.zeros_like(self.src[0,:,:])),dim = -1)
            src = np.concatenate((np.real(src)[:,:,np.newaxis],np.imag(src)[:,:,np.newaxis],theta),axis = -1)
            #print(src.shape)
        else:
            memmap_index = bisect(self.start_indices_target, index) - 1
            index_in_memmap = index - self.start_indices_target[memmap_index]
            index_1 = index//64
            index_2 = index - index_1*64
            data = (1500/np.load(self.data_paths[index_1], mmap_mode='r')-1)*30
            #data = (np.load(self.data_paths[index_1], mmap_mode='r')[0,:,:]-1500)*1e-2
            #data = self.data_norm.encode(np.copy(self.data_memmaps[index_1][0,:,:]))
            target = np.load(self.target_paths[memmap_index], mmap_mode='r')[index_in_memmap]*2e-3
            #target = self.target_norm.encode(np.copy(self.target_memmaps[memmap_index][index_in_memmap]))
            target = np.concatenate((np.real(target)[:,:,np.newaxis],np.imag(target)[:,:,np.newaxis]),axis = -1)
            #src = self.target_norm.encode(torch.view_as_complex(self.src[index_2*4,:,:,:])) #torch.cat((self.src[0,:,:],torch.zeros_like(self.src[0,:,:])),dim = -1)
            if index_1 < self.split_num[0]:
                src = self.src_breast[index_2,:,:]*2e-3
            elif index_1 < self.split_num[0] + self.split_num[1]:
                src = self.src_arm[index_2,:,:]*2e-3
            else:
                src = self.src_limb[index_2,:,:]*2e-3
            src = np.concatenate((np.real(src)[:,:,np.newaxis],np.imag(src)[:,:,np.newaxis]),axis = -1)
        return torch.tensor(data, dtype=torch.float).view(self.size,self.size,1),torch.tensor(src,dtype = torch.float).view(self.size,self.size,2), torch.tensor(target, dtype=torch.float).view(self.size,self.size,2), torch.tensor(index)

class GettingLists(object):
    def __init__(self,train_num_breast = 1600,
                    valid_num_breast = 400 ,
                    train_num_arm = 1600,
                    valid_num_arm = 400 ,
                    train_num_limb = 1600,
                    valid_num_limb = 400 ,
                    PATH_data_train_breast = 'lbs',
                    PATH_target_train_breast = 'lbs',
                    PATH_data_test_breast = 'lbs',
                    PATH_target_test_breast = 'lbs',
                    PATH_data_train_arm = 'lbs',
                    PATH_target_train_arm = 'lbs',
                    PATH_data_test_arm = 'lbs',
                    PATH_target_test_arm = 'lbs',
                    PATH_data_train_limb = 'lbs',
                    PATH_target_train_limb = 'lbs',
                    PATH_data_test_limb = 'lbs',
                    PATH_target_test_limb = 'lbs',
                    u_homo_path_breast = None,
                    u_homo_path_arm = None,
                    u_homo_path_limb = None,
                    batchsize= int(2000),
                    exp = 'single'):
        super(GettingLists, self).__init__()
        self.PATH_data_train_breast = PATH_data_train_breast
        self.PATH_data_test_breast = PATH_data_test_breast
        self.PATH_target_train_breast = PATH_target_train_breast
        self.PATH_target_test_breast = PATH_target_test_breast

        self.PATH_data_train_arm = PATH_data_train_arm
        self.PATH_data_test_arm = PATH_data_test_arm
        self.PATH_target_train_arm = PATH_target_train_arm
        self.PATH_target_test_arm = PATH_target_test_arm

        self.PATH_data_train_limb = PATH_data_train_limb
        self.PATH_data_test_limb = PATH_data_test_limb
        self.PATH_target_train_limb = PATH_target_train_limb
        self.PATH_target_test_limb = PATH_target_test_limb

        self.u_homo_path_breast = u_homo_path_breast
        self.u_homo_path_arm = u_homo_path_arm
        self.u_homo_path_limb = u_homo_path_limb

        self.batchsize = batchsize
        self.valid_num_breast = valid_num_breast
        self.train_num_breast = train_num_breast
        self.valid_num_arm = valid_num_arm
        self.train_num_arm = train_num_arm
        self.valid_num_limb = valid_num_limb
        self.train_num_limb = train_num_limb

        self.exp = exp

        self.velo_list_train_breast = (np.arange(self.train_num_breast)+1).tolist()
        self.velo_list_test_breast = (np.arange(self.valid_num_breast)+1).tolist()
        self.velo_list_train_arm = (np.arange(self.train_num_arm)+1).tolist()
        self.velo_list_train_limb = (np.arange(self.train_num_limb)+1).tolist()
        self.velo_list_test_arm = (np.arange(self.valid_num_arm)+1).tolist()
        self.velo_list_test_limb = (np.arange(self.valid_num_limb)+1).tolist()

        if self.exp == 'single':
            self.pressure_list_train_breast = self.velo_list_train_breast
            self.pressure_list_test_breast = self.velo_list_test_breast
            self.pressure_list_train_arm = self.velo_list_train_arm
            self.pressure_list_test_arm = self.velo_list_test_arm
            self.pressure_list_train_limb = self.velo_list_train_limb
            self.pressure_list_test_limb = self.velo_list_test_limb
        else:
            self.pressure_list_train_breast = self.velo_list_train_breast
            self.pressure_list_test_breast = self.velo_list_test_breast
            self.pressure_list_train_arm = self.velo_list_train_arm
            self.pressure_list_test_arm = self.velo_list_test_arm
            self.pressure_list_train_limb = self.velo_list_train_limb
            self.pressure_list_test_limb = self.velo_list_test_limb

    def get_list(self, do):
        if do == 'train':
            list_in_breast = [os.path.join(self.PATH_data_train_breast,
                                                        f'train_{k}.npy') for k in \
                                                        self.velo_list_train_breast]
            list_in_arm = [os.path.join(self.PATH_data_train_arm,
                                                        f'train_{k}.npy') for k in \
                                                        self.velo_list_train_arm]
            list_in_limb = [os.path.join(self.PATH_data_train_limb,
                                                        f'train_{k}.npy') for k in \
                                                        self.velo_list_train_limb]
            list_out_breast = [os.path.join(self.PATH_target_train_breast,
                                                        f'train_{k}.npy')for k in \
                                                        self.pressure_list_train_breast]
            list_out_arm = [os.path.join(self.PATH_target_train_arm,
                                                        f'train_{k}.npy')for k in \
                                                        self.pressure_list_train_arm]
            list_out_limb = [os.path.join(self.PATH_target_train_limb,
                                                        f'train_{k}.npy')for k in \
                                                        self.pressure_list_train_limb]
            in_list = list_in_breast + list_in_arm + list_in_limb
            out_list = list_out_breast + list_out_arm + list_out_limb

            in_limit_train  = np.array(in_list)
            out_limit_train = np.array(out_list)
            return in_limit_train, out_limit_train

        elif do == 'validation':
            list_in_breast = [os.path.join(self.PATH_data_test_breast,
                                                        f'test_{k}.npy') for k in \
                                                        self.velo_list_test_breast]
            list_in_arm = [os.path.join(self.PATH_data_test_arm,
                                                        f'test_{k}.npy') for k in \
                                                        self.velo_list_test_arm]
            list_in_limb = [os.path.join(self.PATH_data_test_limb,
                                                        f'test_{k}.npy') for k in \
                                                        self.velo_list_test_limb]

            list_out_breast = [os.path.join(self.PATH_target_test_breast,
                                                        f'test_{k}.npy')for k in \
                                                        self.pressure_list_test_breast]

            list_out_arm = [os.path.join(self.PATH_target_test_arm,
                                                        f'test_{k}.npy')for k in \
                                                        self.pressure_list_test_arm]
            list_out_limb = [os.path.join(self.PATH_target_test_limb,
                                                        f'test_{k}.npy')for k in \
                                                        self.pressure_list_test_limb]
            in_limit_valid  = np.array(list_in_breast + list_in_arm + list_in_limb)
            out_limit_valid= np.array(list_out_breast + list_out_arm + list_out_limb)
            return  in_limit_valid, out_limit_valid
        elif do =='test':
            list_in_breast = [os.path.join(self.PATH_data_test_breast,
                                                        f'test_{k}.npy') for k in \
                                                        self.velo_list_test_breast]
            list_in_arm = [os.path.join(self.PATH_data_test_arm,
                                                        f'test_{k}.npy') for k in \
                                                        self.velo_list_test_arm]
            list_in_limb = [os.path.join(self.PATH_data_test_limb,
                                                        f'test_{k}.npy') for k in \
                                                        self.velo_list_test_limb]
            list_out_breast = [os.path.join(self.PATH_target_test_breast,
                                                        f'test_{k}.npy')for k in \
                                                        self.pressure_list_test_breast]
            list_out_arm = [os.path.join(self.PATH_target_test_arm,
                                                        f'test_{k}.npy')for k in \
                                                        self.pressure_list_test_arm]
            list_out_limb = [os.path.join(self.PATH_target_test_limb,
                                                        f'test_{k}.npy')for k in \
                                                        self.pressure_list_test_limb]

            in_limit_test  = np.array(list_in_breast + list_in_arm + list_in_limb)
            out_limit_test = np.array(list_out_breast + list_out_arm + list_out_limb)
            return in_limit_test, out_limit_test
    def __call__(self, do = 'train'):
        return self.get_list(do)
    def get_dataloader(self,do,config):
        workers = config['data']['load_workers']
        size = config['data']['size']
        batchsize = self.batchsize
        if do == 'train':
            list_x_train, list_y_train = self.__call__('train')
            list_x_valid, list_y_valid = self.__call__('validation')
            Train_Data_set = File_Loader(list_x_train,list_y_train, size = size, exp = self.exp,u_homo_path_breast = self.u_homo_path_breast,  u_homo_path_arm = self.u_homo_path_arm, u_homo_path_limb = self.u_homo_path_limb, split_num = [self.train_num_breast,self.train_num_arm])
            Valid_Data_set = File_Loader(list_x_valid,list_y_valid, size = size, exp = self.exp,u_homo_path_breast = self.u_homo_path_breast, u_homo_path_arm = self.u_homo_path_arm,u_homo_path_limb = self.u_homo_path_limb, split_num = [self.valid_num_breast,self.valid_num_arm])
            train_loader = DataLoader(dataset = Train_Data_set,
                                    shuffle = True,
                                    batch_size = batchsize,
                                    num_workers= workers)
            valid_loader = DataLoader(dataset = Valid_Data_set,
                                    shuffle = False,
                                    batch_size =batchsize,
                                    num_workers= workers)
            return train_loader, valid_loader
        elif do == 'test':
            list_x_test, list_y_test = self.__call__('test')
            Test_Data_set = File_Loader(list_x_test, list_y_test, size = size, exp = self.exp,u_homo_path_breast = self.u_homo_path_breast,  u_homo_path_arm = self.u_homo_path_arm, u_homo_path_limb = self.u_homo_path_limb, split_num = [self.train_num_breast,self.train_num_arm])
            test_loader = DataLoader(dataset = Test_Data_set,
                                    shuffle = False,
                                    batch_size = batchsize,
                                    num_workers= workers)
            return test_loader

def datasetFactory(config,do='train'):
    c_data = config['data']
    gl = GettingLists(train_num_breast = c_data['train_num_breast'],
                    valid_num_breast = c_data['valid_num_breast'] ,
                    train_num_arm = c_data['train_num_arm'],
                    valid_num_arm = c_data['valid_num_arm'] ,
                    train_num_limb = c_data['train_num_limb'],
                    valid_num_limb = c_data['valid_num_limb'] ,
                    PATH_data_train_breast = c_data['PATH_data_train_breast'],
                    PATH_target_train_breast = c_data['PATH_target_train_breast'],
                    PATH_data_test_breast = c_data['PATH_data_test_breast'],
                    PATH_target_test_breast = c_data['PATH_target_test_breast'],
                    PATH_data_train_arm = c_data['PATH_data_train_arm'],
                    PATH_target_train_arm = c_data['PATH_target_train_arm'],
                    PATH_data_test_arm = c_data['PATH_data_test_arm'],
                    PATH_target_test_arm = c_data['PATH_target_test_arm'],
                    PATH_data_train_limb = c_data['PATH_data_train_limb'],
                    PATH_target_train_limb = c_data['PATH_target_train_limb'],
                    PATH_data_test_limb = c_data['PATH_data_test_limb'],
                    PATH_target_test_limb = c_data['PATH_target_test_limb'],
                    u_homo_path_breast = c_data['u_homo_path_breast'],
                    u_homo_path_arm = c_data['u_homo_path_arm'],
                    u_homo_path_limb = c_data['u_homo_path_limb'],
                     batchsize = c_data['batch'],
                     exp = c_data['exp'])
    return gl.get_dataloader(do = do,config = config)
