import numpy as np
import torch
import torch.nn as nn
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
    def abs(self, x, y):
        num_examples = x.size()[0]
        h = 1.0 / (x.size()[1] - 1.0)
        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)
        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


class RRMSE(object):
    def __init__(self, ):
        super(RRMSE, self).__init__()

    def __call__(self, x, y):
        num_examples = x.size()[0]
        norm = torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), 2 , 1)**2
        normy = torch.norm( y.view(num_examples,-1), 2 , 1)**2
        mean_norm = torch.mean((norm/normy)**(1/2))
        return mean_norm


# class LpLoss_new(object):
#     def __init__(self, d=2, p=2, size_average=True, reduction=True):
#         super(LpLoss_new, self).__init__()
#         assert d > 0 and p > 0
#         self.d = d
#         self.p = p
#         self.reduction = reduction
#         self.size_average = size_average
#     def abs(self, x, y):
#         num_examples = x.size()[0]
#         h = 1.0 / (x.size()[1] - 1.0)
#         all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)
#         if self.reduction:
#             if self.size_average:
#                 return torch.mean(all_norms)
#             else:
#                 return torch.sum(all_norms)
#         return all_norms

#     def rel(self, x, y):
#         '''
#         x: b,h,w,2
#         y: b,h,w,2
#         '''

#         num_examples = x.size()[0]
#         y_norm = torch.norm(y,p = 2, dim=3,keepdim = True) # b,h,w,2
#         y_mean = torch.mean(y_norm, dim = [1,2,3],keepdim = True) #b,h,w,2
#         weight = y_norm / y_mean/2

#         diff = (x-y)/weight

#         diff_norms = torch.norm(diff.reshape(num_examples, -1), self.p, 1)
#         y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

#         if self.reduction:
#             if self.size_average:
#                 return torch.mean(diff_norms/y_norms)
#             else:
#                 return torch.sum(diff_norms/y_norms)

#         return diff_norms/y_norms

#     def __call__(self, x, y):
#         return self.rel(x, y)
