

import torch


def softmax_helper(x):
    """
    same as F.softmax
    """
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)

if __name__ == '__main__':
    x=torch.rand(1,3,2,2)
    print(x)
    y=softmax_helper(x)
    print('-'*33)
    print(y)
    print('-'*33)
    import torch.nn as nn
    import torch.nn.functional as F

    print(F.softmax(x,dim=1))