import torch
import torch.nn as nn
import torch.optim as optim

def IAML_loss(Z1, Z2, cls_list, margin=1.0):
    '''
    Z1 = Z2 (both are same features)
    Z1, Z2 could be (sketch, model)
    '''
    total_loss = 0
    dist_table = torch.cdist(Z1, Z2, p=2)
    positive_idx = list()
    negative_idx = list()
    for i, c in enumerate(cls_list.tolist()):
        positive_idx = [index for index, el in enumerate(cls_list.tolist()) if el == c]
        positive_idx.remove(i)
        positive_dist_table = dist_table[i, positive_idx]
        positive_dist_max = torch.max(positive_dist_table)

        negative_idx = [index for index, el in enumerate(cls_list.tolist()) if el != c]
        negative_dist_table = dist_table[i, negative_idx]
        negative_dist_min = torch.min(negative_dist_table)

        loss_sub = margin - negative_dist_min + positive_dist_max
        loss = loss_sub if loss_sub > 0 else 0

        total_loss += loss
    return total_loss

def CMD_loss(Zt, Z2, cls_t, cls_2):
    '''
    Zt : features after tranformation network
    Z2 : features after model metric network
    '''
    cmd_loss = 0
    cls_list = sorted(list(set(cls_t.tolist())))
    K_sketch = int(len(cls_t.tolist())/len(cls_list))
    K_model = int(len(cls_2.tolist())/len(cls_list))
    for i, cl in enumerate(cls_list):
        cmd_sub_t = Zt[K_sketch*i : K_sketch*(i+1)]
        cmd_sub_t = torch.mean(cmd_sub_t, 0)

        cmd_sub_2 = Z2[K_model*i : K_model*(i+1)]
        cmd_sub_2 = torch.mean(cmd_sub_2, 0)
        cmd_loss_sub = torch.norm(cmd_sub_t - cmd_sub_2, 2)
        cmd_loss += cmd_loss_sub

    return cmd_loss

