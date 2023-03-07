import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
class ConLoss(nn.Module):
    def __init__(self):
        super(ConLoss, self).__init__()
        self.kldivloss = torch.nn.KLDivLoss(reduction = 'batchmean')

    def forward(self, inputs_q, inputs_k):
        normalized_inputs_q = inputs_q / torch.norm(inputs_q, dim=1, keepdim=True)
        normalized_inputs_k = inputs_k / torch.norm(inputs_k, dim=1, keepdim=True)
        # Compute similarity matrix
        sim_mat_g2s = torch.matmul(normalized_inputs_q, normalized_inputs_k.t())
        sim_mat_s2g = torch.matmul(normalized_inputs_k, normalized_inputs_q.t())
        sim_mat_g2s = F.softmax(sim_mat_g2s,dim = 1)
        sim_mat_s2g = F.softmax(sim_mat_s2g,dim = 1)
        loss1 = self.kldivloss(sim_mat_g2s.log(), sim_mat_s2g.detach())
        loss2 = self.kldivloss(sim_mat_s2g.log(), sim_mat_g2s.detach())
        loss = 0.5 * (loss1 + loss2)
        return loss
