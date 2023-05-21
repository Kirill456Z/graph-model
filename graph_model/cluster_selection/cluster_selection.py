from torch import nn
import torch

def prepare_input(cur_embeddings, cur_transitions, first_stage_emb):
    return torch.cat([torch.Tensor(cur_embeddings),
                    torch.Tensor(cur_transitions),
                    torch.Tensor(first_stage_emb)])

class PredictNextCluster(nn.Module):
    def __init__(self, emb_size, hid_size = 256, outp_size = 11):
        super().__init__()
        self.l1 = nn.Linear(emb_size, hid_size)
        self.act = nn.SiLU()
        self.l2 = nn.Linear(hid_size, outp_size)
        self.dropout = nn.Dropout(0.15)
    
    def forward(self, emb):
        x = self.l1(emb)
        x = self.act(x)
        x = self.dropout(x)
        x = self.l2(x)
        return x
