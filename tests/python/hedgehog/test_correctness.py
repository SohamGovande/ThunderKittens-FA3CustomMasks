import torch
torch.random.manual_seed(42)
from tqdm import trange
import sys
import itertools
import thunderkittens as tk

import torch.nn.functional as F

D_QK = 128
D_VO = 128

def pytorch_softmax_gt(x, map_mat, label=None):

    x = torch.einsum('bhmd,hdn->bhmn', x, map_mat)

    if label is not None:
        with open(label, 'w') as f:
            for row in x[0,0,0:256]:
                f.write(' '.join(map(lambda k: f"{k:8.4f}", row[:256].tolist())) + '\n')
    
    x_pos = x
    x_neg = -x
    
    x_pos_max = torch.amax(x_pos, dim=-1, keepdim=True)
    x_neg_max = torch.amax(x_neg, dim=-1, keepdim=True)
    
    x_pos = x_pos - x_pos_max
    x_neg = x_neg - x_neg_max
    
    x_pos_num = torch.exp(x_pos)
    x_pos_den = torch.sum(torch.exp(x_pos), dim=-1, keepdim=True)
    
    x_neg_num = torch.exp(x_neg)
    x_neg_den = torch.sum(torch.exp(x_neg), dim=-1, keepdim=True)
    
    x_pos = x_pos_num / x_pos_den
    x_neg = x_neg_num / x_neg_den
    
    x = torch.cat([x_pos, x_neg], dim=-1).clamp(min=1e-6)
    
    return x

generator_mat = torch.block_diag(*[torch.ones((64,64), device='cuda')]*256) # 16384 x 16384 should be big enough for most porpoises
generator_mat += torch.roll(generator_mat, -64, -1) # this adds the terracing
lin_mask = torch.tril(1-generator_mat).reshape((1,1,16384,16384))
exp_mask = torch.tril(generator_mat).reshape((1,1,16384,16384))
exp_mask = 10000*exp_mask - 10000 # we actually want to effectively subtract infinity instead with this mask, since it should happen pre-exp

def pytorch_test(Q, K, V, Qmap, Kmap, alphas, betas):
    
    Qs = pytorch_softmax_gt(Q, Qmap)
    Ks = pytorch_softmax_gt(K, Kmap)
    
    a_lin = torch.einsum('bhmd,bhnd->bhmn', Qs, Ks).to(torch.float32)
    a_exp = torch.einsum('bhmd,bhnd->bhmn', Q, K).to(torch.float32)
    # mask
    a_lin *= lin_mask[:,:,:a_lin.shape[2], :a_lin.shape[3]] * alphas.reshape((1,-1,1,1)) # zero out unwanted entries
    a_exp += exp_mask[:,:,:a_exp.shape[2], :a_exp.shape[3]] # subtract infinity off of unwanted entries
    a_exp -= a_exp.amax(dim=-1, keepdim=True)
    a_exp = torch.exp(a_exp / (128**.5)) * betas.reshape((1,-1,1,1))

    a = a_exp + a_lin
    a = (a / (a.sum(dim=-1, keepdim=True)+1e-6)).to(torch.bfloat16) # normalize
    
    out = torch.einsum('bhmn,bhnd->bhmd', a, V).to(torch.bfloat16)
    
    kv_state = torch.einsum('bhlf,bhld->bhfd', Ks[:,:,:-64,:], V[:,:,:-64,:]).to(torch.float32).detach()
    k_state  = Ks[:,:,:-64,:].to(torch.float32).sum(dim=-2).detach()
    
    return out, kv_state, k_state

