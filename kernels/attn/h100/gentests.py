import torch
from tqdm import trange
import numpy as np
import sys
import math
from torch.nn.functional import scaled_dot_product_attention
# pip install tqdm tabulate git+https://github.com/fkodom/grouped-query-attention-pytorch.git
from grouped_query_attention_pytorch.attention import scaled_dot_product_gqa

# only generate a single batch/head of data, which makes file loading much faster.
# it does mean we'll have to check batch/head behavior separately later, but that should be much easier to debug.
B = 1
N = int(sys.argv[1])
D = int(sys.argv[2])

H_QO = int(sys.argv[3])
H_KV = int(sys.argv[4])

TILE_HEIGHT_Q = 192
TILE_HEIGHT_KV = 128
N_TILES_Q = N // TILE_HEIGHT_Q
N_TILES_KV = N // TILE_HEIGHT_KV

causal = False

def make_causal_mask(N):
    return torch.tril(torch.ones((1, 1, N, N), device='cuda', dtype=torch.bool))

def make_randn_mask(N):
    return torch.randint(0, 2, (1, 1, N, N), dtype=torch.bool, device='cuda')

def make_striped_mask(N):
    mask = torch.zeros((1, 1, N, N), device='cuda', dtype=torch.bool)
    mask[:, :, ::2, 1::2] = 1
    mask[:, :, 1::2, ::2] = 1
    return mask

def make_randomly_striped_mask(N):
    mask = torch.zeros((1, 1, N, N), device='cuda', dtype=torch.bool)
    sections = torch.randperm(N * N).reshape(N, N)
    sections = sections < (N * N // 4)
    mask[:, :, sections] = 1
    return mask

def make_ones_mask(N):
    return torch.ones((1, 1, N, N), device='cuda', dtype=torch.bool)

def bool_matrix_to_indices(blocksparsity: torch.Tensor) -> torch.Tensor:
    col_indices = torch.arange(N_TILES_KV, device=blocksparsity.device).unsqueeze(0).expand(N_TILES_Q, N_TILES_KV)
    placeholder = N_TILES_KV
    assigned_indices = torch.where(blocksparsity, col_indices, torch.full_like(col_indices, placeholder))
    sorted_indices, _ = torch.sort(assigned_indices, dim=1)
    blocksparsity_indices = torch.where(sorted_indices < N_TILES_KV, sorted_indices, torch.full_like(sorted_indices, -1))
    return blocksparsity_indices

torch.random.manual_seed(20)
q = (torch.empty((B, H_QO, N, D), dtype=torch.bfloat16, device='cuda').normal_(mean=0.0, std=0.5).requires_grad_())
k = (torch.empty((B, H_KV, N, D), dtype=torch.bfloat16, device='cuda').normal_(mean=0.0, std=0.5).requires_grad_())
v = (torch.empty((B, H_KV, N, D), dtype=torch.bfloat16, device='cuda').normal_(mean=0.0, std=0.5).requires_grad_())
grad_output = (torch.randn((B, H_QO, N, D), dtype=torch.bfloat16, device='cuda'))
blocksparsity = torch.rand((N_TILES_Q, N_TILES_KV), device="cuda") > 0.3
blocksparsity_indices = torch.arange(0, N_TILES_KV, device="cuda")[None, :].expand(N_TILES_Q, -1).where(blocksparsity, N_TILES_KV + 10).sort(dim=-1).values.to(torch.int16)
blocksparsity_indices = torch.where(blocksparsity_indices == N_TILES_KV + 10, -1, blocksparsity_indices)
breakpoint()

assert blocksparsity_indices.shape == blocksparsity.shape, "blocksparsity_indices shape does not match blocksparsity shape"
blocksparsity_indices += (blocksparsity_indices != -1).to(torch.int32) * 0
blocksparsity_indices = torch.min(blocksparsity_indices, torch.full_like(blocksparsity_indices, N_TILES_KV - 1))
print('Number of nonzero blocksparsity values:', blocksparsity.sum())

# pad seqlen to multiple of 128
o = scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=causal)
print('Expected output:', o[0][0][0][:15])
# o, _ = scaled_dot_product_gqa(
#     q.permute(0, 2, 1, 3).contiguous(),
#     k.permute(0, 2, 1, 3).contiguous(),
#     v.permute(0, 2, 1, 3).contiguous(),
#     is_causal=causal,
#     need_weights=False,
# )
# o = o.permute(0, 2, 1, 3).contiguous()

##########################################
### EXACT GQA COMPUTATION FROM LLAMA 3 ###
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


keys_l = repeat_kv(k.permute(0, 2, 1, 3), H_QO // H_KV).permute(0, 2, 1, 3)
values_l = repeat_kv(v.permute(0, 2, 1, 3), H_QO // H_KV).permute(0, 2, 1, 3)

scores = torch.matmul(q, keys_l.transpose(2, 3)) / math.sqrt(D)

mask = torch.full((N, N), float('-inf'), device=q.device, dtype=q.dtype)
mask = torch.triu(mask, diagonal=1)
if mask is not None and causal:
    scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
scores = torch.nn.functional.softmax(scores, dim=-1).type_as(q)
output = torch.matmul(scores, values_l)  # (bs, n_local_heads, seqlen, head_dim)
### EXACT GQA COMPUTATION FROM LLAMA 3 ###
##########################################

# now do backwards computations
o.backward(grad_output)

q_grad = q.grad
k_grad = k.grad
v_grad = v.grad
    
softmax_scale = 1 / math.sqrt(D)
l_vec = torch.empty((B, H_QO, N, N), dtype=torch.bfloat16, device=q.device)

for i in range(H_QO):
    group_idx = i // (H_QO // H_KV)
    l_vec[:, i] = torch.einsum("bnd,bmd->bnm", q[:, i], k[:, group_idx]) * softmax_scale

mask = torch.triu(torch.ones(N, N), diagonal=1).to('cuda').bool().unsqueeze(0).unsqueeze(0).expand(B, H_QO, -1, -1)
if causal:
    l_vec = l_vec.masked_fill(mask, float('-inf'))

max_vec = l_vec.max(dim=-1, keepdim=True).values
l_vec = l_vec - max_vec
l_vec = torch.exp(l_vec)
l_vec_sum = l_vec.sum(dim=-1, keepdim=True)
l_vec = max_vec + torch.log(l_vec_sum)

d_vec = torch.mul(o.to(torch.bfloat16), grad_output.to(torch.bfloat16))
d_vec = d_vec.to(torch.bfloat16).sum(dim=-1, keepdim=True)


print("--------------------------------------")
print("Q shape: ",      q.shape)
print("K shape: ",      k.shape)
print("V shape: ",      v.shape)
print("O shape: ",      o.shape)
print("Q grad shape: ", q_grad.shape)
print("K grad shape: ", k_grad.shape)
print("V grad shape: ", v_grad.shape)
print("L shape: ",      l_vec.shape)
print("D shape: ",      d_vec.shape)
print("--------------------------------------")

# print out avg magnitude of output tensor
print(f'Average magnitude of OUTPUT tensor: {o.abs().mean()}')
print(f'1/100 magnitude of OUTPUT tensor:   {o.abs().mean()/100}')
print(f'Average magnitude of Q_GRAD tensor: {q_grad.abs().mean()}')
print(f'1/100 magnitude of Q_GRAD tensor:   {q_grad.abs().mean()/100}')
print(f'Average magnitude of K_GRAD tensor: {k_grad.abs().mean()}')
print(f'1/100 magnitude of K_GRAD tensor:   {k_grad.abs().mean()/100}')
print(f'Average magnitude of V_GRAD tensor: {v_grad.abs().mean()}')
print(f'1/100 magnitude of V_GRAD tensor:   {v_grad.abs().mean()/100}')
print(f'Average magnitude of L tensor:      {l_vec.abs().mean()}')
print(f'1/100 magnitude of L tensor:        {l_vec.abs().mean()/100}')
print(f'Average magnitude of D tensor:      {d_vec.abs().mean()}')
print(f'1/100 magnitude of D tensor:        {d_vec.abs().mean()/100}')
print("--------------------------------------")

filename = f'randn_{N}N_{D}D_{H_QO}QO_{H_KV}KV'

if causal:
    filename += '_causal'
if H_QO != H_KV:
    filename += '_gqa'

filename += '.txt'

with open(filename, 'w') as f:
    tensors = [blocksparsity, blocksparsity_indices, q, k, v, o, l_vec, d_vec, grad_output, q_grad, k_grad, v_grad]
    bs_idx = 0
    bs_ind_idx = 1
    for i, tensor in enumerate(tensors):
        print(f'Writing tensor {i} of {len(tensors)}')
        array = tensor.to(torch.float32).flatten().detach().cpu().numpy()
        if i == bs_idx:
            print('# of nonzero blocksparsity values:', (array != 0).sum(), 'out of', array.size)
        elif i == bs_ind_idx:
            array = tensor.to(torch.int32).flatten().detach().cpu().numpy()
            print('# of nonzero blocksparsity indices:', (array != -1).sum(), 'out of', array.size)
        f.write(' '.join(map(str, array)) + ' \n')
