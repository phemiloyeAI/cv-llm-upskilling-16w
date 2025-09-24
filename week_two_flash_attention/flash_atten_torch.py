# x: b x seq_len x dim 
# w_q, w_k, w_v 
# S = (Q @ K^T) V 
# we divide the q, k, and v into blocks of equal sizes such that
# we can perform the matrix multiplication on each block.
# so, what's the end goal? to have O which is  PxV, where P is the softmax of each block (rowwise),
# that means we can initialize O matrix, compute each block PxV and copy the result to the correct 
# position in O

import torch
import torch.nn as nn
from math import sqrt 
from torch.functional import F

def flash_attention_forward(
        Q: torch. Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        Bc: int = 4,
        Br: int = 4
):
    
    N, D = Q.shape  # 16 x 128 

    assert (N % Br == 0) and (N % Bc == 0), (
    f"Seq len {N} must be divisible by both Br={Br} and Bc={Bc}"
    )

    O, L = torch.zeros(Q.shape), torch.zeros(N, 1)
    scale = 1.0 / sqrt(D)

    # print(O.shape, L.shape)
    for i in range(0, N, Br):
        q_block = Q[i: i + Br, :] # chunk: 4 x 128 

        # initialize O, l and m
        # O = PV; P = softmax(q_block @ k_block.T)
        len_q = len(q_block)

        O_blk, l_blk = torch.zeros_like(q_block), torch.zeros(len_q, 1)
        m_blk = torch.full((len_q, 1), fill_value=float('-inf'))

        for j in range(0, N, Bc):
            # load k and v blocks 
            k_block = K[j: j + Bc, :] # K.T: chunk 128 x 4 
            v_block = V[j: j + Bc, :]

            S = (q_block @ k_block.T) * scale 

            row_max = torch.max(S, dim=1)[0]
            m_new = torch.max(m_blk.squeeze(), row_max).unsqueeze(1)

            cf = torch.exp(m_blk - m_new)
            P = torch.exp(S - m_new)

            O_blk = cf * O_blk  + P @ v_block
            l_blk = cf * l_blk + torch.sum(P, dim=1).reshape(-1, 1)
            m_blk = m_new
       
        # normalize PV by l 
        O_blk = O_blk / l_blk

        # write o_block and l_block to O and L
        O[i: i + Br, :] = O_blk
        L[i: i + Br] = l_blk 
    
    return O, L 

class FlashAttention(nn.Module):
    def __init__(self, bc, br): 
        self.bc = bc 
        self.br = br 
        super().__init__()

    def forward(self, q, k, v):
        return flash_attention_forward(
            q, k, v,
            self.bc, self.br
        )
    
def check_correct(flash_out, sdpa_out, atol=1e-4, rtol=1e-3):
    with torch.no_grad():
        max_abs = (flash_out - sdpa_out).abs().max().item()
        max_rel = ((flash_out - sdpa_out).abs() / (sdpa_out.abs() + 1e-8)).max().item()
    return max_abs, max_rel


def sdpa(Q, K, V):
    return F.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=False)

if __name__ == "__main__":
    x = torch.randn(3, 16, 128)
    q, k, v = x.unbind(0)

    flash_attn = FlashAttention(4, 4)
    flash_out = flash_attn(q, k, v)[0]

    sdpa_out = sdpa(q, k, v)

    max_abs, max_rel = check_correct(flash_out, sdpa_out)
    print(f"max_abs={max_abs:.3e}  max_rel={max_rel:.3e}")


