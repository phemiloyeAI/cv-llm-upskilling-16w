import torch
import triton
import triton.language as tl
from typing import Optional

# softmax statistics is calculated along the col axis (row based)
# grid: create multiple warps (threads) to calculate each row softmax in parallel (independently)
# unlike torch tensors where we slice, everything in triton is a pointer so we need a way to know which data we are pointing to or need to access. 


@triton.jit
def _softmax_forwd_kernel(
        x_out_ptr,
        x_ptr,
        row_stride,
        num_cols, 
        block_size: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_offset = row_stride * row_idx
    col_offset = tl.arange(0, block_size) 

    block_ptr = x_ptr + row_offset + col_offset
    mask = col_offset < num_cols

    # load x onto sram
    x = tl.load(block_ptr, mask=mask, other=float('-inf'))

    x = x - tl.max(x, axis=0)
    x = tl.exp(x)
    x = x / tl.sum(x, axis=0) 

    out_block_ptr = x_out_ptr + row_offset + col_offset
    
    tl.store(out_block_ptr, x, mask=mask) # write results back to the HBM
    
def softmax_forwd(
        x: torch.Tensor
    ):
    
    x_out = torch.empty_like(x) # where to store the softmax(x)
    n_rows, n_cols = x.shape
    grid = (n_rows, )
    
    block_size = triton.next_power_of_2(n_cols)
    
    _softmax_forwd_kernel[grid](
        x_out,
        x, 
        x.stride(0),
        n_cols,
        block_size
    )

    return x_out


