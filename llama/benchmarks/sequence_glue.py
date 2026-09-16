"""Ordinary two-stage argmax with first-index and first-NaN semantics."""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def partial_max(X,V,I,N:tl.constexpr,B:tl.constexpr):
    offsets=tl.program_id(0)*B+tl.arange(0,B)
    valid=offsets<N
    x=tl.load(X+offsets,valid,other=-float('inf')).to(tl.float32)
    nan=valid & (x!=x)
    any_nan=tl.max(nan.to(tl.int32),0)
    value=tl.max(tl.where(nan,-float('inf'),x),0)
    selected=valid & tl.where(any_nan!=0,nan,x==value)
    index=tl.min(tl.where(selected,offsets,2147483647),0)
    tl.store(V+tl.program_id(0),tl.where(any_nan!=0,float('nan'),value))
    tl.store(I+tl.program_id(0),index)

@tr.jit
def final_max(V,I,Y,N:tl.constexpr,B:tl.constexpr):
    offsets=tl.arange(0,B)
    valid=offsets<N
    x=tl.load(V+offsets,valid,other=-float('inf'))
    indices=tl.load(I+offsets,valid,other=2147483647)
    nan=valid & (x!=x)
    any_nan=tl.max(nan.to(tl.int32),0)
    value=tl.max(tl.where(nan,-float('inf'),x),0)
    selected=valid & tl.where(any_nan!=0,nan,x==value)
    index=tl.min(tl.where(selected,indices,2147483647),0)
    tl.store(Y,index.to(tl.int64))

def prepare_argmax(logits):
    """Allocate input-independent scratch on the logits device."""
    n=logits.numel();groups=tr.cdiv(n,1024)
    return torch.empty(groups,device=logits.device,dtype=torch.float32),torch.empty(groups,device=logits.device,dtype=torch.int32)

def argmax_out(logits,out,scratch):
    """Write a scalar int64 argmax without allocating or synchronizing."""
    values,indices=scratch
    partial_max[(values.numel(),)](logits,values,indices,logits.numel(),1024,num_warps=4)
    final_max[(1,)](values,indices,out,values.numel(),tr.next_power_of_2(values.numel()),num_warps=4)
