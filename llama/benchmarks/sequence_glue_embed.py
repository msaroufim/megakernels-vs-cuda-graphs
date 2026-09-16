"""Two-stage greedy selection with optional direct next-token embedding."""
import triton as tr
import triton.language as tl
from sequence_glue import partial_max,prepare_argmax

@tr.jit
def final_max_embed(V,I,Y,W,H,N:tl.constexpr,B:tl.constexpr,D:tl.constexpr,EMBED:tl.constexpr):
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
    if EMBED:
        columns=tl.arange(0,D)
        embedding=tl.load(W+index.to(tl.int64)*D+columns)
        tl.store(H+columns,embedding)

def argmax_embed_out(logits,out,scratch,weight,hidden,embed_next):
    """Write token and optionally its full embedding, preserving all logits."""
    values,indices=scratch
    partial_max[(values.numel(),)](logits,values,indices,logits.numel(),1024,num_warps=4)
    final_max_embed[(1,)](values,indices,out,weight,hidden,values.numel(),tr.next_power_of_2(values.numel()),hidden.numel(),embed_next,num_warps=4)
