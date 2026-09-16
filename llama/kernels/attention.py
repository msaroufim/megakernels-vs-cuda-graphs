"""Phase3 attention with explicit eager-versus-SDPA-style arithmetic boundaries.

The caller's normal input-reset operation is a completion boundary at each
decode step. Past KV is ready before this step; only its current slot and Q
are produced by the immediately preceding projection.
"""
import triton
import triton.language as tl


@triton.jit
def attend(Q,K,V,O,POS:tl.constexpr,T:tl.constexpr,D:tl.constexpr,PDL:tl.constexpr,EAGER:tl.constexpr):
    if PDL:
        tl.inline_asm_elementwise('griddepcontrol.launch_dependents; mov.u32 $0, 0;',constraints='=r,~{memory}',args=[],dtype=tl.int32,is_pure=False,pack=1)
    h=tl.program_id(0);tile=tl.program_id(1)
    t=tl.arange(0,T);d=tl.arange(0,64);od=tile*D+tl.arange(0,D)
    past_k=tl.load(K+t[:,None]*512+(h//4)*64+d[None,:],t[:,None]<POS,other=0)
    past_v=tl.load(V+t[:,None]*512+(h//4)*64+od[None,:],t[:,None]<POS,other=0)
    if PDL:
        tl.inline_asm_elementwise('griddepcontrol.wait; mov.u32 $0, 0;',constraints='=r,~{memory}',args=[],dtype=tl.int32,is_pure=False,pack=1)
    q=tl.load(Q+h*64+d).to(tl.float32)
    current_k=tl.load(K+POS*512+(h//4)*64+d)
    current_v=tl.load(V+POS*512+(h//4)*64+od)
    k=tl.where(t[:,None]==POS,current_k[None,:],past_k).to(tl.float32)
    v=tl.where(t[:,None]==POS,current_v[None,:],past_v).to(tl.float32)
    score=tl.sum(k*q[None,:],axis=1)
    if EAGER:
        score=score.to(tl.bfloat16).to(tl.float32)
    score=score*0.125
    if EAGER:
        score=score.to(tl.bfloat16).to(tl.float32)
    score=tl.where(t<=POS,score,-float('inf'))
    probability=tl.exp(score-tl.max(score,axis=0))
    probability=probability/tl.sum(probability,axis=0)
    if EAGER:
        probability=probability.to(tl.bfloat16).to(tl.float32)
    result=tl.sum(v*probability[:,None],axis=0)
    tl.store(O+h*64+od,result)


def make(q,k,v,out,position,d=32,warps=4,pdl=False,eager=False):
    """Build one causal decode call with explicitly chosen attention boundaries."""
    def run():
        attend[(32,64//d)](q,k,v,out,position,triton.next_power_of_2(position+1),d,pdl,eager,num_warps=warps,launch_pdl=pdl,enable_fp_fusion=False)
    return run
