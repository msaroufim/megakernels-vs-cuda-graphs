#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
using bf=__nv_bfloat16;
// Both branches accumulate in FP32 after exact BF16-to-FP32 operand conversion.
// Native mixed FMA is optional Blackwell code. The default is portable fmaf.
__device__ __forceinline__ float mixed_dot2(float c,unsigned a,unsigned b){
#ifdef USE_NATIVE_BF16_FMA
 asm("{.reg .b16 a0,a1,b0,b1; mov.b32 {a0,a1},%1; mov.b32 {b0,b1},%2; fma.rn.f32.bf16 %0,a0,b0,%0; fma.rn.f32.bf16 %0,a1,b1,%0;}" : "+f"(c):"r"(a),"r"(b));
#else
 auto av=__bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&a));
 auto bv=__bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&b));
 c=fmaf(av.x,bv.x,c);
 c=fmaf(av.y,bv.y,c);
#endif
 return c;
}
// Prepacked [output_block, k_vector, row_in_block, 8] BF16 weights.
// Weight vectors are prefetched before PDL waits; activations remain dependent.
template<int K, int MODE, bool PDL>
__global__ void projection(const bf* x,const bf* w,const bf* w2,const bf* norm,
                          bf* y,bf* cachek,bf* cachev,const float* cos,const float* sin,
                          int n,int pos,int maxlen,float eps) {
  constexpr int ROWS=2, LANES=128/ROWS, PAIRS=K/(2*LANES);
  int tid=threadIdx.x, r=blockIdx.x*ROWS+tid%ROWS, lane=tid/ROWS;
  uint4 packed[PAIRS/4], packed2[PAIRS/4],activation[PAIRS/4],normalizer[PAIRS/4];
  if constexpr(PDL) cudaTriggerProgrammaticLaunchCompletion();
  #pragma unroll
  for(int i=0;i<PAIRS/4;i++) {
    if constexpr(MODE==0||MODE==2||MODE==4)normalizer[i]=reinterpret_cast<const uint4*>(norm)[lane+i*LANES];
    packed[i]=r<n?reinterpret_cast<const uint4*>(w)[blockIdx.x*(K/8)*ROWS+(lane+i*LANES)*ROWS+tid%ROWS]:make_uint4(0,0,0,0);
    if constexpr((MODE==2||MODE==6)) packed2[i]=r<n?reinterpret_cast<const uint4*>(w2)[blockIdx.x*(K/8)*ROWS+(lane+i*LANES)*ROWS+tid%ROWS]:make_uint4(0,0,0,0);
  }
  // asm dependency prevents the compiler sinking the independent prefetch.
  #pragma unroll
  for(int i=0;i<PAIRS/4;i++) {asm volatile(""::"r"(packed[i].x),"r"(packed[i].y),"r"(packed[i].z),"r"(packed[i].w):"memory");if constexpr((MODE==2||MODE==6)) asm volatile(""::"r"(packed2[i].x),"r"(packed2[i].y),"r"(packed2[i].z),"r"(packed2[i].w):"memory");}
  #pragma unroll
  for(int i=0;i<PAIRS/4;i++)if constexpr(MODE==0||MODE==2||MODE==4)asm volatile(""::"r"(normalizer[i].x),"r"(normalizer[i].y),"r"(normalizer[i].z),"r"(normalizer[i].w):"memory");
  if constexpr(PDL) cudaGridDependencySynchronize();
  __shared__ float sums[256], gates[256], dots[16];

  float scale=1.f;
  if constexpr(MODE==0||MODE==2||MODE==4){
    float square=0;
    #pragma unroll
    for(int i=0;i<PAIRS/4;i++){
      activation[i]=reinterpret_cast<const uint4*>(x)[lane+i*LANES];
      #pragma unroll
      for(int j=0;j<4;j++){auto a=reinterpret_cast<__nv_bfloat162*>(&activation[i])[j];square=mixed_dot2(square,*reinterpret_cast<unsigned*>(&a),*reinterpret_cast<unsigned*>(&a));}
    }
    for(int delta=16;delta>=ROWS;delta/=2)square+=__shfl_down_sync(0xffffffff,square,delta);
    if(tid%32<ROWS)sums[(tid/32)*ROWS+tid%ROWS]=square;
    __syncthreads();
    if(tid<ROWS){float total=0;for(int z=0;z<4;z++)total+=sums[tid+z*ROWS];dots[tid]=rsqrtf(total/K+eps);}
    __syncthreads();scale=dots[tid%ROWS];
  }
  float acc=0,gate=0;
  #pragma unroll
  for(int i=0;i<PAIRS/4;i++) {
   uint4 xpvec;if constexpr(MODE==0||MODE==2||MODE==4)xpvec=activation[i];else xpvec=reinterpret_cast<const uint4*>(x)[lane+i*LANES];
   asm volatile(""::"r"(xpvec.x),"r"(xpvec.y),"r"(xpvec.z),"r"(xpvec.w));
   #pragma unroll
   for(int j=0;j<4;j++){
    auto wp=reinterpret_cast<__nv_bfloat162*>(&packed[i])[j];
    auto xp=reinterpret_cast<const __nv_bfloat162*>(&xpvec)[j];
    if constexpr(MODE==0||MODE==2||MODE==4){float2 a=__bfloat1622float2(xp);float2 n=__bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(&normalizer[i])[j]);xp=__floats2bfloat162_rn((a.x*scale)*n.x,(a.y*scale)*n.y);}
    acc=mixed_dot2(acc,*reinterpret_cast<unsigned*>(&wp),*reinterpret_cast<unsigned*>(&xp));
    if constexpr((MODE==2||MODE==6)){auto gp=reinterpret_cast<__nv_bfloat162*>(&packed2[i])[j];gate=mixed_dot2(gate,*reinterpret_cast<unsigned*>(&gp),*reinterpret_cast<unsigned*>(&xp));}
   }
  }
  for(int d=16;d>=ROWS;d/=2){acc+=__shfl_down_sync(0xffffffff,acc,d);if constexpr((MODE==2||MODE==6))gate+=__shfl_down_sync(0xffffffff,gate,d);}
  if(tid%32<ROWS){sums[(tid/32)*ROWS+tid%ROWS]=acc;if constexpr((MODE==2||MODE==6))gates[(tid/32)*ROWS+tid%ROWS]=gate;}
  __syncthreads();
  if(tid<ROWS&&r<n) {
    float value=0,g=0;
    #pragma unroll
    for(int z=0;z<4;z++){value+=sums[tid+z*ROWS];if constexpr((MODE==2||MODE==6))g+=gates[tid+z*ROWS];}
    if constexpr(MODE==0) dots[tid]=value;
    else if constexpr((MODE==2||MODE==6)){float silu=g/(1.f+expf(-g));y[r]=__float2bfloat16(value*silu);}
    else if constexpr(MODE==1||MODE==3)y[r]=__float2bfloat16(value+__bfloat162float(y[r]));
    else if constexpr(MODE==5)reinterpret_cast<float*>(y)[r]=value;
    else y[r]=__float2bfloat16(value);
  }
  if constexpr(MODE==0){
    __syncthreads();
    if(tid<ROWS&&r<n){float v=dots[tid]; if(r<2560){
        // Retain projection/trigonometric products in FP32 until the BF16 store.
        int d=r%64;
        float rotated=r%2?dots[tid^1]:-dots[tid^1];
        v=v*cos[pos*64+d]+rotated*sin[pos*64+d];
      }
      if(r<2048)y[r]=__float2bfloat16(v);
      else if(r<2560)cachek[pos*512+r-2048]=__float2bfloat16(v);
      else cachev[pos*512+r-2560]=__float2bfloat16(v);
    }
  }
}
template<int K,int MODE,bool PDL>
void proj(torch::Tensor x,torch::Tensor w,torch::Tensor w2,torch::Tensor norm,torch::Tensor y,torch::Tensor ck,torch::Tensor cv,torch::Tensor cos,torch::Tensor sin,int pos,double eps){
 cudaLaunchConfig_t cfg{};constexpr int rows=2;cfg.gridDim=dim3((w.size(0)+rows-1)/rows);cfg.blockDim=dim3(128);cfg.stream=at::cuda::getCurrentCUDAStream();
 cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;attr.val.programmaticStreamSerializationAllowed=1;if constexpr(PDL){cfg.attrs=&attr;cfg.numAttrs=1;}
 auto error=cudaLaunchKernelEx(&cfg,projection<K,MODE,PDL>,(const bf*)x.data_ptr(),(const bf*)w.data_ptr(),(const bf*)w2.data_ptr(),(const bf*)norm.data_ptr(),(bf*)y.data_ptr(),(bf*)ck.data_ptr(),(bf*)cv.data_ptr(),cos.data_ptr<float>(),sin.data_ptr<float>(),(int)w.size(0),pos,(int)ck.size(0),(float)eps);
 TORCH_CHECK(error==cudaSuccess,cudaGetErrorString(error));
}
void project(torch::Tensor x,torch::Tensor w,torch::Tensor w2,torch::Tensor norm,torch::Tensor y,torch::Tensor ck,torch::Tensor cv,torch::Tensor cos,torch::Tensor sin,int pos,double eps,int mode,bool pdl){
 TORCH_CHECK(y.scalar_type()==(mode==5 ? torch::kFloat32 : torch::kBFloat16),"head requires FP32 output; other modes BF16");
 TORCH_CHECK(x.is_cuda()&&x.scalar_type()==torch::kBFloat16,"CUDA BF16 required");
 TORCH_CHECK(cos.is_cuda()&&sin.is_cuda()&&cos.scalar_type()==torch::kFloat32&&sin.scalar_type()==torch::kFloat32,"FP32 CUDA RoPE tables required");
 TORCH_CHECK(pos>=0&&pos<ck.size(0)&&pos<cos.size(0)&&pos<sin.size(0),"position outside cache/RoPE allocation");
 #define CALL(M,K) if(pdl)proj<K,M,true>(x,w,w2,norm,y,ck,cv,cos,sin,pos,eps);else proj<K,M,false>(x,w,w2,norm,y,ck,cv,cos,sin,pos,eps)
 switch(mode){case 0:{CALL(0,2048);break;}case 1:{CALL(1,2048);break;}case 2:{CALL(2,2048);break;}case 3:{CALL(3,8192);break;}case 4:{CALL(4,2048);break;}case 5:{CALL(5,2048);break;}case 6:{CALL(6,2048);break;}default:TORCH_CHECK(false,"bad mode");}
}
template<bool PDL>
__global__ void head_norm_kernel(const bf* x,const bf* norm,bf* y,float eps){
 int tid=threadIdx.x,lane=tid/2;
 uint4 nv[4],xv[4];
 if constexpr(PDL)cudaTriggerProgrammaticLaunchCompletion();
 #pragma unroll
 for(int i=0;i<4;i++)nv[i]=reinterpret_cast<const uint4*>(norm)[lane+i*64];
 if constexpr(PDL)cudaGridDependencySynchronize();
 float square=0;
 #pragma unroll
 for(int i=0;i<4;i++){
  xv[i]=reinterpret_cast<const uint4*>(x)[lane+i*64];
  #pragma unroll
  for(int j=0;j<4;j++){auto a=reinterpret_cast<__nv_bfloat162*>(&xv[i])[j];square=mixed_dot2(square,*reinterpret_cast<unsigned*>(&a),*reinterpret_cast<unsigned*>(&a));}
 }
 for(int delta=16;delta>=2;delta/=2)square+=__shfl_down_sync(0xffffffff,square,delta);
 __shared__ float sums[8],scale[2];
 if(tid%32<2)sums[(tid/32)*2+tid%2]=square;
 __syncthreads();
 if(tid<2){float total=0;for(int z=0;z<4;z++)total+=sums[tid+z*2];scale[tid]=rsqrtf(total/2048+eps);}
 __syncthreads();
 if(tid%2==0){
  #pragma unroll
  for(int i=0;i<4;i++){
   uint4 result;
   #pragma unroll
   for(int j=0;j<4;j++){
    auto a=__bfloat1622float2(reinterpret_cast<__nv_bfloat162*>(&xv[i])[j]);
    auto n=__bfloat1622float2(reinterpret_cast<__nv_bfloat162*>(&nv[i])[j]);
    reinterpret_cast<__nv_bfloat162*>(&result)[j]=__floats2bfloat162_rn((a.x*scale[0])*n.x,(a.y*scale[0])*n.y);
   }
   reinterpret_cast<uint4*>(y)[lane+i*64]=result;
  }
 }
}
void head_norm(torch::Tensor x,torch::Tensor norm,torch::Tensor y,double eps,bool pdl){
 cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(1);cfg.blockDim=dim3(128);cfg.stream=at::cuda::getCurrentCUDAStream();
 cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;attr.val.programmaticStreamSerializationAllowed=1;
 if(pdl){cfg.attrs=&attr;cfg.numAttrs=1;}
 cudaError_t error;
 if(pdl)error=cudaLaunchKernelEx(&cfg,head_norm_kernel<true>,(const bf*)x.data_ptr(),(const bf*)norm.data_ptr(),(bf*)y.data_ptr(),(float)eps);
 else error=cudaLaunchKernelEx(&cfg,head_norm_kernel<false>,(const bf*)x.data_ptr(),(const bf*)norm.data_ptr(),(bf*)y.data_ptr(),(float)eps);
 TORCH_CHECK(error==cudaSuccess,cudaGetErrorString(error));
}

template<int SPLIT,bool PDL>
__global__ void down_partial_kernel(const bf* x,const bf* w,float* partial,int n){
 constexpr int CHUNK=8192/SPLIT,LANES=64,VECS=CHUNK/8/LANES;
 int tid=threadIdx.x,lane=tid/2,r=blockIdx.x*2+tid%2,chunk=blockIdx.y;
 uint4 packed[VECS];
 if constexpr(PDL)cudaTriggerProgrammaticLaunchCompletion();
 #pragma unroll
 for(int i=0;i<VECS;i++)packed[i]=reinterpret_cast<const uint4*>(w)[blockIdx.x*(8192/8)*2+(chunk*(CHUNK/8)+lane+i*LANES)*2+tid%2];
 #pragma unroll
 for(int i=0;i<VECS;i++)asm volatile(""::"r"(packed[i].x),"r"(packed[i].y),"r"(packed[i].z),"r"(packed[i].w):"memory");
 if constexpr(PDL)cudaGridDependencySynchronize();
 float acc=0;
 #pragma unroll
 for(int i=0;i<VECS;i++){
  uint4 xv=reinterpret_cast<const uint4*>(x)[chunk*(CHUNK/8)+lane+i*LANES];
  #pragma unroll
  for(int j=0;j<4;j++)acc=mixed_dot2(acc,reinterpret_cast<unsigned*>(&packed[i])[j],reinterpret_cast<unsigned*>(&xv)[j]);
 }
 for(int delta=16;delta>=2;delta/=2)acc+=__shfl_down_sync(0xffffffff,acc,delta);
 __shared__ float sums[8];
 if(tid%32<2)sums[(tid/32)*2+tid%2]=acc;
 __syncthreads();
 if(tid<2){float result=0;for(int z=0;z<4;z++)result+=sums[tid+z*2];partial[chunk*n+r]=result;}
}
template<int SPLIT,bool PDL>
__global__ void down_reduce_kernel(const float* partial,bf* residual,int n){
 if constexpr(PDL)cudaTriggerProgrammaticLaunchCompletion();
 if constexpr(PDL)cudaGridDependencySynchronize();
 int r=blockIdx.x*blockDim.x+threadIdx.x;
 if(r<n){float sum=0;for(int chunk=0;chunk<SPLIT;chunk++)sum+=partial[chunk*n+r];residual[r]=__float2bfloat16(sum+__bfloat162float(residual[r]));}
}
template<int SPLIT,bool PDL>
void down_split_impl(torch::Tensor x,torch::Tensor w,torch::Tensor partial,torch::Tensor residual){
 cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(w.size(0)/2,SPLIT);cfg.blockDim=dim3(128);cfg.stream=at::cuda::getCurrentCUDAStream();
 cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;attr.val.programmaticStreamSerializationAllowed=1;
 if constexpr(PDL){cfg.attrs=&attr;cfg.numAttrs=1;}
 auto error=cudaLaunchKernelEx(&cfg,down_partial_kernel<SPLIT,PDL>,(const bf*)x.data_ptr(),(const bf*)w.data_ptr(),(float*)partial.data_ptr(),(int)w.size(0));
 TORCH_CHECK(error==cudaSuccess,cudaGetErrorString(error));
 cfg.gridDim=dim3((w.size(0)+255)/256);cfg.blockDim=dim3(256);
 error=cudaLaunchKernelEx(&cfg,down_reduce_kernel<SPLIT,PDL>,(const float*)partial.data_ptr(),(bf*)residual.data_ptr(),(int)w.size(0));
 TORCH_CHECK(error==cudaSuccess,cudaGetErrorString(error));
}
void down_split(torch::Tensor x,torch::Tensor w,torch::Tensor partial,torch::Tensor residual,bool pdl){
 TORCH_CHECK(x.numel()==8192&&w.size(0)==2048&&w.size(1)==8192&&partial.numel()==4*2048,"Unexpected down split shape");
 if(pdl)down_split_impl<4,true>(x,w,partial,residual);else down_split_impl<4,false>(x,w,partial,residual);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("project",&project);m.def("down_split",&down_split);m.def("head_norm",&head_norm);}
