"""Export pre-RMS BF16 values from the reference head's existing HC computation.

Private equal-work helper, installed before capture. Its extra 40,960-byte
write is part of the modified reference's timed work.
"""

import ast
import hashlib
import inspect
from pathlib import Path
from types import FunctionType, SimpleNamespace

PINS = {
    "warp_sum": "61f9dce3b560261ca2e849931c0e61f74ac6df1667006801b316de78e7fa3432",
    "kernel": "c76d46a48c8304baa9cc45d78cd27da7859df35ffd851b910f38092f18df4610",
    "launcher": "8e29f46d429e1ba19eb34f1189b4abae1bfaf8ac9e3c55aef9654c0adcbd6b69",
    "target_hc_head_rmsnorm_cuda": (
        "442b486047d3df6ea5c38e93cc1804bb6d4b31d0f3155ef6e67fd471cc459012"
    ),
    "_load_route_pack_extension": (
        "2bfc73fb466b86a278b52eda3f96da7a4309f936a7a55da0980933e3a103e81c"
    ),
}
CFLAGS = ["-O3", "-std=c++17", "-DNDEBUG"]
CUDA_FLAGS = [
    "-O3",
    "--std=c++17",
    "-DNDEBUG",
    "-lineinfo",
    "-gencode=arch=compute_103a,code=sm_103a",
]
OLD = "target_hc_head_rmsnorm_kernel"
NEW = "equal_work_hc_head_rmsnorm_kernel"
HOST = "dspark_equal_work_hc_head_rmsnorm_cuda"


def digest(value):
    """Hash source text or exact artifact bytes."""
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def cpp_function(text, start):
    """Extract a pinned function whose body has no braces in literals/comments."""
    if text.count(start) != 1:
        raise ValueError(f"Ambiguous head source: {start}")
    begin = text.index(start)
    end, depth = text.index("{", begin) + 1, 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[begin:end]


def replace_once(text, before, after):
    """Reject source drift instead of applying an approximate numerical patch."""
    if text.count(before) != 1:
        raise ValueError(f"Head export anchor changed: {before}")
    return text.replace(before, after)


def source_for(wrapper_file):
    """Extract only the authenticated HC leaf, reduction and host validation."""
    wrapper_file = Path(wrapper_file)
    path = wrapper_file.parent / "csrc/dspark_route_pack_kernel.cu"
    raw, python = path.read_text(), wrapper_file.read_text()
    blocks = {
        "warp_sum": cpp_function(raw, "__device__ __forceinline__ float warp_sum("),
        "kernel": cpp_function(raw, f"__global__ __launch_bounds__(1024, 1) void {OLD}("),
        "launcher": cpp_function(raw, "void dspark_target_hc_head_rmsnorm_cuda("),
    }
    for node in ast.parse(python).body:
        if isinstance(node, ast.FunctionDef) and node.name in PINS:
            blocks[node.name] = ast.get_source_segment(python, node)
    if {name: digest(value) for name, value in blocks.items()} != PINS:
        raise ValueError("Reference HC source or wrapper differs from the qualified source")
    kernel = replace_once(blocks["kernel"], OLD, NEW)
    kernel = replace_once(
        kernel,
        "    __nv_bfloat16* output,",
        "    __nv_bfloat16* output,\n    __nv_bfloat16* pre_norm,",
    )
    anchor = "    output[row * kWidth + feature] = rounded;"
    kernel = replace_once(
        kernel, anchor, anchor + "\n    pre_norm[row * kWidth + feature] = rounded;"
    )
    inverse = (
        kernel.replace(NEW, OLD)
        .replace("\n    __nv_bfloat16* pre_norm,", "")
        .replace("\n    pre_norm[row * kWidth + feature] = rounded;", "")
    )
    if inverse != blocks["kernel"]:
        raise ValueError("Head export changed original normalized arithmetic")
    launcher = (
        blocks["launcher"].replace("dspark_target_hc_head_rmsnorm_cuda", HOST).replace(OLD, NEW)
    )
    launcher = replace_once(
        launcher,
        "    const torch::Tensor& output,",
        "    const torch::Tensor& output,\n    const torch::Tensor& pre_norm,",
    )
    launcher = replace_once(launcher, "&norm_weight, &output}", "&norm_weight, &output, &pre_norm}")
    launcher = replace_once(
        launcher,
        "&& output.scalar_type() == torch::kBFloat16,",
        "&& output.scalar_type() == torch::kBFloat16\n"
        "                  && pre_norm.scalar_type() == torch::kBFloat16,",
    )
    launcher = replace_once(
        launcher,
        "  const c10::cuda::CUDAGuard device_guard(streams.device());",
        """  TORCH_CHECK(pre_norm.dim() == 2 && pre_norm.size(0) == streams.size(0)
                  && pre_norm.size(1) == 4096, "pre-RMS export shape changed");
  for (const auto* tensor : {&hc_fn, &hc_scale, &hc_base, &norm_weight, &output, &pre_norm}) {
    TORCH_CHECK(tensor->device() == streams.device(), "HC tensors must share one device");
  }
  const c10::cuda::CUDAGuard device_guard(streams.device());""",
    )
    launcher = replace_once(
        launcher,
        "      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),",
        "      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),\n"
        "      reinterpret_cast<__nv_bfloat16*>(pre_norm.data_ptr()),",
    )
    headers = """#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
"""
    cuda = (
        headers + "\nnamespace {\n" + blocks["warp_sum"] + "\n" + kernel + "\n}\n" + launcher + "\n"
    )
    declaration = launcher[: launcher.index("{")].rstrip() + ";\n"
    return (
        declaration,
        cuda,
        {
            "source_file": str(path),
            "source_file_sha256": digest(path.read_bytes()),
            "wrapper_file": str(wrapper_file),
            "wrapper_file_sha256": digest(wrapper_file.read_bytes()),
            "function_sha256": PINS,
            "cuda_source_sha256": digest(cuda),
            "declaration_sha256": digest(declaration),
            "kernel_inverse_exact": True,
            "extra_cflags": CFLAGS,
            "extra_cuda_cflags": CUDA_FLAGS,
            "additional_timed_bytes": 5 * 4096 * 2,
            "launch": [5, 1024],
            "scope": "Unchanged HC/normalized math; one additional BF16 pre-RMS store per feature",
        },
    )


def install_head_export(torch, graft_module, output):
    """Patch this graft before capture; return a JSON-serializable source/build receipt.

    `output` is the caller-owned CUDA BF16[5,4096] pre-RMS destination. Keep this
    graft alive for every graph replay. `_equal_work_head_export` retains the
    original callable, private leaf and output. Its restore() changes only future
    Python dispatch, not already captured graphs.
    """
    if (
        tuple(output.shape) != (5, 4096)
        or output.dtype != torch.bfloat16
        or not output.is_cuda
        or not output.is_contiguous()
    ):
        raise ValueError("Pre-RMS export must be contiguous CUDA BF16[5,4096]")
    if hasattr(graft_module, "_equal_work_head_export"):
        raise ValueError("Head export is already installed on this graft")
    if torch.cuda.is_current_stream_capturing():
        raise ValueError("Install the head export before graph capture")
    original = graft_module.target_hc_head_rmsnorm_cuda
    wrapper_file = inspect.getsourcefile(original)
    if wrapper_file is None or original.__code__.co_name != "target_hc_head_rmsnorm_cuda":
        raise ValueError("Expected the original HC wrapper")
    declaration, cuda, receipt = source_for(wrapper_file)
    from torch.utils.cpp_extension import load_inline

    library = load_inline(
        name="dspark_equal_work_head_" + receipt["cuda_source_sha256"][:16],
        cpp_sources=declaration,
        cuda_sources=cuda,
        functions=[HOST],
        extra_cflags=CFLAGS,
        extra_cuda_cflags=CUDA_FLAGS,
        with_cuda=True,
        verbose=False,
    )
    pointer, extent = output.data_ptr(), output.numel() * output.element_size()

    def launch(streams, hc_fn, hc_scale, hc_base, norm_weight, normalized, norm_eps, hc_eps):
        """Original validation runs first; check the extra owner before dispatch."""
        if output.data_ptr() != pointer or tuple(output.shape) != (5, 4096):
            raise ValueError("Pre-RMS output was rebound")
        for tensor in (streams, hc_fn, hc_scale, hc_base, norm_weight, normalized):
            if tensor.device != output.device:
                raise ValueError("HC export owners must share one device")
            begin, end = (
                tensor.data_ptr(),
                tensor.data_ptr() + tensor.numel() * tensor.element_size(),
            )
            if pointer < end and begin < pointer + extent:
                raise ValueError("Pre-RMS export must not alias any HC input or output")
        getattr(library, HOST)(
            streams, hc_fn, hc_scale, hc_base, norm_weight, normalized, output, norm_eps, hc_eps
        )

    namespace = dict(original.__globals__)
    namespace["_load_route_pack_extension"] = lambda: SimpleNamespace(target_hc_head_rmsnorm=launch)
    wrapper = FunctionType(
        original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__
    )
    wrapper.__kwdefaults__ = original.__kwdefaults__
    graft_module.target_hc_head_rmsnorm_cuda = wrapper
    receipt.update(
        native_file=str(library.__file__),
        native_sha256=digest(Path(library.__file__).read_bytes()),
        output_pointer=pointer,
        output_bytes=extent,
        original_python_validation_preserved=True,
    )

    def restore():
        """Avoid undoing another caller's later patch."""
        if graft_module.target_hc_head_rmsnorm_cuda is not wrapper:
            raise ValueError("HC export dispatch changed before restore")
        graft_module.target_hc_head_rmsnorm_cuda = original

    graft_module._equal_work_head_export = SimpleNamespace(
        output=output,
        library=library,
        original=original,
        wrapper=wrapper,
        receipt=receipt,
        restore=restore,
    )
    return receipt
