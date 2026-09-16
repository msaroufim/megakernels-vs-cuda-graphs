"""Inspect CUDA graph kernels with parameter arrays, packed buffers, or no arguments."""

from __future__ import annotations

import ctypes as ct


def parameter_bytes(kernel_params, extra, layout):
    """Read driver-owned launch parameters according to CUDA's two launch representations."""
    if not layout:
        return []
    if kernel_params and not extra:
        result = []
        for index, (offset, size) in enumerate(layout):
            pointer = ct.c_void_p.from_address(kernel_params + ct.sizeof(ct.c_void_p) * index).value
            if not pointer:
                raise ValueError(f"Null argument pointer: {index}")
            result.append(
                {"offset": offset, "bytes": size, "hex": ct.string_at(pointer, size).hex()}
            )
        return result
    if extra and not kernel_params:
        # CU_LAUNCH_PARAM_BUFFER_POINTER=1, BUFFER_SIZE=2, END=0.
        values = {}
        for index in range(0, 8, 2):
            key = ct.c_void_p.from_address(extra + ct.sizeof(ct.c_void_p) * index).value
            if not key:
                break
            if key not in (1, 2) or key in values:
                raise ValueError(f"Unsupported CUDA launch extra entry: {key}")
            values[key] = ct.c_void_p.from_address(
                extra + ct.sizeof(ct.c_void_p) * (index + 1)
            ).value
        else:
            raise ValueError("Unterminated CUDA launch extra array")
        if set(values) != {1, 2} or not all(values.values()):
            raise ValueError("Incomplete CUDA packed launch buffer")
        size = ct.c_size_t.from_address(values[2]).value
        if size > 1048576 or any(offset + width > size for offset, width in layout):
            raise ValueError("CUDA argument layout exceeds packed launch buffer")
        return [
            {"offset": offset, "bytes": width, "hex": ct.string_at(values[1] + offset, width).hex()}
            for offset, width in layout
        ]
    raise ValueError("Nonempty CUDA argument layout has no unambiguous launch storage")


def identity(node, nc, fallback):
    """Record native kernel identity and arguments without assuming a launch representation."""
    cu, checked = nc.cu, nc.checked
    kind = int(checked(cu.cuGraphNodeGetType(node)))
    if kind != int(cu.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL):
        return fallback(node)
    params = checked(cu.cuGraphKernelNodeGetParams(node))
    name = checked(cu.cuFuncGetName(params.func))
    name = name.decode() if isinstance(name, bytes) else str(name)
    layout = []
    for index in range(64):
        answer = cu.cuFuncGetParamInfo(params.func, index)
        if int(answer[0]) == int(cu.CUresult.CUDA_ERROR_INVALID_VALUE):
            break
        offset, size = checked(answer)
        if not 0 < size <= 1048576:
            raise ValueError(f"Invalid kernel parameter extent: {name}/{index}")
        layout.append((offset, size))
    else:
        raise ValueError(f"Unsupported argument count: {name}")
    try:
        arguments = parameter_bytes(params.kernelParams, params.extra, layout)
    except ValueError as error:
        raise ValueError(f"{name}: {error}; layout={layout}") from error
    attributes = {}
    for key in dir(cu.CUkernelNodeAttrID):
        if not key.startswith("CU_LAUNCH_ATTRIBUTE_") or key.endswith("_IGNORE"):
            continue
        answer = cu.cuGraphKernelNodeGetAttribute(node, getattr(cu.CUkernelNodeAttrID, key))
        row = {"status": int(answer[0])}
        if row["status"] == 0:
            row["hex"] = ct.string_at(answer[1].getPtr(), 64).hex()
        attributes[key] = row
    return {
        "kind": kind,
        "name": name,
        "function": int(params.func),
        "module": int(checked(cu.cuFuncGetModule(params.func))),
        "grid": [params.gridDimX, params.gridDimY, params.gridDimZ],
        "block": [params.blockDimX, params.blockDimY, params.blockDimZ],
        "dynamic_smem": params.sharedMemBytes,
        "parameters": arguments,
        "attrs": attributes,
        "argument_representation": "array"
        if params.kernelParams
        else "packed"
        if params.extra
        else "none",
    }
