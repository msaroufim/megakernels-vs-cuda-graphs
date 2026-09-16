"""Bind existing Hopper wrapper labels and strict guards to one GPU family.

The default H200 stage remains unchanged. H100 binding changes literal hardware
names in orchestration and its generated guard modules, never kernel arithmetic.
"""


def bind_hardware(text: str, gpu_type: str = "H200") -> str:
  """Render an explicit supported profile; do not infer it from visible hardware."""
  if gpu_type not in ("H200", "H100"):
    raise ValueError("GPU type must be H200 or H100")
  return text.replace("H200", gpu_type).replace("h200", gpu_type.lower())
