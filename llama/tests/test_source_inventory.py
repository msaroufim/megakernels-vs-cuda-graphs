"""Keep the published evaluation rules unchanged while allowing kernel research."""

import hashlib
from pathlib import Path

from tools.source_inventory import SOURCE_SHA256


def test_evaluation_sources_match_recorded_extraction():
  """Kernel changes get PR benchmarks; evaluation changes need a separate study."""
  root = Path(__file__).resolve().parents[1]
  assert {"kernels/projection.cu", "reference/metrics.py", "benchmarks/summarize.py"} <= set(SOURCE_SHA256)
  for name, expected in SOURCE_SHA256.items():
    if name.startswith("kernels/"):
      continue
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
