"""Guard the extracted numerical implementation against silent packaging changes."""

import hashlib
from pathlib import Path

from tools.source_inventory import SOURCE_SHA256


def test_frozen_sources_match_recorded_extraction():
  """Require byte identity, including the unchanged numerical acceptance code."""
  root = Path(__file__).resolve().parents[1]
  assert {"kernels/projection.cu", "reference/metrics.py", "benchmarks/summarize.py"} <= set(SOURCE_SHA256)
  for name, expected in SOURCE_SHA256.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
