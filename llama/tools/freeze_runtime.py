"""Freeze the staged H200 revision before generating reserved reference fixtures."""

import argparse
import datetime
import json
from pathlib import Path

from h200_runtime import digest, stage_identity


def freeze(root: Path, candidates: list[str], output: Path) -> dict:
  """Bind the legacy-only runtime and preserve the original analysis schema."""
  if output.exists() or output.is_symlink():
    raise FileExistsError(output)
  identities = stage_identity(root)
  staged = json.loads((root / "stage-manifest.json").read_text())
  entries = {}
  for value in candidates:
    name, path_string = value.split("=", 1)
    path = Path(path_string).resolve()
    if not name or name in entries or not path.is_file():
      raise ValueError("Candidate names must be unique and point to existing files")
    if path != (root / "phase3/fp32_headout_fastmath/candidate.py").resolve():
      raise ValueError("H200 hardware replication only accepts the staged original candidate")
    entries[name] = {
      "entrypoint": path.name,
      "sources": {
        str(file.relative_to(path.parent)): digest(file)
        for file in sorted(path.parent.rglob("*"))
        if file.is_file() and file.suffix in [".py", ".cu", ".cuh"] and "__pycache__" not in file.parts
      },
      "configuration": "source defaults; native_bf16_fma=False; PDL=True; pinned HF config",
      "expected_environment": {},
    }
  if len(entries) != 1:
    raise ValueError("Freeze exactly one original portable candidate")
  harness = "phase3/harness/matched_final.py"
  data = {
    "source_commit": staged["source_commit"],
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "candidates": entries,
    "campaign_sha256": digest(root / "phase3/CAMPAIGN.json"),
    "protocol_sha256": digest(root / "phase3/PROTOCOL.md"),
    "harness_sha256": digest(root / harness),
    "harness_sources": {
      name: digest(root / name)
      for name in [
        harness,
        "phase3/harness/sequence_glue.py",
        "phase3/harness/sequence_glue_embed.py",
        "phase3/upstream/upstream_api.py",
        "phase3/harness/h200_runtime.py",
      ]
    },
    "model_sha256": json.loads((root / "phase3/reference/model-sha256.json").read_text()),
    "reference_sources": {
      name: digest(root / "phase3/reference" / name)
      for name in ["generate.py", "metrics.py", "model-sha256.json", "h200_runtime.py"]
    },
    "legacy_runtime_manifest_sha256": digest(root / "phase3/upstream/legacy-runtime-sha256.json"),
    "evaluation_sources": {
      name: digest(root / name)
      for name in [
        "phase3/reference/check_execution.py",
        "phase3/reference/metrics.py",
        "phase3/harness/summarize.py",
        "phase3/harness/freeze_final.py",
        "phase3/reference/h200_runtime.py",
      ]
    },
    "suite_sha256": digest(root / "phase3/reference/prompts-v1.json"),
    "upstream_manifest": json.loads((root / "phase3/upstream/source-manifest.json").read_text()),
    "h200_runtime_manifest_sha256": digest(root / "phase3/H200_RUNTIME.json"),
    "stage_manifest_sha256": identities["stage-manifest.json"],
  }
  if stage_identity(root) != identities:
    raise RuntimeError("Staged files changed while freezing")
  with output.open("x") as handle:
    handle.write(json.dumps(data, indent=2) + "\n")
  return data


def main() -> None:
  """Use the source-commit provenance already bound by deterministic staging."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--candidate", action="append", required=True)
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  freeze(Path(__file__).resolve().parents[2], args.candidate, args.out)
  print(digest(args.out))


if __name__ == "__main__":
  main()
