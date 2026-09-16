"""Stage measured sources with checked Hopper adaptations."""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.hardware_profile import bind_hardware  # noqa: E402

SOURCE_PATHS = {
  "kernels/candidate.py": "phase3/fp32_headout_fastmath/candidate.py",
  "kernels/projection.cu": "phase3/fp32_headout_fastmath/projection.cu",
  "kernels/attention.py": "phase3/fp32_headout_fastmath/attention.py",
  "benchmarks/matched_final.py": "phase3/harness/matched_final.py",
  "benchmarks/summarize.py": "phase3/harness/summarize.py",
  "benchmarks/sequence_glue.py": "phase3/harness/sequence_glue.py",
  "benchmarks/sequence_glue_embed.py": "phase3/harness/sequence_glue_embed.py",
  "benchmarks/upstream_api.py": "phase3/upstream/upstream_api.py",
  "reference/generate.py": "phase3/reference/generate.py",
  "reference/metrics.py": "phase3/reference/metrics.py",
  "reference/check_execution.py": "phase3/reference/check_execution.py",
}
METADATA_PATHS = (
  "phase3/CAMPAIGN.json",
  "phase3/reference/model-sha256.json",
  "phase3/reference/prompts-v1.json",
  "phase3/upstream/legacy-runtime-sha256.json",
  "phase3/upstream/source-manifest.json",
  "phase3/H200_RUNTIME.json",
)


def replace_exact(text: str, old: str, new: str) -> str:
  """Fail closed if the frozen source no longer has the audited edit site."""
  if text.count(old) != 1:
    raise ValueError(f"Expected exactly one original source anchor: {old!r}")
  return text.replace(old, new)


def adapt(source: str, text: str, gpu_type: str = "H200") -> str:
  """Change architecture/runtime binding only, preserving evaluation arithmetic."""
  edits: list[tuple[str, str]] = []
  if source == "kernels/candidate.py":
    edits = [("native_bf16_fma: bool = True", "native_bf16_fma: bool = False")]
  elif source == "benchmarks/matched_final.py":
    edits = [
      (
        "import torch\n",
        "import torch\nfrom h200_runtime import validate_hardware, validate_runtime, stage_identity\n",
      ),
      ("choices=['legacy','v2']", "choices=['legacy']"),
      (
        "    assert torch.cuda.device_count()==1, 'Expose exactly one assigned GPU'",
        "    assert torch.cuda.device_count()==1, 'Expose exactly one assigned GPU'\n"
        "    validate_hardware(torch)\n    h200_start=stage_identity(ROOT)",
      ),
      (
        "    props=torch.cuda.get_device_properties(0);report['gpu']",
        "    report['runtime']['h200_manifest_sha256']=validate_runtime("
        "report['runtime'],ROOT/'phase3/H200_RUNTIME.json')\n"
        "    report['h200_start_identities']=h200_start\n"
        "    report['runtime']['capability']=list(torch.cuda.get_device_capability())\n"
        "    report['runtime']['host_machine']=platform.machine()\n"
        "    props=torch.cuda.get_device_properties(0);report['gpu']",
      ),
      (
        "Path(__file__).with_name('sequence_glue.py'),Path(__file__).with_name('sequence_glue_embed.py'),ROOT/'phase3/upstream/upstream_api.py']}",
        "Path(__file__).with_name('sequence_glue.py'),Path(__file__).with_name('sequence_glue_embed.py'),ROOT/'phase3/upstream/upstream_api.py',Path(__file__).with_name('h200_runtime.py')]}",
      ),
      (
        "        frozen=manifest['candidate_freeze']['contents']",
        "        frozen=manifest['candidate_freeze']['contents']\n"
        "        assert frozen['h200_runtime_manifest_sha256']==report['runtime']['h200_manifest_sha256']\n"
        "        assert frozen['stage_manifest_sha256']==h200_start['stage-manifest.json']",
      ),
      (
        "    report['telemetry_after']=telemetry();report['status']='complete';",
        "    assert stage_identity(ROOT)==h200_start, 'Staged files changed during execution'\n"
        "    assert all(digest(legacy_root/name)==value for name,value in legacy_identity.items()), "
        "'Legacy inputs changed during execution'\n"
        "    report['h200_end_identities_verified']=True\n"
        "    report['telemetry_after']=telemetry();report['status']='complete';",
      ),
    ]
  elif source == "reference/generate.py":
    edits = [
      (
        "def configure_torch():",
        "from h200_runtime import validate_hardware, validate_runtime, runtime_values, stage_identity\n\n\n"
        "def configure_torch():",
      ),
      (
        "    torch.set_num_threads(4)",
        "    validate_hardware(torch)\n"
        "    stage_identity(ROOT.parents[1])\n"
        "    validate_runtime(runtime_values(torch,transformers),ROOT.parent/'H200_RUNTIME.json')\n"
        "    torch.set_num_threads(4)",
      ),
      (
        "    return {'python':sys.version,'platform':platform.platform(),",
        "    return {'h200_manifest_sha256':sha256(ROOT.parent/'H200_RUNTIME.json'),"
        "'h200_runtime':runtime_values(torch,transformers),'python':sys.version,'platform':platform.platform(),",
      ),
      (
        "ROOT/'metrics.py',ROOT/'model-sha256.json']",
        "ROOT/'metrics.py',ROOT/'model-sha256.json',ROOT/'h200_runtime.py']",
      ),
      (
        "        freeze={'sha256':sha256(args.candidate_freeze),'contents':data}",
        "        assert data['h200_runtime_manifest_sha256']==sha256(ROOT.parent/'H200_RUNTIME.json'), "
        "'H200 runtime differs from freeze'\n"
        "        assert data['stage_manifest_sha256']==sha256(ROOT.parents[1]/'stage-manifest.json'), "
        "'Stage differs from freeze'\n"
        "        freeze={'sha256':sha256(args.candidate_freeze),'contents':data}",
      ),
    ]
  for old, new in edits:
    text = replace_exact(text, old, bind_hardware(new, gpu_type))
  return text


def safe_file(root: Path, name: str) -> Path:
  """Require input containment and a regular file before staging anything."""
  path = (root / name).resolve()
  if not path.is_relative_to(root.resolve()) or not path.is_file():
    raise ValueError(f"Missing or escaping input: {root / name}")
  return path


def stage(study: Path, metadata: Path, output: Path, source_commit: str, gpu_type: str = "H200") -> Path:
  """Preflight every input and transformation, then write a fresh external tree."""
  if output.exists() or output.is_symlink() or not output.parent.is_dir():
    raise ValueError("Output must be a new directory within an existing parent")
  if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
    raise ValueError("Provide the full source archive Git commit")
  bind_hardware("", gpu_type)
  original = study
  files: dict[str, bytes] = {}
  source_hashes = {}
  for source, target in SOURCE_PATHS.items():
    data = safe_file(original, source).read_bytes()
    source_hashes[source] = hashlib.sha256(data).hexdigest()
    files[bind_hardware(target, gpu_type)] = adapt(source, data.decode(), gpu_type).encode()
  for template in METADATA_PATHS:
    name = bind_hardware(template, gpu_type)
    files[name] = safe_file(metadata, name).read_bytes()
    json.loads(files[name])
  import campaign

  runtime = json.loads(files[f"phase3/{gpu_type}_RUNTIME.json"])
  if runtime.get("schema") != f"llama1b-{gpu_type.lower()}-runtime-v1" or runtime.get("hardware") != {
    "capability": [9, 0],
    "name_contains": gpu_type,
  }:
    raise ValueError("Runtime metadata differs from the selected hardware profile")
  if json.loads(files["phase3/CAMPAIGN.json"]) != json.loads(bind_hardware(json.dumps(campaign.plan()), gpu_type)):
    raise ValueError("Campaign metadata differs from the selected hardware profile")
  for target in ["phase3/harness/h200_runtime.py", "phase3/reference/h200_runtime.py"]:
    files[bind_hardware(target, gpu_type)] = bind_hardware(
      safe_file(study, "tools/h200_runtime.py").read_text(), gpu_type
    ).encode()
  files["phase3/harness/freeze_final.py"] = bind_hardware(
    safe_file(study, "tools/freeze_runtime.py").read_text(), gpu_type
  ).encode()
  files["phase3/harness/run_campaign.py"] = bind_hardware(
    safe_file(study, "campaign.py").read_text(), gpu_type
  ).encode()
  files["phase3/PROTOCOL.md"] = (
    safe_file(original, "docs/PROTOCOL.md").read_bytes()
    + bind_hardware("\n\n---\n\n# H200 hardware-replication supplement\n\n", gpu_type).encode()
    + bind_hardware(safe_file(study, "PROTOCOL.md").read_text(), gpu_type).encode()
  )
  if gpu_type == "H100":
    files["phase3/PROTOCOL.md"] += b"\n\n---\n\n" + safe_file(study, "docs/H100_PROTOCOL.md").read_bytes()
  for name, data in files.items():
    if name.endswith(".py"):
      compile(data, name, "exec")
  receipt = {
    "schema": bind_hardware("llama1b-h200-stage-v1", gpu_type),
    "gpu_type": gpu_type,
    "hardware_profile_sha256": hashlib.sha256(safe_file(study, "tools/hardware_profile.py").read_bytes()).hexdigest(),
    "source_commit": source_commit,
    "stager_sha256": hashlib.sha256(safe_file(study, "tools/stage_runtime.py").read_bytes()).hexdigest(),
    "original_source_sha256": source_hashes,
    "staged_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())},
    "adapted_sources": ["kernels/candidate.py", "benchmarks/matched_final.py", "reference/generate.py"],
    "portable_bf16": True,
    "control": "legacy",
  }
  output.mkdir()
  for name, data in files.items():
    target = output / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
  (output / "stage-manifest.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
  return output


def main() -> None:
  """Stage code without importing CUDA, loading models, or launching work."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--metadata-root", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--source-commit", required=True)
  parser.add_argument("--gpu-type", choices=["H200", "H100"], default="H200")
  args = parser.parse_args()
  output = stage(Path(__file__).resolve().parents[1], args.metadata_root, args.out, args.source_commit, args.gpu_type)
  print(output)


if __name__ == "__main__":
  main()
