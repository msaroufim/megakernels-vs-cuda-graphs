"""PR archives are data, never executable setup code on the credentialed runner."""

import gzip
import io
import tarfile
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ci.run_modal import bounded_output, extract_snapshot, read_bounded, read_pins, selected_file


def archive(files):
  out = io.BytesIO()
  with tarfile.open(fileobj=out, mode="w:gz") as stream:
    for name, body in files.items():
      info = tarfile.TarInfo("snapshot/" + name)
      info.size = len(body)
      stream.addfile(info, io.BytesIO(body))
  return out.getvalue()


def source_files():
  return {
    "llama/kernels/candidate.py": b"raise RuntimeError('must not run on host')",
    "llama/upstream/prepare.py": b"UPSTREAM = '" + b"a" * 40 + b"'\nTHUNDERKITTENS = '" + b"b" * 40 + b"'\n",
    "llama/upstream/patches/fix.patch": b"patch data",
  }


def test_snapshot_only_copies_candidate_inputs(tmp_path):
  files = source_files() | {".github/workflows/evil.yml": b"ignored", "llama/ci/run_modal.py": b"ignored"}
  extract_snapshot(archive(files), tmp_path)
  assert (tmp_path / "llama/kernels/candidate.py").read_bytes() == files["llama/kernels/candidate.py"]
  assert not (tmp_path / ".github").exists()
  assert not (tmp_path / "llama/ci").exists()
  assert read_pins(tmp_path) == {"UPSTREAM": "a" * 40, "THUNDERKITTENS": "b" * 40}


@pytest.mark.parametrize("path", ["../escape", "/absolute", "llama/kernels/../../outside"])
def test_unsafe_paths_rejected(path):
  with pytest.raises(ValueError, match="Unsafe"):
    selected_file(path)


def test_archive_rejects_selected_symlinks(tmp_path):
  out = io.BytesIO()
  with tarfile.open(fileobj=out, mode="w:gz") as stream:
    info = tarfile.TarInfo("snapshot/llama/kernels/candidate.py")
    info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
    stream.addfile(info)
  with pytest.raises(ValueError, match="regular"):
    extract_snapshot(out.getvalue(), tmp_path)


def test_pins_are_not_evaluated(tmp_path):
  files = source_files()
  files["llama/upstream/prepare.py"] = b"UPSTREAM = __import__('os').environ['GH_TOKEN']\n"
  with pytest.raises(ValueError, match="literal"):
    extract_snapshot(archive(files), tmp_path)


def test_duplicate_source_path_rejected(tmp_path):
  out = io.BytesIO()
  with tarfile.open(fileobj=out, mode="w:gz") as stream:
    for _ in range(2):
      info = tarfile.TarInfo("snapshot/llama/kernels/candidate.py")
      info.size = 1
      stream.addfile(info, io.BytesIO(b"x"))
  with pytest.raises(ValueError, match="unique"):
    extract_snapshot(out.getvalue(), tmp_path)


def test_download_is_bounded_before_writing_to_disk():
  with pytest.raises(ValueError, match="size limit"):
    bounded_output([sys.executable, "-c", "print('x' * 10000)"], 100, 5)
  assert bounded_output([sys.executable, "-c", "print('ok')"], 100, 5) == b"ok\n"


def test_ignored_archive_members_still_have_size_limits(tmp_path):
  info = tarfile.TarInfo("snapshot/ignored.bin")
  info.size = 129 * 1024 * 1024
  raw = gzip.compress(info.tobuf() + b"\0" * 1024)
  with pytest.raises(ValueError, match="Expanded"):
    extract_snapshot(raw, tmp_path)


def test_artifact_download_uses_current_filesystem_api():
  fs = SimpleNamespace(
    stat=Mock(return_value=SimpleNamespace(is_file=lambda: True, size=2)),
    read_bytes=Mock(return_value=b"ok"),
  )
  sandbox = SimpleNamespace(filesystem=fs)
  assert read_bounded(sandbox, "/out/report.json", 3) == b"ok"
  fs.stat.return_value.size = 4
  fs.read_bytes.reset_mock()
  with pytest.raises(ValueError, match="size limit"):
    read_bounded(sandbox, "/out/report.json", 3)
  fs.read_bytes.assert_not_called()


def test_vendored_sources_are_snapshot_data_and_unsupported_files_are_ignored(tmp_path):
  files = source_files() | {
    "llama/megakernel/Makefile": b"all:\n\tDO_NOT_EXECUTE_ON_HOST\n",
    "llama/megakernel/llama.cu": b"// changed kernel",
    "llama/megakernel/new.cuh": b"// new kernel header",
    "llama/megakernel/include/nested/new.hpp": b"// shared header",
    "llama/megakernel/evil.py": b"raise RuntimeError('must not execute')",
    "llama/megakernel/README.md": b"ignored documentation",
    "llama/megakernel/include/Makefile": b"ignored nested build script",
  }
  extract_snapshot(archive(files), tmp_path)
  for name in ("Makefile", "llama.cu", "new.cuh", "include/nested/new.hpp"):
    assert (tmp_path / "llama/megakernel" / name).read_bytes() == files["llama/megakernel/" + name]
  for name in ("evil.py", "README.md", "include/Makefile"):
    assert not (tmp_path / "llama/megakernel" / name).exists()
  assert not (tmp_path / "llama/upstream/patches").exists()


@pytest.mark.parametrize("name", ["Makefile", "llama.cu", "include/only.cuh"])
def test_incomplete_vendor_tree_is_rejected(tmp_path, name):
  files = source_files() | {"llama/megakernel/" + name: b"incomplete"}
  with pytest.raises(ValueError, match="both Makefile and llama.cu"):
    extract_snapshot(archive(files), tmp_path)
