"""Freeze candidate trees and protocol before reserved evaluation."""
import argparse,datetime,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--candidate',action='append',required=True,help='name=/absolute/path/to/candidate.py');p.add_argument('--out',required=True);p.add_argument('--harness',default='phase3/harness/matched_final.py');a=p.parse_args()
root=Path(__file__).resolve().parents[2];out=Path(a.out)
if Path(a.harness).is_absolute():raise ValueError('Harness must be repository-relative')
if out.exists():raise FileExistsError(out)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
candidates={}
for value in a.candidate:
    name,path=value.split('=',1);path=Path(path).resolve();assert path.is_file()
    if name in candidates:raise ValueError('Duplicate candidate name: '+name)
    sources={str(f.relative_to(path.parent)):digest(f) for f in path.parent.rglob('*') if f.is_file() and f.suffix in ['.py','.cu','.cuh'] and '__pycache__' not in f.parts}
    candidates[name]={'entrypoint':path.name,'sources':sources,'configuration':'source defaults; HF config from pinned model','expected_environment':{'ATTN_PREFETCH':'1','ATTN_D':'32','ATTN_WARPS':'4','FUSED_PDL':'1','TORCH_CUDA_ARCH_LIST':'10.0'}}
data={'source_commit':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),'campaign_sha256':digest(root/'phase3/CAMPAIGN.json'),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'candidates':candidates,'protocol_sha256':digest(root/'phase3/PROTOCOL.md'),'harness_sha256':digest(root/a.harness),'harness_sources':{name:digest(root/name) for name in [a.harness,'phase3/harness/sequence_glue.py','phase3/harness/sequence_glue_embed.py','phase3/upstream/upstream_api.py']},'upstream_full_source_manifest_sha256':digest(root/'phase3/upstream/full-source-sha256.json'),'model_sha256':json.loads((root/'phase3/reference/model-sha256.json').read_text()),'reference_sources':{name:digest(root/'phase3/reference'/name) for name in ['generate.py','metrics.py','model-sha256.json']},'legacy_runtime_manifest_sha256':digest(root/'phase3/upstream/legacy-runtime-sha256.json'),'evaluation_sources':{name:digest(root/name) for name in ['phase3/reference/check_execution.py','phase3/reference/metrics.py','phase3/harness/summarize.py','phase3/harness/freeze_final.py']},'suite_sha256':digest(root/'phase3/reference/prompts-v1.json'),'upstream_manifest':json.loads((root/'phase3/upstream/source-manifest.json').read_text())}
out.write_text(json.dumps(data,indent=2)+'\n');print(digest(out))
