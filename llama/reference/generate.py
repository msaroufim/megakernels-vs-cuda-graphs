"""Generate immutable independent HF FP32/BF16 fixtures; no candidate execution."""
import argparse
import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

from metrics import logit_metrics, summarize

ROOT = Path(__file__).resolve().parent
SUITE_SHA256 = 'b890e9f84f6ecf9af592bcda3a9ff2c424d7164daf82818ee03363104b41b92f'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path=Path(path)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def check_inputs(args):
    if args.out.exists():
        raise ValueError('Use a new --out directory; existing results are immutable')
    suite_path=ROOT/'prompts-v1.json'
    if sha256(suite_path) != SUITE_SHA256:
        raise ValueError('Frozen prompt suite SHA256 mismatch')
    suite=json.loads(suite_path.read_text())
    expected=json.loads((ROOT/'model-sha256.json').read_text())
    observed={name:sha256(args.model/name) for name in expected}
    if observed != expected:
        raise ValueError('Model/config/tokenizer does not match the frozen checkpoint')
    freeze=None
    if args.candidate_freeze:
        data=json.loads(args.candidate_freeze.read_text())
        if not isinstance(data.get('candidates'),dict) or not data['candidates']:
            raise ValueError('Candidate freeze must contain a nonempty candidates mapping')
        freeze={'sha256':sha256(args.candidate_freeze),'contents':data}
    if args.split=='validation' and freeze is None:
        raise ValueError('Reserved validation requires an earlier --candidate-freeze JSON')
    return suite,observed,freeze


def configure_torch():
    # Set before creating a CUDA context, required by deterministic cuBLAS.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    import transformers
    if transformers.__version__ != '4.48.3':
        raise RuntimeError('Reference requires transformers==4.48.3')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Expose exactly the assigned GPU with CUDA_VISIBLE_DEVICES')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20310914)
    return torch,transformers


def rollout(model, prompt, torch, *, forced=None, retain_logits=True, retain_prefill_cache=False):
    """True sequential decode with a new cache; forced tokens never use model argmax."""
    from transformers.cache_utils import DynamicCache
    device=next(model.parameters()).device
    ids=torch.tensor([prompt],dtype=torch.long,device=device)
    cache=DynamicCache()
    result=model(input_ids=ids,attention_mask=torch.ones_like(ids),
                 position_ids=torch.arange(32,device=device)[None,:],
                 cache_position=torch.arange(32,device=device),past_key_values=cache,
                 use_cache=True,return_dict=True,num_logits_to_keep=1)
    logit=result.logits[0,-1].detach().float()
    prefill=logit.cpu().clone()
    native_logit_dtype=str(result.logits.dtype)
    first=int(logit.argmax())
    tokens=[first] if forced is None else [int(forced[0])]
    cache=result.past_key_values
    prefill_cache={}
    if retain_prefill_cache:
        # HF cache [batch,kv_head,time,dim] -> fixture [layer,time,kv_head,dim].
        # Clone before any decode updates; preserve native noninterleaved K.
        prefill_cache['k']=torch.stack([layer[0].transpose(0,1).detach().cpu().clone() for layer in cache.key_cache])
        prefill_cache['v']=torch.stack([layer[0].transpose(0,1).detach().cpu().clone() for layer in cache.value_cache])
        if tuple(prefill_cache['k'].shape)!=(16,32,8,64):
            raise RuntimeError('Unexpected HF prefill cache shape')
    logits=[]
    for index,position in enumerate(range(32,159),start=1):
        inp=torch.tensor([[tokens[-1]]],dtype=torch.long,device=device)
        result=model(input_ids=inp,
                     attention_mask=torch.ones((1,position+1),dtype=torch.long,device=device),
                     position_ids=torch.tensor([[position]],device=device),
                     cache_position=torch.tensor([position],device=device),
                     past_key_values=cache,use_cache=True,return_dict=True,num_logits_to_keep=1)
        cache=result.past_key_values
        logit=result.logits[0,-1].detach().float()
        if retain_logits:logits.append(logit.cpu().clone())
        tokens.append(int(logit.argmax()) if forced is None else int(forced[index]))
    if len(tokens)!=128 or int(cache.get_seq_length())!=159:
        raise RuntimeError('Reference decode shape/cache-length invariant failed')
    return {'tokens':torch.tensor(tokens,dtype=torch.long),'prefill':prefill,
            'logits':torch.stack(logits) if retain_logits else None,
            'native_logit_dtype':native_logit_dtype,'cache_length':int(cache.get_seq_length()),
            'prefill_cache':prefill_cache}


def runtime_manifest(torch,transformers):
    import transformers.models.llama.modeling_llama as llama
    import transformers.models.llama.configuration_llama as configuration
    import transformers.cache_utils as cache_utils
    props=torch.cuda.get_device_properties(0)
    sources={str(Path(inspect.getfile(module)).resolve()):sha256(inspect.getfile(module))
             for module in [llama,configuration,cache_utils]}
    try:
        driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).splitlines()[0]
    except (OSError,subprocess.CalledProcessError):driver=None
    return {'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,
            'transformers':transformers.__version__,'cuda':torch.version.cuda,
            'driver':driver,'gpu':props.name,'gpu_uuid':str(props.uuid),
            'sm_count':props.multi_processor_count,'capability':list(torch.cuda.get_device_capability()),
            'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'attention_implementation':'eager','cache_implementation':'DynamicCache',
            'autocast':False,'float32_matmul_precision':torch.get_float32_matmul_precision(),
            'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
            'allow_bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            'allow_fp16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'hf_source_sha256':sources}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--split',choices=['development','validation'],required=True)
    parser.add_argument('--candidate-freeze',type=Path)
    args=parser.parse_args()
    suite,model_hashes,freeze=check_inputs(args)
    args.out.mkdir(parents=True)
    manifest={'schema':'llama1b-hf-reference-v1','status':'running',
              'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'split':args.split,'suite_sha256':SUITE_SHA256,'suite':suite,
              'model_sha256':model_hashes,'candidate_freeze':freeze,
              'source_sha256':{f.name:sha256(f) for f in [Path(__file__),ROOT/'metrics.py',ROOT/'model-sha256.json']},
              'prompts':{},'failures':[],'reference_description':'Native HF eager implementation; FP32 promotion of BF16 checkpoint versus native BF16; no candidate execution'}
    save_json(args.out/'manifest.json',manifest)
    try:
        torch,transformers=configure_torch()
        manifest['runtime']=runtime_manifest(torch,transformers)
        save_json(args.out/'manifest.json',manifest)
        tokenizer=transformers.AutoTokenizer.from_pretrained(args.model,local_files_only=True)
        prompt_items=[item for item in suite['prompts'] if item['split']==args.split]
        prompts={item['id']:tokenizer.encode(item['text'],add_special_tokens=True)[:32] for item in prompt_items}
        if not all(len(ids)==32 for ids in prompts.values()):
            raise RuntimeError('Every frozen prompt must have at least32 tokens; do not drop cases')
        manifest['tokenized_prompt_ids']=prompts
        save_json(args.out/'manifest.json',manifest)
        fp32=transformers.AutoModelForCausalLM.from_pretrained(
            args.model,torch_dtype=torch.float32,attn_implementation='eager',local_files_only=True).to('cuda').eval()
        bf16=transformers.AutoModelForCausalLM.from_pretrained(
            args.model,torch_dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True).to('cuda').eval()
        for model,dtype in [(fp32,torch.float32),(bf16,torch.bfloat16)]:
            if model.config._attn_implementation!='eager' or any(p.dtype!=dtype for p in model.parameters()):
                raise RuntimeError('Reference dtype or attention backend differs from protocol')
            model.requires_grad_(False)
        all_rows=[];per_prompt={}
        with torch.inference_mode(),torch.autocast('cuda',enabled=False):
            for item in prompt_items:
                key=item['id'];start=time.monotonic();directory=args.out/key;directory.mkdir()
                canonical=rollout(fp32,prompts[key],torch)
                forced=rollout(bf16,prompts[key],torch,forced=canonical['tokens'].tolist(),retain_prefill_cache=True)
                native=rollout(bf16,prompts[key],torch,retain_logits=False)
                tensors={'prompt_ids':torch.tensor(prompts[key],dtype=torch.long),
                         'tokens':canonical['tokens'],'bf16_own_tokens':native['tokens'],
                         'positions':torch.arange(32,159,dtype=torch.long),
                         'input_ids':canonical['tokens'][:-1].clone(),'target_ids':canonical['tokens'][1:].clone(),
                         'fp32_logits':canonical['logits'],'bf16_logits':forced['logits'],
                         'prefill_fp32_logits':canonical['prefill'],'prefill_bf16_logits':forced['prefill'],
                         'prefill_bf16_k':forced['prefill_cache']['k'],'prefill_bf16_v':forced['prefill_cache']['v']}
                torch.save(tensors,directory/'fixture.pt')
                rows=[]
                for i,position in enumerate(range(32,159)):
                    row=logit_metrics(canonical['logits'][i].tolist(),forced['logits'][i].tolist(),int(canonical['tokens'][i+1]))
                    row.update(position=position,output_index=i+1);rows.append(row)
                prefill=logit_metrics(canonical['prefill'].tolist(),forced['prefill'].tolist(),int(canonical['tokens'][0]))
                mismatches=[i for i,(r,b) in enumerate(zip(canonical['tokens'].tolist(),native['tokens'].tolist())) if r!=b]
                metrics={'prompt_id':key,'prefill':prefill,'decode':rows,'summary':summarize(rows),
                         'canonical_tokens':canonical['tokens'].tolist(),'bf16_own_tokens':native['tokens'].tolist(),
                         'native_bf16_exact_fp32':not mismatches,'native_bf16_mismatch_indices':mismatches,
                         'native_bf16_first_mismatch':mismatches[0] if mismatches else None,
                         'fp32_native_logit_dtype':canonical['native_logit_dtype'],
                         'bf16_native_logit_dtype':forced['native_logit_dtype']}
                save_json(directory/'metrics.json',metrics)
                all_rows.extend(rows);per_prompt[key]=metrics['summary']
                manifest['prompts'][key]={'prompt_ids':prompts[key],
                                        'fixture_sha256':sha256(directory/'fixture.pt'),
                                        'metrics_sha256':sha256(directory/'metrics.json'),
                                        'elapsed_seconds':time.monotonic()-start,
                                        'finite':prefill['reference_finite'] and prefill['actual_finite'] and metrics['summary']['finite_rows']==127,
                                        'native_bf16_exact_fp32':not mismatches}
                save_json(args.out/'manifest.json',manifest)
                print(json.dumps({'prompt':key,'split':args.split,'finite':manifest['prompts'][key]['finite'],'bf16_exact_fp32':not mismatches,'elapsed_seconds':time.monotonic()-start}),flush=True)
        envelope={'description':'Descriptive BF16-HF versus FP32-HF envelope, no automatic acceptance threshold',
                  'split':args.split,'decode':summarize(all_rows),'per_prompt':per_prompt,
                  'all_prompts_included':list(prompts),'suite_sha256':SUITE_SHA256}
        save_json(args.out/'envelope.json',envelope)
        manifest['envelope_sha256']=sha256(args.out/'envelope.json')
        manifest['status']='complete' if all(r['finite'] for r in manifest['prompts'].values()) else 'invalid_reference'
        manifest['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        save_json(args.out/'manifest.json',manifest)
        if manifest['status']!='complete':raise RuntimeError('Reference nonfinite values retained; suite is invalid')
    except Exception as error:
        manifest['status']='failed'
        manifest['failures'].append({'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
        save_json(args.out/'manifest.json',manifest)
        raise


if __name__=='__main__':main()
