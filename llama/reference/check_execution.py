"""Validate every archived realized execution using independent HF on its history.

No candidate kernels run here. Distinct input histories are evaluated once and
all prefix/final-logit variants retain their observation references/frequencies.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time
import traceback

from generate import ROOT,configure_torch,rollout,runtime_manifest,save_json,sha256
from metrics import logit_metrics,summarize


def state_digest(tokens,final_logits):
    prefix=json.dumps(tokens[:-1],separators=(',',':')).encode()
    stream=hashlib.sha256(prefix).hexdigest()
    raw=final_logits.contiguous().numpy().tobytes()
    return stream,hashlib.sha256(prefix+b'\0'+raw).hexdigest()


def within(root,name):
    path=(root/name).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('Artifact path escapes result directory')
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--fixture',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise ValueError('Use a new output directory; prior checks are immutable')
    source=json.loads((args.results/'results.json').read_text())
    if source.get('status')!='complete':raise ValueError('Matched result collection must be complete')
    if sha256(args.fixture)!=source['fixture_sha256']:raise ValueError('Fixture digest differs from matched run')
    expected=json.loads((ROOT/'model-sha256.json').read_text())
    if {name:sha256(args.model/name) for name in expected}!=expected:raise ValueError('Frozen model hashes mismatch')
    args.out.mkdir(parents=True)
    report={'schema':'llama1b-realized-execution-check-v1','status':'running',
            'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'matched_results_sha256':sha256(args.results/'results.json'),
            'fixture_sha256':sha256(args.fixture),'model_sha256':expected,
            'source_sha256':{f.name:sha256(f) for f in [Path(__file__),ROOT/'generate.py',ROOT/'metrics.py']},
            'streams':{},'unique_executions':{},'failures':[],
            'interpretation':'All observed histories are retained. Reproducibility and numerical validity are distinct. No tolerance inferred from the tested kernel; acceptance is external and preregistered.'}
    save_json(args.out/'results.json',report)
    try:
        torch,transformers=configure_torch();report['runtime']=runtime_manifest(torch,transformers)
        fixture=torch.load(args.fixture,map_location='cpu',weights_only=True)
        prompt=fixture['prompt_ids'].tolist();vocab=fixture['fp32_logits'].shape[-1]
        entries={};streams={};count=0
        for arm,checks in source['execution_checks'].items():
            for index,check in enumerate(checks):
                if 'output_artifact' not in check or 'output_sha256' not in check:
                    raise ValueError('Legacy unchecked result cannot enter this validator')
                path=within(args.results,check['output_artifact'])
                if sha256(path)!=check['output_sha256']:raise ValueError('Execution artifact digest mismatch')
                state=torch.load(path,map_location='cpu',weights_only=True)
                tokens,final=state['tokens'],state['final_logits']
                if tokens.dtype!=torch.int64 or tuple(tokens.shape)!=(128,) or final.dtype!=torch.float32 or tuple(final.shape)!=(vocab,):
                    raise ValueError('Archived tensor schema mismatch')
                token_list=tokens.tolist()
                if not all(0<=x<vocab for x in token_list):raise ValueError('Out-of-range generated token retained in source; cannot reference-replay')
                if check.get('tokens')!=token_list:raise ValueError('JSON and tensor token arrays differ')
                stream,key=state_digest(token_list,final)
                observed={'arm':arm,'stage':check['stage'],'index':index,
                          'output_artifact':check['output_artifact'],'output_sha256':check['output_sha256'],
                          'selected_final_token':token_list[-1]}
                if key not in entries:
                    finite=bool(final.isfinite().all())
                    entries[key]={'stream_id':stream,'tokens':token_list,'actual_final':final.clone(),
                                  'finite':finite,'observations':[],
                                  'final_argmax':int(final.argmax()) if finite else None,
                                  'initial_token_matches_fixture':token_list[0]==int(fixture['tokens'][0])}
                entries[key]['observations'].append(observed)
                if stream not in streams:streams[stream]=token_list
                count+=1
        if not count:raise ValueError('No archived executions to validate')
        report['observation_count']=count;report['unique_execution_count']=len(entries)
        report['unique_input_history_count']=len(streams)
        save_json(args.out/'results.json',report)
        fp32=transformers.AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.float32,attn_implementation='eager',local_files_only=True).to('cuda').eval()
        bf16=transformers.AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True).to('cuda').eval()
        for model in [fp32,bf16]:model.requires_grad_(False)
        final_references={}
        with torch.inference_mode(),torch.autocast('cuda',enabled=False):
            for stream,tokens in streams.items():
                start=time.monotonic();directory=args.out/stream;directory.mkdir()
                r=rollout(fp32,prompt,torch,forced=tokens)
                b=rollout(bf16,prompt,torch,forced=tokens)
                final_references[stream]=(r['logits'][-1],b['logits'][-1])
                torch.save({'tokens':torch.tensor(tokens),'fp32_logits':r['logits'],'bf16_logits':b['logits']},directory/'reference.pt')
                decisions=[];calibration=[]
                for i in range(127):
                    rr,bb=r['logits'][i],b['logits'][i];chosen=tokens[i+1]
                    row=logit_metrics(rr.tolist(),bb.tolist(),chosen)
                    calibration.append(row)
                    if row['reference_finite'] and row['actual_finite']:
                        best=float(rr.max());native_choice=row['actual_top1']
                        decisions.append({'position':32+i,'output_index':i+1,'selected_token':chosen,
                                          'fp32_top_token':row['reference_top1'],'native_bf16_top_token':native_choice,
                                          'selected_fp32_logit_loss':best-float(rr[chosen]),
                                          'native_bf16_choice_fp32_logit_loss':best-float(rr[native_choice]),
                                          'selected_equals_fp32':chosen==row['reference_top1'],
                                          'selected_equals_native_bf16':chosen==native_choice,
                                          'fp32_top2_margin':row['reference_top2_margin']})
                    else:decisions.append({'position':32+i,'output_index':i+1,'reference_invalid':True})
                data={'stream_id':stream,'input_prefix_tokens':tokens[:-1],'representative_final_token':tokens[-1],
                      'tokens':tokens,'decisions':decisions,'shared_decision_prefix':decisions[:-1],
                      'bf16_calibration_on_realized_stream':calibration,'summary':summarize(calibration)}
                save_json(directory/'metrics.json',data)
                report['streams'][stream]={'tokens':tokens,'metrics_sha256':sha256(directory/'metrics.json'),
                                          'reference_sha256':sha256(directory/'reference.pt'),
                                          'reference_finite':data['summary']['finite_rows']==127,
                                          'elapsed_seconds':time.monotonic()-start}
                save_json(args.out/'results.json',report)
                print(json.dumps({'stream':stream,'references_finite':report['streams'][stream]['reference_finite'],'elapsed_seconds':time.monotonic()-start}),flush=True)
        for key,entry in entries.items():
            r,b=final_references[entry['stream_id']]
            actual=entry.pop('actual_final')
            targets=sorted({obs['selected_final_token'] for obs in entry['observations']})
            entry['metrics_by_selected_token']={}
            for target in targets:
                vs_fp32=logit_metrics(r.tolist(),actual.tolist(),target)
                calibration=logit_metrics(r.tolist(),b.tolist(),target)
                vs_bf16=logit_metrics(b.tolist(),actual.tolist(),target)
                if vs_fp32['reference_finite'] and vs_fp32['actual_finite'] and calibration['actual_finite']:
                    deltas={'kl_error_minus_native_bf16':vs_fp32['kl_reference_to_actual']-calibration['kl_reference_to_actual'],
                            'absolute_nll_error_minus_native_bf16':abs(vs_fp32['nll_delta'])-abs(calibration['nll_delta'])}
                else:deltas=None
                decision={'position':158,'output_index':127,'selected_token':target,
                          'selected_fp32_logit_loss':float(r.max()-r[target]),
                          'native_bf16_top_token':int(b.argmax()),
                          'native_bf16_choice_fp32_logit_loss':float(r.max()-r[int(b.argmax())])} if bool(r.isfinite().all() and b.isfinite().all()) else {'reference_invalid':True}
                entry['metrics_by_selected_token'][str(target)]={'actual_vs_fp32':vs_fp32,'native_bf16_vs_fp32':calibration,'actual_vs_bf16':vs_bf16,'paired_error_differences':deltas,'final_decision':decision}
            entry['all_observed_final_tokens_match_argmax']=all(target==entry['final_argmax'] for target in targets)
            entry['observation_count']=len(entry['observations'])
            report['unique_executions'][key]=entry
            save_json(args.out/'results.json',report)
        report['all_actual_final_logits_finite']=all(e['finite'] for e in entries.values())
        report['all_reference_logits_finite']=all(s['reference_finite'] for s in report['streams'].values())
        report['all_token_boundaries_valid']=all(e['initial_token_matches_fixture'] and e['all_observed_final_tokens_match_argmax'] for e in entries.values())
        report['status']='complete'
        report['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        save_json(args.out/'results.json',report)
    except Exception as error:
        report['status']='failed';report['failures'].append({'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
        save_json(args.out/'results.json',report);raise


if __name__=='__main__':main()
