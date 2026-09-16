"""Matched conventional/public megakernel runs with independent HF fixtures."""
import argparse, hashlib, importlib.util, inspect, json, os, random, statistics, subprocess, sys, time, uuid
from pathlib import Path
import torch
from megakernels.llama import LlamaForCausalLM
from megakernels.model_types import BatchState, ExtraModelConfig
from megakernels.demos.latency.scheduler import LatencyScheduleBuilder
from megakernels.demos.latency.mk import interpret_with_mk
from megakernels.scheduler import assign_to_sms, tensorize_instructions
from megakernels.mk import get_mk_func

ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def telemetry():
    p=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True)
    return p.stdout.strip()
def metrics(actual,reference,target):
    x=actual.flatten().double().cpu();r=reference.flatten().double().cpu()
    if not bool(x.isfinite().all() and r.isfinite().all()):return {'finite':False}
    lp=x.log_softmax(0);rp=r.log_softmax(0);choice=int(x.argmax());top=r.topk(2)
    centered_x=x-x.mean();centered_r=r-r.mean()
    return {'finite':True,'relative_l2':float((x-r).norm()/r.norm()),'centered_relative_l2':float((centered_x-centered_r).norm()/centered_r.norm()),'max_abs':float((x-r).abs().max()),'kl':float((rp.exp()*(rp-lp)).sum()),'nll_error':float(rp[target]-lp[target]),'abs_nll_error':abs(float(rp[target]-lp[target])),'greedy':choice,'reference_greedy':int(r.argmax()),'greedy_agreement':choice==int(r.argmax()),'reference_margin':float(top.values[0]-top.values[1]),'reference_logit_loss':float(r.max()-r[choice])}

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    for field in ['model','candidate','fixture','out']:p.add_argument('--'+field,required=True)
    p.add_argument('--control',choices=['legacy','v2'],default='legacy')
    p.add_argument('--upstream-root');p.add_argument('--mk-dir',default='demos/low-latency-llama')
    p.add_argument('--mode',choices=['graph','native'],default='graph')
    p.add_argument('--max-len',type=int,choices=[256,16384],default=16384)
    p.add_argument('--samples',type=int,default=10);p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--seed',type=int,default=314159);p.add_argument('--replicate',type=int,default=0)
    p.add_argument('--skip-timing',action='store_true')
    p.add_argument('--sequence-glue',choices=['fast-argmax-embedding'],default='fast-argmax-embedding')
    a=p.parse_args();out=Path(a.out)
    if a.warmup < 0:p.error('warmup must be nonnegative')
    if out.exists():p.error('Output directory already exists; use a fresh run identity')
    if a.samples<2 or a.samples%2:p.error('samples must be positive and even for balanced order')
    out.mkdir(parents=True);torch.set_num_threads(4)
    assert torch.cuda.device_count()==1, 'Expose exactly one assigned GPU'
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    torch.set_float32_matmul_precision('highest')
    fixture_path=Path(a.fixture).resolve()
    manifest_path=fixture_path.parent.parent/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    receipt=manifest['prompts'][fixture_path.parent.name]
    assert receipt['finite'] and receipt['fixture_sha256']==digest(fixture_path)
    assert manifest['split']=='development' or manifest['status']=='complete'
    assert manifest['model_sha256']=={name:digest(Path(a.model)/name) for name in manifest['model_sha256']}
    fixture=torch.load(fixture_path,map_location='cpu',weights_only=True)
    assert fixture['prompt_ids'].shape==(32,) and fixture['tokens'].shape==(128,)
    assert fixture['prompt_ids'].tolist()==receipt['prompt_ids']
    assert torch.equal(fixture['positions'],torch.arange(32,159))
    assert torch.equal(fixture['input_ids'],fixture['tokens'][:-1]) and torch.equal(fixture['target_ids'],fixture['tokens'][1:])
    for key in ['prefill_bf16_k','prefill_bf16_v']:
        assert fixture[key].shape==(16,32,8,64) and fixture[key].dtype==torch.bfloat16 and bool(fixture[key].isfinite().all())
    for key in ['fp32_logits','bf16_logits']:
        assert fixture[key].shape[0]==127 and fixture[key].dtype==torch.float32 and bool(fixture[key].isfinite().all())
    report={'configuration':vars(a),'process_uuid':str(uuid.uuid4()),'process_id':os.getpid(),'started_unix':time.time(),'fixture_sha256':digest(a.fixture),'fixture_manifest_sha256':digest(manifest_path),'fixture_manifest':manifest,'harness_sha256':digest(__file__),'status':'initializing','telemetry_before':telemetry(),'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,'float32_matmul_precision':torch.get_float32_matmul_precision(),'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'allow_bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,'allow_fp16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction},'environment':{k:v for k,v in os.environ.items() if k.startswith(('ATTN_','FUSED_','PHASE3_')) or k in ['TORCH_CUDA_ARCH_LIST','CUBLAS_WORKSPACE_CONFIG','CUDA_MODULE_LOADING','CUDA_DEVICE_MAX_CONNECTIONS']}}
    import platform,triton,transformers
    report['runtime'].update({'python':platform.python_version(),'triton':triton.__version__,'transformers':transformers.__version__,'torch_git':torch.version.git_version,'nvcc':subprocess.check_output(['nvcc','--version'],text=True).strip()})
    props=torch.cuda.get_device_properties(0);report['gpu']={'name':props.name,'uuid':str(props.uuid),'sms':props.multi_processor_count}
    report['upstream_source_identity']={}
    for name,directory in [('legacy',str(Path(a.mk_dir).resolve().parents[1])),('v2',a.upstream_root)]:
        if directory:
            proc=subprocess.run(['git','-C',directory,'rev-parse','HEAD'],text=True,capture_output=True)
            report['upstream_source_identity'][name]={'directory':directory,'git_head':proc.stdout.strip() if proc.returncode==0 else None}
    report['sources']={str(f.relative_to(Path(a.candidate).parent)):digest(f) for f in Path(a.candidate).parent.rglob('*') if f.suffix in ['.py','.cu','.cuh'] and '__pycache__' not in f.parts}
    legacy_root=Path(a.mk_dir).resolve().parents[1]
    assert Path(inspect.getfile(LlamaForCausalLM)).resolve()==legacy_root/'megakernels/llama.py', 'Legacy model import differs from checked tree'
    legacy_manifest=ROOT/'phase3/upstream/legacy-runtime-sha256.json'
    legacy_identity=json.loads(legacy_manifest.read_text())
    assert all(digest(legacy_root/name)==value for name,value in legacy_identity.items()), 'Legacy source/binary differs from archived manifest'
    report['legacy_runtime_manifest_sha256']=digest(legacy_manifest)
    report['harness_sources']={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),Path(__file__).with_name('sequence_glue.py'),Path(__file__).with_name('sequence_glue_embed.py'),ROOT/'phase3/upstream/upstream_api.py']}
    if a.control=='v2':
        import megakittens
        assert Path(megakittens.__file__).resolve().parent==Path(a.upstream_root).resolve()/'megakittens', 'Resolved v2 import differs from checked tree'
        identity=json.loads((ROOT/'phase3/upstream/full-source-sha256.json').read_text())
        assert all(digest(Path(a.upstream_root)/name)==value for name,value in identity.items()), 'Public v2 source differs from archived manifest'
        report['upstream_full_source_manifest_sha256']=digest(ROOT/'phase3/upstream/full-source-sha256.json')
    if manifest['split']=='validation':
        frozen=manifest['candidate_freeze']['contents']
        assert frozen['harness_sha256']==report['harness_sha256'], 'Validation harness differs from freeze'
        assert frozen['harness_sources']==report['harness_sources'], 'Harness dependency differs from freeze'
        assert frozen['protocol_sha256']==digest(ROOT/'phase3/PROTOCOL.md'), 'Protocol differs from freeze'
        assert frozen['campaign_sha256']==digest(ROOT/'phase3/CAMPAIGN.json'), 'Campaign differs from freeze'
        campaign=json.loads((ROOT/'phase3/CAMPAIGN.json').read_text())
        prompt_index=campaign['prompts'].index(fixture_path.parent.name)
        assert 0<=a.replicate<campaign['fresh_processes_per_prompt'], 'Unexpected process replicate index'
        assert a.seed==314159+100*prompt_index+a.replicate, 'Randomized timing seed differs from predeclared campaign'
        if a.mode=='native':assert fixture_path.parent.name==campaign['native_diagnostic']['prompt'], 'Native diagnostic prompt differs from campaign'
        assert frozen['model_sha256']==manifest['model_sha256'], 'Model differs from freeze'
        assert a.max_len==16384, 'Primary reserved matrix uses KV capacity16384'
        assert a.samples==10 and a.warmup==5, 'Reserved matrix requires5warmup10samples'
        if a.control=='v2':assert frozen['upstream_full_source_manifest_sha256']==report['upstream_full_source_manifest_sha256']
        assert frozen['suite_sha256']==manifest['suite_sha256']==digest(ROOT/'phase3/reference/prompts-v1.json'), 'Prompt suite differs from freeze'
        assert frozen['reference_sources']==manifest['source_sha256'], 'Reference generator differs from freeze'
        assert frozen['legacy_runtime_manifest_sha256']==report['legacy_runtime_manifest_sha256'], 'Legacy runtime differs from freeze'
        assert all(digest(ROOT/name)==value for name,value in frozen['evaluation_sources'].items()), 'Evaluation implementation differs from freeze'
        matches=[x for x in frozen['candidates'].values() if x['sources']==report['sources'] and x['entrypoint']==Path(a.candidate).name]
        assert len(matches)==1, 'Candidate source does not match a unique preregistered freeze'
        expected=matches[0].get('expected_environment',{})
        assert all(os.environ.get(k,default)==default for k,default in expected.items()), 'Candidate environment differs from freeze'
        report['candidate_freeze_verified']=True
    def save():
        temp=out/'results.tmp';temp.write_text(json.dumps(report,indent=2));temp.replace(out/'results.json')
    save()
    model=LlamaForCausalLM.from_pretrained(a.model,device='cuda',dtype=torch.bfloat16,extra_config=ExtraModelConfig(interleave_rope=True,max_len_override=a.max_len))
    schedule=LatencyScheduleBuilder.build(model);g=schedule.globs
    tensorize_instructions(g,assign_to_sms('rr',schedule=schedule))
    candidate=load('phase3_candidate',a.candidate)
    kwargs={}
    if 'hf_config' in inspect.signature(candidate.prepare).parameters:kwargs['hf_config']=json.loads((Path(a.model)/'config.json').read_text())
    before=torch.cuda.memory_allocated();start=time.monotonic();prepared=candidate.prepare(g,**kwargs)
    steps=[candidate.build(g,pos,prepared) for pos in range(32,159)]
    torch.cuda.synchronize();report['candidate_preparation']={'seconds':time.monotonic()-start,'extra_allocated_bytes':torch.cuda.memory_allocated()-before}
    if a.control=='legacy':
        mk=get_mk_func(Path(a.mk_dir));runner=None
        binary=Path(sys.modules['mk_llama'].__file__).resolve()
        assert binary.parent==Path(a.mk_dir).resolve() and digest(binary)==legacy_identity[str(binary.relative_to(legacy_root))], 'Loaded legacy binary differs from checked control'
    else:
        assert a.upstream_root
        adapter=load('phase3_upstream_api',ROOT/'phase3/upstream/upstream_api.py')
        runner=adapter.from_legacy_globals(g,a.upstream_root)
    from sequence_glue_embed import prepare_argmax,argmax_embed_out
    argmax_scratch=prepare_argmax(g.logits)
    report['sequence_glue_sources']={name:digest(Path(__file__).with_name(name)) for name in ['sequence_glue.py','sequence_glue_embed.py']}
    prefix_k=fixture['prefill_bf16_k'].cuda().reshape(16,32,8,2,32).transpose(-1,-2).reshape(16,32,8,64).contiguous()
    prefix_v=fixture['prefill_bf16_v'].cuda();canonical=fixture['tokens'].cuda()
    def reset():
        g.k_cache[:,0,:32].copy_(prefix_k);g.v_cache[:,0,:32].copy_(prefix_v)
    def embed(token):
        torch.index_select(model.model.embed_tokens.embed_tokens.weight,0,token.reshape(1),out=g.hidden_states.reshape(1,-1))
    def decode(arm,index):
        if arm=='candidate':
            steps[index]();result=candidate.output_logits(g,prepared) if hasattr(candidate,'output_logits') else g.logits
        elif runner is not None:result=runner.run()
        else:
            g.barriers.zero_();g.pos_id=32+index;interpret_with_mk(g,mk);result=g.logits
        return result
    report['teacher_forced']={};report['bf16_calibration']=[]
    for index in range(127):report['bf16_calibration'].append(metrics(fixture['bf16_logits'][index],fixture['fp32_logits'][index],int(fixture['target_ids'][index])))
    for arm in ['candidate','control']:
        reset()
        if runner is not None:runner.position.fill_(32)
        rows=[]
        for index in range(127):
            embed(canonical[index]);logits=decode(arm,index);torch.cuda.synchronize()
            if index==0:report.setdefault('logits_layouts',{})[arm]={'dtype':str(logits.dtype),'shape':list(logits.shape),'stride':list(logits.stride())}
            row=metrics(logits,fixture['fp32_logits'][index],int(fixture['target_ids'][index]));row['vs_bf16']=metrics(logits,fixture['bf16_logits'][index],int(fixture['target_ids'][index]));rows.append(row)
        report['teacher_forced'][arm]=rows;save()
    # Instrumented graph checks are untimed and reported separately: output
    # copies alter the graph. Actual timed final outputs are archived below.
    report['teacher_forced_graph']={}
    stored=torch.empty((127,fixture['fp32_logits'].shape[1]),device='cuda',dtype=torch.float32)
    def teacher_graph_run(arm):
        if runner is not None and arm=='control':runner.position.fill_(32)
        for index in range(127):
            embed(canonical[index]);stored[index].copy_(decode(arm,index).flatten())
    if a.mode=='graph':
        for arm in ['candidate','control']:
            reset();teacher_graph_run(arm);torch.cuda.synchronize()
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                reset();teacher_graph_run(arm)
            torch.cuda.current_stream().wait_stream(stream)
            reset();validation_graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(validation_graph):teacher_graph_run(arm)
            repetitions=[]
            for repetition in range(3):
                reset();validation_graph.replay();torch.cuda.synchronize();actual=stored.float().cpu()
                repetitions.append([metrics(actual[index],fixture['fp32_logits'][index],int(fixture['target_ids'][index])) for index in range(127)])
            report['teacher_forced_graph'][arm]=repetitions;save()
            del validation_graph
    outputs={arm:torch.empty(128,device='cuda',dtype=torch.long) for arm in ['candidate','control']}
    final_logits={};eager={};execution_checks={arm:[] for arm in outputs};report['execution_checks']=execution_checks
    def run(arm):
        output=outputs[arm];output[0].copy_(canonical[0])
        if runner is not None and arm=='control':runner.position.fill_(32)
        embed(output[0])
        for index in range(127):
            logits=decode(arm,index);argmax_embed_out(logits,output[index+1],argmax_scratch,model.model.embed_tokens.embed_tokens.weight,g.hidden_states,index<126)
        final_logits[arm]=logits
    def check(arm,stage):
        tokens=outputs[arm].cpu();logits=final_logits[arm].float().cpu()
        baseline=eager[arm]
        same=bool(torch.equal(tokens,baseline['tokens']))
        equal=bool(torch.equal(logits,baseline['logits']))
        row={'stage':stage,'tokens':tokens.tolist(),'tokens_equal_eager':same,'final_logits_bitwise_equal_eager':equal,'finite':bool(logits.isfinite().all()),'final_max_abs_vs_eager':float((logits-baseline['logits']).abs().max()) if same else None}
        artifact=out/(arm+'-'+stage+'.pt')
        torch.save({'tokens':tokens,'final_logits':logits},artifact)
        row['output_artifact']=artifact.name;row['output_sha256']=digest(artifact)
        execution_checks[arm].append(row)
    actions={}
    for arm in outputs:
        reset();run(arm);torch.cuda.synchronize()
        eager[arm]={'tokens':outputs[arm].cpu().clone(),'logits':final_logits[arm].float().cpu().clone()}
        report.setdefault('eager_tokens',{})[arm]=eager[arm]['tokens'].tolist()
        report.setdefault('eager_fp32_token_agreement',{})[arm]=float((eager[arm]['tokens']==fixture['tokens']).float().mean())
        torch.save(eager[arm],out/(arm+'-eager.pt'))
        if a.mode=='graph':
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):reset();run(arm)
            torch.cuda.current_stream().wait_stream(stream)
            reset();graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):run(arm)
            actions[arm]=graph.replay
        else:actions[arm]=lambda arm=arm:run(arm)
        reset();actions[arm]();torch.cuda.synchronize();check(arm,'after_capture' if a.mode=='graph' else 'native_repeat');save()
    timing={arm:[] for arm in outputs};report['timing_us']=timing;report['orders']=[]
    for iteration in range(a.warmup):
        for arm in (['candidate','control'] if iteration%2==0 else ['control','candidate']):
            reset();actions[arm]();torch.cuda.synchronize();check(arm,'warmup-'+str(iteration))
    orders=[['candidate','control'],['control','candidate']]*(a.samples//2);random.Random(a.seed).shuffle(orders)
    if not a.skip_timing:
        for iteration,order in enumerate(orders):
            report['orders'].append(order)
            for arm in order:
                reset();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record();actions[arm]();end.record();end.synchronize()
                timing[arm].append(start.elapsed_time(end)*1000);check(arm,'timed-'+str(iteration));save()
    report['execution_reproducible']={arm:all(x['tokens_equal_eager'] and x['final_logits_bitwise_equal_eager'] and x['finite'] for x in checks) for arm,checks in execution_checks.items()}
    report['mean_us_per_step']={arm:statistics.mean(values)/127 for arm,values in timing.items() if values}
    report['telemetry_after']=telemetry();report['status']='complete';report['interpretation']='Numerical acceptance is evaluated across the entire reserved suite; timings alone do not establish a valid win. Bitwise execution stability is diagnostic, not a correctness gate. Each actual execution output is archived for independent HF validation on its realized trajectory; instrumented graph checks are separate from timings.'
    save();print(json.dumps({k:report[k] for k in ['status','execution_reproducible','mean_us_per_step']}),flush=True)

if __name__=='__main__':main()
