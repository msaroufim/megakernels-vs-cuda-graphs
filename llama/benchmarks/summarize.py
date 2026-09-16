"""CPU-only immutable-receipt join and prompt/process statistical summary."""
import argparse
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

ARMS = ('candidate', 'control')
METRICS = ('kl', 'abs_nll_error')
REALIZED_KEYS = {'kl': 'kl_error_minus_native_bf16', 'abs_nll_error': 'absolute_nll_error_minus_native_bf16'}

def digest_bytes(raw):
    return hashlib.sha256(raw).hexdigest()

def interval(values):
    values = sorted(values)
    return [values[int(.025 * (len(values)-1))], values[int(.975 * (len(values)-1))]]

def bootstrap_prompt(values, seed=2731, n=10000):
    rng = random.Random(seed)
    return interval([statistics.mean(rng.choices(values, k=len(values))) for _ in range(n)])

def paired_summary(by_prompt):
    means = {key: statistics.mean(values) for key, values in sorted(by_prompt.items())}
    if not means or not all(math.isfinite(x) for x in means.values()):
        return {'accepted': False, 'reason': 'Missing or nonfinite prompt deltas'}
    ci = bootstrap_prompt(list(means.values()))
    return {'prompt_deltas': means, 'mean_delta': statistics.mean(means.values()),
            'paired_prompt_bootstrap95': ci, 'accepted': ci[1] <= 0}

def prompt_id(run):
    return Path(run['configuration']['fixture']).parent.name

def deduplicate_runs(runs):
    """Collapse byte-identical receipts; reject conflicting/missing process IDs."""
    unique = {}; process_hashes = defaultdict(set); issues = []; copies = []
    for run in runs:
        digest = run.get('_receipt_sha256')
        if not digest:
            issues.append('Missing immutable receipt SHA256'); continue
        if digest in unique:
            copies.append(run.get('_receipt_path', digest)); continue
        unique[digest] = run
        process = run.get('process_uuid')
        if not isinstance(process, str) or not process:
            issues.append('Missing process_uuid: ' + digest)
        else:
            process_hashes[process].add(digest)
    conflicts = {p: sorted(v) for p, v in process_hashes.items() if len(v) > 1}
    if conflicts: issues.append('A process UUID identifies multiple different receipts')
    return list(unique.values()), {'duplicate_copies': copies, 'process_conflicts': conflicts, 'issues': issues}

def numerical(runs, field, arm, metric):
    by_prompt = defaultdict(list)
    for run in runs:
        rows = run.get(field, {}).get(arm, [])
        if field == 'teacher_forced_graph':
            if len(rows) != 3 or any(len(repeat) != 127 for repeat in rows):
                return {'accepted': False, 'reason': 'Incomplete three-replay numerical matrix'}
            rows = [row for repeat in rows for row in repeat]
        elif len(rows) != 127:
            return {'accepted': False, 'reason': 'Incomplete 127-step numerical matrix'}
        baseline = run.get('bf16_calibration', [])
        if len(baseline) != 127 or not all(row.get('finite') and math.isfinite(row.get(metric, math.nan)) for row in rows + baseline):
            return {'accepted': False, 'reason': 'Missing or nonfinite numerical outputs'}
        by_prompt[prompt_id(run)].append(statistics.mean(row[metric] for row in rows) - statistics.mean(row[metric] for row in baseline))
    return paired_summary(by_prompt)

def stage_class(stage):
    if stage in ('after_capture', 'native_repeat'): return 'initial'
    if stage.startswith('warmup-'): return 'warmup'
    if stage.startswith('timed-'): return 'timed'
    raise ValueError('Unknown execution stage: ' + stage)

def realized_process(run, validation):
    """Check exact archived-observation coverage and preserve its multiplicity."""
    if validation is None: raise ValueError('Missing independent realized-output validation')
    if validation.get('status') != 'complete' or validation.get('matched_results_sha256') != run['_receipt_sha256']:
        raise ValueError('Incomplete validation or immutable receipt join mismatch')
    if validation.get('fixture_sha256') != run['fixture_sha256']:
        raise ValueError('Validation fixture mismatch')
    if run.get('fixture_manifest',{}).get('split')=='validation':
        frozen=(run['fixture_manifest'].get('candidate_freeze') or {}).get('contents') or {}
        evaluation=frozen.get('evaluation_sources',{});reference=frozen.get('reference_sources',{})
        expected_sources={'check_execution.py':evaluation.get('phase3/reference/check_execution.py'),'generate.py':reference.get('generate.py'),'metrics.py':reference.get('metrics.py')}
        if not all(expected_sources.values()) or validation.get('source_sha256')!=expected_sources:
            raise ValueError('Realized checker implementation differs from frozen evaluation')

    for key in ('all_actual_final_logits_finite', 'all_reference_logits_finite', 'all_token_boundaries_valid'):
        if validation.get(key) is not True: raise ValueError('Failed ' + key)
    if validation.get('failures'): raise ValueError('Validation contains failures')
    expected = {}; actual = {}; values = defaultdict(lambda: defaultdict(list))
    for arm in ARMS:
        checks = run.get('execution_checks', {}).get(arm, [])
        config = run['configuration']
        stages = ['after_capture' if config['mode'] == 'graph' else 'native_repeat']
        stages += ['warmup-' + str(i) for i in range(config['warmup'])]
        stages += ['timed-' + str(i) for i in range(config['samples'])]
        if [row['stage'] for row in checks] != stages: raise ValueError('Incomplete execution-stage matrix')
        for index, check in enumerate(checks):
            if not check.get('finite'): raise ValueError('Nonfinite actual execution')
            expected[(arm, index)] = (check['stage'], check['output_artifact'], check['output_sha256'], check['tokens'][-1])
    for entry in validation.get('unique_executions', {}).values():
        if not all(entry.get(k) is True for k in ('finite', 'initial_token_matches_fixture', 'all_observed_final_tokens_match_argmax')):
            raise ValueError('Invalid realized state')
        if not validation.get('streams', {}).get(entry['stream_id'], {}).get('reference_finite'):
            raise ValueError('Nonfinite or missing realized reference stream')
        observations = entry['observations']
        if entry['observation_count'] != len(observations): raise ValueError('Wrong observation multiplicity')
        for observation in observations:
            arm = observation['arm']; key = (arm, observation['index'])
            if key in actual: raise ValueError('Duplicate observation in validation')
            actual[key] = (observation['stage'], observation['output_artifact'], observation['output_sha256'], observation['selected_final_token'])
            row = entry['metrics_by_selected_token'][str(observation['selected_final_token'])]
            for name in ('actual_vs_fp32', 'native_bf16_vs_fp32', 'actual_vs_bf16'):
                if not all(row[name].get(k) is True for k in ('reference_finite', 'actual_finite')):
                    raise ValueError('Nonfinite realized metrics')
            for metric, name in REALIZED_KEYS.items():
                delta = row['paired_error_differences'][name]
                if not math.isfinite(delta): raise ValueError('Nonfinite realized delta')
                values[(arm, stage_class(observation['stage']))][metric].append(delta)
    if actual != expected or validation.get('observation_count') != len(expected):
        raise ValueError('Realized validation does not cover exactly every archived execution')
    return {key: {metric: statistics.mean(rows) for metric, rows in metrics.items()} for key, metrics in values.items()}

def realized_summary(runs, validations):
    grouped = defaultdict(lambda: defaultdict(list)); failures = []
    for run in runs:
        try:
            record = validations.get(run['_receipt_sha256'])
            if isinstance(record, list): raise ValueError('Conflicting validation receipts for same matched run')
            for (arm, stage), metrics in realized_process(run, record).items():
                for metric, delta in metrics.items(): grouped[(arm, stage, metric)][prompt_id(run)].append(delta)
        except (KeyError, TypeError, ValueError) as error:
            failures.append({'matched_results_sha256': run['_receipt_sha256'], 'reason': str(error)})
    results = {}
    for (arm, stage, metric), prompts in sorted(grouped.items()):
        results.setdefault(arm, {}).setdefault(stage, {})[metric] = paired_summary(prompts)
    accepted = not failures and bool(grouped) and all(row['accepted'] for stages in results.values() for metrics in stages.values() for row in metrics.values())
    return {'complete': not failures and bool(grouped), 'failures': failures, 'by_arm_stage': results, 'accepted': accepted,
            'weighting': 'Mean observations within process/stage, then mean processes within prompt; paired prompt bootstrap. Both metrics required at each stage for both arms.'}

def summarize(runs, expected, min_processes, validations=None):
    runs, dedup = deduplicate_runs(runs)
    grouped = defaultdict(list)
    for run in runs: grouped[prompt_id(run)].append(run)
    missing = sorted(set(expected)-set(grouped))
    counts = {p: len({r.get('process_uuid') for r in rows if r.get('process_uuid')}) for p, rows in grouped.items()}
    summary_hash=digest_bytes(Path(__file__).read_bytes())
    freeze_issues=[]
    for run in runs:
        if run.get('fixture_manifest',{}).get('split')=='validation':
            frozen=(run['fixture_manifest'].get('candidate_freeze') or {}).get('contents') or {}
            if run.get('candidate_freeze_verified') is not True or frozen.get('evaluation_sources',{}).get('phase3/harness/summarize.py')!=summary_hash:
                freeze_issues.append(run['_receipt_sha256'])
    fixture_conflicts={p:sorted({r.get('fixture_sha256','missing') for r in rows}) for p,rows in grouped.items() if len({r.get('fixture_sha256','missing') for r in rows})!=1}
    primary_eligible=bool(runs) and all(r.get('fixture_manifest',{}).get('split')=='validation' for r in runs)
    complete = not freeze_issues and not fixture_conflicts and all(r.get('status') == 'complete' and r['configuration'].get('samples') == 10 and r['configuration'].get('warmup') == 5 for r in runs) and not dedup['issues'] and not missing and set(grouped) == set(expected) and all(n >= min_processes for n in counts.values())
    process_ratios = {}; timing_issues = []
    for prompt, rows in grouped.items():
        process_ratios[prompt] = []
        for run in rows:
            times = run.get('timing_us', {})
            if any(len(times.get(arm, [])) != run['configuration']['samples'] or not all(math.isfinite(x) and x > 0 for x in times.get(arm, [])) for arm in ARMS):
                timing_issues.append(run['_receipt_sha256']); continue
            process_ratios[prompt].append(math.log(statistics.mean(times['control']) / statistics.mean(times['candidate'])))
    complete = complete and not timing_issues
    valid_ratios = {p: v for p, v in process_ratios.items() if v}
    ci = None; mean = None
    if valid_ratios:
        rng = random.Random(314159); prompts = sorted(valid_ratios)
        boot = [statistics.mean(statistics.mean(rng.choices(valid_ratios[p], k=len(valid_ratios[p]))) for p in rng.choices(prompts, k=len(prompts))) for _ in range(10000)]
        ci = [math.exp(x) for x in interval(boot)]
        mean = math.exp(statistics.mean(statistics.mean(v) for v in valid_ratios.values()))
    fields = ['teacher_forced'] + (['teacher_forced_graph'] if runs and runs[0]['configuration']['mode'] == 'graph' else [])
    numbers = {arm: {field: {metric: numerical(runs, field, arm, metric) for metric in METRICS} for field in fields} for arm in ARMS}
    numerical_pass = bool(runs) and all(x['accepted'] for fields in numbers.values() for metrics in fields.values() for x in metrics.values())
    realized = realized_summary(runs, validations or {})
    return {'complete_matrix': complete, 'missing_prompts': missing, 'process_counts': counts, 'deduplication': dedup, 'invalid_timing_receipts': timing_issues,'unverified_freeze_receipts':freeze_issues,'fixture_conflicts':fixture_conflicts,'primary_eligible':primary_eligible,
            'geometric_mean_speedup': mean, 'hierarchical_prompt_process_bootstrap95': ci, 'process_log_ratios': process_ratios,
            'numerical': numbers, 'canonical_numerical_pass': numerical_pass, 'realized_output_validation': realized,
            'realized_output_validation_complete': realized['complete'], 'accepted_speed_win': bool(primary_eligible and complete and numerical_pass and realized['accepted'] and ci and ci[0] > 1),
            'interpretation': 'Conditional on these prompts, processes, GPU and pinned sources; incomplete or unvalidated timing is not an accepted win.'}

def load_validations(roots):
    records = {}; seen = set()
    for root in roots:
        for path in Path(root).rglob('results.json'):
            raw = path.read_bytes(); digest = digest_bytes(raw)
            if digest in seen: continue
            seen.add(digest); record = json.loads(raw)
            if record.get('schema') != 'llama1b-realized-execution-check-v1': continue
            key = record['matched_results_sha256']
            if key in records:
                if not isinstance(records[key], list): records[key] = [records[key]]
                records[key].append(record)
            else: records[key] = record
    return records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-root', required=True)
    parser.add_argument('--validation-root', action='append', default=[])
    parser.add_argument('--split', choices=['development','validation'], default='validation')
    parser.add_argument('--min-processes', type=int, default=3)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.min_processes < 3: parser.error('At least three fresh processes are required')
    out = Path(args.out)
    if out.exists(): parser.error('Use a new output path; prior summaries are immutable')
    expected = [x['id'] for x in json.loads((Path(__file__).parents[1]/'reference/prompts-v1.json').read_text())['prompts'] if x['split'] == args.split]
    groups = defaultdict(list)
    for path in Path(args.results_root).rglob('results.json'):
        raw = path.read_bytes(); run = json.loads(raw)
        if 'configuration' not in run or 'fixture_manifest' not in run: continue
        if run['fixture_manifest'].get('split') != args.split: continue
        run['_receipt_sha256'] = digest_bytes(raw); run['_receipt_path'] = str(path)
        config = run['configuration']
        identity = {'sources': run.get('sources'), 'harness': run.get('harness_sha256'),'harness_sources':run.get('harness_sources'),'sequence_glue_sha256':run.get('sequence_glue_sha256'),'sequence_glue_sources':run.get('sequence_glue_sources'),'logits_layouts':run.get('logits_layouts'), 'upstream': run.get('upstream_source_identity'),'freeze':(run['fixture_manifest'].get('candidate_freeze') or {}).get('sha256'),'suite':run['fixture_manifest'].get('suite_sha256'),'protocol':((run['fixture_manifest'].get('candidate_freeze') or {}).get('contents') or {}).get('protocol_sha256'),'reference_sources':run['fixture_manifest'].get('source_sha256'),'reference_runtime':run['fixture_manifest'].get('runtime'),'fixture_manifest_sha256':run.get('fixture_manifest_sha256'),
                    'control': config.get('control'), 'mode': config.get('mode'), 'max_len': config.get('max_len'),
                    'samples': config.get('samples'), 'warmup': config.get('warmup'), 'gpu': run.get('gpu', {}).get('uuid'),
                    'runtime': run.get('runtime'), 'environment': run.get('environment'), 'model': run['fixture_manifest'].get('model_sha256')}
        groups[json.dumps(identity, sort_keys=True)].append(run)
    validations = load_validations([args.results_root] + args.validation_root)
    results = [{'group': json.loads(key), 'summary': summarize(runs, expected, args.min_processes, validations)} for key, runs in groups.items()]
    out.write_text(json.dumps({'groups': results, 'summary_source_sha256': digest_bytes(Path(__file__).read_bytes())}, indent=2)+'\n')

if __name__ == '__main__': main()
