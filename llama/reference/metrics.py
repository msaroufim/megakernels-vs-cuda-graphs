"""FP64 scalar reference metrics; no Torch dependency, thresholds or token filtering."""
import heapq
import math
import statistics


def logit_metrics(reference, actual, target_id):
    """Reference is FP32-HF; actual is BF16-HF or another tested implementation.

    Inputs are one-dimensional Python float sequences. Python arithmetic/exp and
    fsum provide FP64 evaluation independently of the model's CUDA kernels.
    NLL scores the supplied canonical token, not a ground-truth continuation.
    """
    r, a = list(reference), list(actual)
    if len(r) != len(a) or len(r) < 2 or not 0 <= target_id < len(r):
        raise ValueError('Equal vocabulary dimensions and an in-range target are required')
    result = {'reference_finite': all(math.isfinite(x) for x in r),
              'actual_finite': all(math.isfinite(x) for x in a), 'target_id': int(target_id)}
    if not result['reference_finite'] or not result['actual_finite']:
        return result
    ri = heapq.nlargest(2, range(len(r)), key=r.__getitem__)
    ai = heapq.nlargest(2, range(len(a)), key=a.__getitem__)
    rmax, amax = r[ri[0]], a[ai[0]]
    rz = math.log(math.fsum(math.exp(x-rmax) for x in r))
    az = math.log(math.fsum(math.exp(x-amax) for x in a))
    rp = [(x-rmax)-rz for x in r]
    ap = [(x-amax)-az for x in a]
    error_square = math.fsum((x-y)**2 for x,y in zip(r,a))
    r_square = math.fsum(x*x for x in r)
    rmean, amean = statistics.fmean(r), statistics.fmean(a)
    centered_error = math.fsum(((x-rmean)-(y-amean))**2 for x,y in zip(r,a))
    centered_reference = math.fsum((x-rmean)**2 for x in r)
    result.update({
        'relative_l2': math.sqrt(error_square)/max(math.sqrt(r_square),1e-150),
        'centered_relative_l2': math.sqrt(centered_error)/max(math.sqrt(centered_reference),1e-150),
        'max_abs': max(abs(x-y) for x,y in zip(r,a)),
        'kl_reference_to_actual': math.fsum(math.exp(x)*(x-y) for x,y in zip(rp,ap)),
        'reference_nll': -rp[target_id], 'actual_nll': -ap[target_id],
        'nll_delta': rp[target_id]-ap[target_id],
        'reference_entropy': -math.fsum(math.exp(x)*x for x in rp),
        'reference_top1': ri[0], 'actual_top1': ai[0],
        'top1_agreement': ri[0] == ai[0],
        'reference_top2_margin': r[ri[0]]-r[ri[1]],
        'actual_top2_margin': a[ai[0]]-a[ai[1]],
        'reference_margin_for_actual_choice': rmax-r[ai[0]],
    })
    return result


def quantile(values, probability):
    values = sorted(values)
    if not values:
        return None
    index = probability*(len(values)-1)
    left = int(index)
    right = min(left+1,len(values)-1)
    return values[left]+(values[right]-values[left])*(index-left)


def summarize(rows):
    """Descriptive envelope only; retains counts and never defines acceptance."""
    valid = [r for r in rows if r['reference_finite'] and r['actual_finite']]
    metrics = {}
    for key in ['relative_l2','centered_relative_l2','max_abs','kl_reference_to_actual',
                'reference_nll','actual_nll','nll_delta','reference_top2_margin',
                'actual_top2_margin','reference_margin_for_actual_choice']:
        values = [r[key] for r in valid]
        metrics[key] = {'mean': statistics.fmean(values) if values else None,
                        'min': min(values) if values else None,
                        'p50': quantile(values,.5),'p95':quantile(values,.95),
                        'max': max(values) if values else None}
    return {'rows':len(rows),'finite_rows':len(valid),
            'greedy_disagreements':sum(not r['top1_agreement'] for r in valid),
            'metrics':metrics,'acceptance_threshold':None}
