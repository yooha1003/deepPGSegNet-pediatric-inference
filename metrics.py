"""Participant-level overlap and volume-agreement summaries."""
import numpy as np
from scipy import stats


def icc_absolute(reference, predicted):
    values = np.column_stack([reference, predicted]).astype(float)
    n,k = values.shape
    grand = values.mean()
    row_mean = values.mean(axis=1)
    col_mean = values.mean(axis=0)
    ms_rows = k*np.sum((row_mean-grand)**2)/(n-1)
    ms_cols = n*np.sum((col_mean-grand)**2)/(k-1)
    residual = values-row_mean[:,None]-col_mean[None,:]+grand
    ms_error = np.sum(residual**2)/((n-1)*(k-1))
    denominator = ms_rows+(k-1)*ms_error+k*(ms_cols-ms_error)/n
    return (ms_rows-ms_error)/denominator if denominator else np.nan


def bootstrap_ci(values, statistic=np.mean, repeats=10000, seed=260917):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0,len(values),(repeats,len(values)))]
    result = np.array([statistic(sample) for sample in samples])
    return np.nanpercentile(result,[2.5,97.5]).tolist()


def summarize(frame):
    result = {'n':len(frame)}
    for metric in ['dice','signed_error_pct','absolute_error_pct','absolute_error_mm3','hd95']:
        vals=frame[metric].to_numpy()
        result[metric]={'mean':float(np.nanmean(vals)), 'sd':float(np.nanstd(vals,ddof=1)),
                        'ci':bootstrap_ci(vals, np.nanmean), 'defined_n':int(np.isfinite(vals).sum())}
    ref=frame.ref_mm3.to_numpy(); pred=frame.pred_mm3.to_numpy()
    delta=pred-ref
    result['pearson_r']=float(stats.pearsonr(ref,pred).statistic)
    result['icc']=float(icc_absolute(ref,pred))
    result['icc_ci']=bootstrap_ci(np.column_stack([ref,pred]),lambda a:icc_absolute(a[:,0],a[:,1]))
    result['bias_mm3']=float(delta.mean())
    result['loa_mm3']=[float(delta.mean()-1.96*delta.std(ddof=1)),float(delta.mean()+1.96*delta.std(ddof=1))]
    result['near_complete_failures']=int((frame.dice<0.1).sum())
    return result


def paired_comparison(a,b):
    delta=np.asarray(a)-np.asarray(b)
    observed=abs(delta.mean())
    signs=2*((np.arange(2**len(delta))[:,None] >> np.arange(len(delta)))&1)-1
    perm=np.abs((signs*delta).mean(axis=1))
    return {'difference':float(delta.mean()),'ci':bootstrap_ci(delta),
            'paired_t_p':float(stats.ttest_rel(a,b).pvalue),
            'exact_signflip_p':float(np.mean(perm>=observed-1e-14))}
