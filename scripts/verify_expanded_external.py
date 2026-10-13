"""Reproduce external estimates from archived risk and survival tables.

Frozen adapter/output hashes are checked. Full expression inputs are
required only with --verify-predictions; they are stored separately.
"""
from pathlib import Path
import argparse, json, hashlib, math
import numpy as np
import pandas as pd
from scipy.stats import norm
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def pool(records):
    y=np.array([r['log_hr'] for r in records]);v=np.array([r['se_log_hr']**2 for r in records]);w=1/v
    m=np.sum(w*y)/w.sum();q=float(np.sum(w*(y-m)**2));k=len(y)
    tau=max(0.,(q-(k-1))/(w.sum()-(w*w).sum()/w.sum()))
    results=[]
    for name,t in [('fixed_effect',0.),('random_effects_DL',tau)]:
        weights=1/(v+t);mu=float(np.sum(weights*y)/weights.sum());se=float(np.sqrt(1/weights.sum()))
        results.append({'model':name,'hr_per_sd':math.exp(mu),'ci_low':math.exp(mu-1.96*se),'ci_high':math.exp(mu+1.96*se),'p':float(2*norm.sf(abs(mu/se))),'Q':q,'I2':100*max(0.,(q-(k-1))/q) if q>0 else 0.,'tau2':tau})
    return results

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/expanded_external'))
    parser.add_argument('--verify-predictions', action='store_true', help='Require separately supplied full selected-expression inputs.')
    args=parser.parse_args(); revision=args.root/'revision_20261012'
    reference=json.loads((revision/'expanded_external/audit_results.json').read_text(encoding='utf-8'))
    assert digest(revision/'external_analysis_plan.json')==reference['analysis_plan_sha256']
    adapter_checks={}
    for cohort in ['GSE57495','E-MTAB-6134']:
        folder=revision/'frozen_predictions'/cohort
        manifest=json.loads((folder/'frozen_manifest.json').read_text(encoding='utf-8'))
        for filename,value in manifest['outputs_sha256'].items(): assert digest(folder/filename)==value
        input_path=revision/'inference_inputs'/cohort/'selected_expression.csv.gz'
        if not input_path.exists():
            if args.verify_predictions:
                raise FileNotFoundError(f'Restore the separately stored expression input: {input_path}')
            adapter_checks[cohort]={'artifact_hashes_verified':True,'expression_to_score_reconstruction':'not run; full expression input stored separately'}
            continue
        model=np.load(folder/'adapter.npz',allow_pickle=False)
        x=pd.read_csv(revision/'inference_inputs'/cohort/'selected_expression.csv.gz',index_col=0,float_precision='round_trip')
        saved=pd.read_csv(folder/'predictions.csv',index_col=0,float_precision='round_trip')
        assert not x.index.duplicated().any() and set(x.index)==set(saved.index)
        a=x.loc[saved.index,model['genes']].fillna(0).to_numpy(float)
        scores=((a-model['mean'])/model['scale'])@model['coef']+float(model['intercept'])
        error=float(np.max(np.abs(scores-saved['student_risk_score'].to_numpy())))
        assert error<1e-10,(cohort,error)
        adapter_checks[cohort]={'profiles':len(x),'selected_genes':len(model['genes']),'max_abs_score_error':error}
    records=[]
    for expected in reference['external']:
        cohort=expected['cohort']
        d=pd.read_csv(revision/'expanded_external'/f'{cohort}_survival_scores.csv')
        assert not d.PATIENT_ID.duplicated().any()
        assert d.OS_TIME.gt(0).all() and d.OS_EVENT.isin([0,1]).all()
        d['risk_z']=(d.student_risk_score-d.student_risk_score.mean())/d.student_risk_score.std(ddof=0)
        fit=CoxPHFitter().fit(d[['OS_TIME','OS_EVENT','risk_z']],duration_col='OS_TIME',event_col='OS_EVENT')
        r=fit.summary.loc['risk_z']
        actual={'cohort':cohort,'role':expected['role'],'n_survival':len(d),'events':int(d.OS_EVENT.sum()),'log_hr':float(r['coef']),'se_log_hr':float(r['se(coef)']),'hr_per_sd':float(r['exp(coef)']),'ci_low':float(r['exp(coef) lower 95%']),'ci_high':float(r['exp(coef) upper 95%']),'p':float(r['p']),'c_index':float(concordance_index(d.OS_TIME,-d.risk_z,d.OS_EVENT))}
        for key in ['hr_per_sd','ci_low','ci_high','p','c_index']: assert abs(actual[key]-expected[key])<1e-6,(cohort,key)
        assert actual['n_survival']==expected['n_survival'] and actual['events']==expected['events']
        records.append(actual)
    independent=[r for r in records if r['role']=='independent primary OS']
    assert len(independent)==5 and sum(r['n_survival'] for r in independent)==618 and sum(r['events'] for r in independent)==412
    meta=pool(independent); loo=[]
    for cohort in [r['cohort'] for r in independent]:
        for result in pool([r for r in independent if r['cohort']!=cohort]): loo.append({'omitted_cohort':cohort,**result})
    for actual,expected in zip(meta,reference['meta']):
        for key in ['hr_per_sd','ci_low','ci_high','p','I2','tau2']: assert abs(actual[key]-expected[key])<1e-6,key
    for actual,expected in zip(loo,reference['leave_one_out']):
        assert actual['omitted_cohort']==expected['omitted_cohort'] and actual['model']==expected['model']
        for key in ['hr_per_sd','ci_low','ci_high','p']: assert abs(actual[key]-expected[key])<1e-6,key
    args.output_dir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(args.output_dir/'external_summary.csv',index=False)
    pd.DataFrame(meta).to_csv(args.output_dir/'meta_summary.csv',index=False)
    pd.DataFrame(loo).to_csv(args.output_dir/'leave_one_out.csv',index=False)
    result={'verified':True,'patients':618,'events':412,'adapter_checks':adapter_checks,'meta':meta,'all_ten_omission_results_reproduced':True}
    (args.output_dir/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
