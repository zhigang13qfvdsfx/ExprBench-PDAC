"""Fit the locked TCGA-only platform adapter; never read external outcomes.

Usage: python freeze_new_predictions.py --cohort GSE57495
Paths are relative to this revision workspace. Coefficients and scores are
serialized before the separate external survival audit is run.
"""
from pathlib import Path
import sys, json, hashlib, re, argparse, warnings
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'revision_20261012'
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

p=argparse.ArgumentParser();p.add_argument('--cohort',required=True);p.add_argument('--output-dir',type=Path)
a=p.parse_args(); cohort=a.cohort
PROJECT=ROOT/'pdac_project'
OUT=a.output_dir or ROOT/'outputs/refit'/cohort;OUT.mkdir(exist_ok=True,parents=True)
assert not (OUT/'frozen_manifest.json').exists(), 'Previously frozen model: reuse, do not overwrite.'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
plan=WORK/'external_analysis_plan.json'
assert sha(plan)=='9f718d67a0c7cdc773e7e953ef21d81c4e6d7cee953abcce40c379ed86eac7d6'
xp=PROJECT/'results/01_tcga_prep/expression_matrix.csv'
rp=PROJECT/'results/02_benchmark/artifacts/full_cohort_risk_scores.csv'
ep=WORK/'new_cohorts'/cohort/'expression.csv.gz'
prov=WORK/'new_cohorts'/cohort/'provenance.json'
assert 'required before modeling' not in prov.read_text(encoding='utf-8')
x=pd.read_csv(xp,index_col=0)
x.columns=[re.sub(r'\.\d+$','',str(g).strip()).upper() for g in x.columns]
x=x.loc[:,~x.columns.duplicated()]
y=pd.read_csv(rp,index_col=0)['Expr-Penalized-Cox']
ids=sorted(set(x.index)&set(y.index));x=x.loc[ids];y=y.loc[ids]
assert len(ids)==166 and y.notna().all()
ext=pd.read_csv(ep,index_col=0)
shared=sorted(set(x.columns)&set(ext.columns));assert len(shared)>=500
genes=x[shared].var(axis=0).sort_values(ascending=False).head(3000).index.tolist()
train=x[genes].apply(pd.to_numeric,errors='coerce').fillna(0).to_numpy(float)
test=ext[genes].apply(pd.to_numeric,errors='coerce').fillna(0).to_numpy(float)
scaler=StandardScaler();tr=scaler.fit_transform(train);te=scaler.transform(test)
print(f'{cohort}: fitting TCGA-only adapter; {len(shared)} shared genes; {len(genes)} selected',flush=True)
with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
    model=ElasticNetCV(l1_ratio=[.1,.5,.9],n_alphas=50,cv=3,max_iter=4000,random_state=42,n_jobs=1)
    model.fit(tr,y.to_numpy(float))
pred=model.predict(te)
pd.DataFrame({'PATIENT_ID':ext.index,'student_risk_score':pred}).to_csv(OUT/'predictions.csv',index=False)
np.savez_compressed(OUT/'adapter.npz',genes=np.array(genes),mean=scaler.mean_,scale=scaler.scale_,coef=model.coef_,intercept=np.array(model.intercept_),training_ids=np.array(ids),alpha=np.array(model.alpha_),l1_ratio=np.array(model.l1_ratio_))
reload=np.load(OUT/'adapter.npz',allow_pickle=False)
reconstructed=((test-reload['mean'])/reload['scale'])@reload['coef']+float(reload['intercept'])
assert np.max(np.abs(pred-reconstructed))<1e-10
manifest={'cohort':cohort,'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'stage':'predictions saved before external outcomes joined','plan_sha256':sha(plan),'inputs_sha256':{str(q.relative_to(ROOT)):sha(q) for q in [xp,rp,ep,prov]},'teacher':'Expr-Penalized-Cox archived TCGA risk','n_training':len(ids),'n_external_profiles':len(ext),'n_shared_genes':len(shared),'n_selected_genes':len(genes),'alpha':float(model.alpha_),'l1_ratio':float(model.l1_ratio_),'n_iter':int(model.n_iter_),'dual_gap':float(model.dual_gap_),'training_surrogate_r2':float(model.score(tr,y.to_numpy(float))),'warnings':list(dict.fromkeys(str(w.message) for w in caught)),'outputs_sha256':{q.name:sha(q) for q in [OUT/'adapter.npz',OUT/'predictions.csv']},'prediction_reconstruction_max_abs_error':float(np.max(np.abs(pred-reconstructed)))}
(OUT/'frozen_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest,indent=2),flush=True)
