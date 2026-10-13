from pathlib import Path
import sys, gzip, csv, io, re, json, hashlib
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'revision_20261012'
import pandas as pd
import numpy as np
SRC=WORK/'sources'; OUT=WORK/'new_cohorts';OUT.mkdir(exist_ok=True,parents=True)
PROJECT=ROOT/'pdac_project'
plan=WORK/'external_analysis_plan.json'
plan_hash=hashlib.sha256(plan.read_bytes()).hexdigest()

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def gene(g):return re.sub(r'\.\d+$','',str(g).strip()).upper()
def phenotype(header):
    ids=header['!Sample_geo_accession'][0]
    records=[]
    for i,sid in enumerate(ids):
        d={'PATIENT_ID':sid,'title':header['!Sample_title'][0][i]}
        for k,values in header.items():
            if 'characteristics' in k:
                for row in values:
                    if ':' in row[i]:
                        key,value=row[i].split(':',1);d[key.strip().lower()]=value.strip()
        records.append(d)
    return pd.DataFrame(records).set_index('PATIENT_ID')
def platform(gpl):
    lines=[];on=False
    with gzip.open(SRC/(gpl+'_platform_only.soft.gz'),'rt',encoding='utf-8',errors='replace') as f:
        for line in f:
            if line.startswith('!platform_table_begin'):on=True;continue
            if line.startswith('!platform_table_end'):break
            if on:lines.append(line)
    d=pd.read_csv(io.StringIO(''.join(lines)),sep='\t',low_memory=False,dtype=str)
    print('platform',gpl,'columns',d.columns.tolist(),flush=True)
    sym=next((c for c in d.columns if re.sub(r'[ _]','',c.lower()) in ['genesymbol','symbol']),None)
    if sym is None:raise ValueError('No explicit gene symbol column in '+gpl)
    def single(v):
        if pd.isna(v):return ''
        tokens=[x.strip() for x in re.split(r'\s*///\s*|\s*//\s*|;',v) if x.strip() and x.strip() not in ['---','NA','N/A']]
        tokens=list(dict.fromkeys(tokens))
        return gene(tokens[0]) if len(tokens)==1 else ''
    return dict(zip(d.iloc[:,0].astype(str),d[sym].map(single)))
def expression(cohort,gpl):
    lines=[];on=False
    with gzip.open(SRC/(cohort+'_series_matrix.txt.gz'),'rt',encoding='utf-8',errors='replace') as f:
        for line in f:
            if line.startswith('!series_matrix_table_begin'):on=True;continue
            if line.startswith('!series_matrix_table_end'):break
            if on:lines.append(line)
    d=pd.read_csv(io.StringIO(''.join(lines)),sep='\t',low_memory=False)
    mp=platform(gpl)
    symbols=d.iloc[:,0].astype(str).map(mp).fillna('')
    x=d.iloc[:,1:].apply(pd.to_numeric,errors='coerce');x.index=symbols
    x=x.loc[x.index!=''].groupby(level=0,sort=True).mean().T
    x.index.name='PATIENT_ID'
    return x
def save(cohort,x,c,audit,metadata):
    folder=OUT/cohort;folder.mkdir(exist_ok=True)
    x.to_csv(folder/'expression.csv.gz')
    c.to_csv(folder/'clinical.csv')
    audit.to_csv(folder/'sample_audit.csv')
    metadata.update({'analysis_plan_sha256':plan_hash,'n_expression':len(x),'n_genes':x.shape[1],'n_endpoint_annotations':int(c[['OS_TIME','OS_EVENT']].notna().all(axis=1).sum()),'expression_quantiles':np.nanquantile(x.to_numpy(),[0,.01,.5,.99,1]).tolist()})
    (folder/'provenance.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'cohort':cohort,**metadata},ensure_ascii=True),flush=True)

h=json.loads((WORK/'GSE71729_metadata.json').read_text(encoding='utf-8'))
ph=phenotype(h)
primary=ph['tissue type'].eq('Primary')
primary_ids=ph.index[primary]
c=pd.DataFrame(index=primary_ids)
c['OS_TIME']=pd.to_numeric(ph.loc[primary_ids,'survival_months'],errors='coerce')
c['OS_EVENT']=pd.to_numeric(ph.loc[primary_ids,'death_event_1death_0censor'],errors='coerce')
legacy=PROJECT/'results'/'07_external_validation_rescue'
x=pd.read_csv(legacy/'GSE71729_expression_processed.csv',index_col=0).loc[primary_ids]
ph['eligible_primary_tumor']=primary
ph['patient_key']=ph.title.str.split('-').str[0]
save('GSE71729',x,c,ph,{'source':'GEO GSE71729 second-channel phenotype; unmodified archived expression and predictions','endpoint':'overall survival months; 1 death / 0 censored','expression_scale':'GEO log2 Cy5 signal, negative values set to zero, gene-symbol means and source quantile normalization','n_raw_profiles':357,'n_primary_tumor_profiles':int(primary.sum())})

h=json.loads((SRC/'GSE57495_metadata.json').read_text(encoding='utf-8'))
ph=phenotype(h)
c=pd.DataFrame(index=ph.index)
c['OS_TIME']=pd.to_numeric(ph['overall survival (month)'],errors='coerce')
print('GSE57495 vital labels',ph['vital.status'].value_counts().to_dict(),flush=True)
c['OS_EVENT']=ph['vital.status'].str.lower().map({'dead':1,'deceased':1,'alive':0,'living':0,'1':1,'0':0})
assert c['OS_EVENT'].notna().all(),ph['vital.status'].unique()
x=expression('GSE57495','GPL15048')
ph['eligible_primary_tumor']=True
ph['patient_key']=ph.title
save('GSE57495',x,c,ph,{'source':'GEO GSE57495, GPL15048 platform gene-symbol annotations','endpoint':'overall survival (month); vital.status dead=1 / alive=0','expression_scale':'source RMA normalized array values','n_raw_profiles':len(ph),'n_primary_tumor_profiles':len(ph)})


OUT=WORK/'new_cohorts/E-MTAB-6134'
OUT.mkdir(parents=True,exist_ok=True)
lines=[];on=False
with gzip.open(SRC/'GPL13667_platform_only.soft.gz','rt',encoding='utf-8',errors='replace') as f:
    for line in f:
        if line.startswith('!platform_table_begin'):on=True;continue
        if line.startswith('!platform_table_end'):break
        if on:lines.append(line)
ann=pd.read_csv(io.StringIO(''.join(lines)),sep='\t',dtype=str,low_memory=False)
symbol=next(c for c in ann.columns if re.sub(r'[ _]','',c.lower())=='genesymbol')
def gene(s):
    if pd.isna(s):return ''
    tokens=list(dict.fromkeys(x.strip().upper() for x in re.split(r'\s*///\s*|;',s) if x.strip() and x.strip()!='---'))
    return re.sub(r'\.\d+$','',tokens[0]) if len(tokens)==1 else ''
mp=dict(zip(ann.iloc[:,0].astype(str),ann[symbol].map(gene)))
d=pd.read_csv(SRC/'ProcessedExpression.tsv.gz',sep='\t',index_col=0,low_memory=False)
sy=pd.Series(d.index.astype(str),index=d.index).map(mp).fillna('')
mapped=d.loc[sy.ne('')].copy();mapped.index=sy.loc[sy.ne('')]
x=mapped.groupby(level=0,sort=True).mean().T;x.index.name='PATIENT_ID'
primary=pd.read_csv(SRC/'E-MTAB-6134.sdrf.txt',sep='\t',dtype=str).set_index('Source Name')
assert set(primary.index)==set(x.index) and len(x)==309
c=pd.DataFrame(index=x.index)
c.index.name='PATIENT_ID'
c['OS_TIME']=pd.to_numeric(primary.loc[x.index,'Characteristics[os.delay]'],errors='coerce')
c['OS_EVENT']=pd.to_numeric(primary.loc[x.index,'Characteristics[os.event]'],errors='coerce')
assert primary.loc[c.index[c.OS_TIME.notna()],'Unit[time unit].1'].str.lower().eq('month').all()
c.to_csv(OUT/'clinical.csv')
primary.index.name='PATIENT_ID'
primary.to_csv(OUT/'sample_audit.csv')
assert set(c.index)==set(x.index)
assert np.isfinite(x.to_numpy(float)).all()
x.to_csv(OUT/'expression.csv.gz')
raw=json.loads((SRC/'ProcessedExpression_provenance.json').read_text(encoding='utf-8'))
p={'source':'Deposited EBI E-MTAB-6134 ProcessedExpression.tsv and primary SDRF; GPL13667 U219 annotation','expression_source_sha256':raw['source_sha256'],'endpoint':'SDRF os.delay months, os.event 1 death / 0 censored','expression_scale':'Source RMA log2-scale array values with clinical-center ComBat correction documented in IDF; mean values of unambiguously gene-symbol-mapped probes','n_raw_profiles':309,'n_primary_tumor_profiles':309,'n_raw_probes':len(d),'n_unambiguously_mapped_probes':len(mapped),'n_expression':len(x),'n_genes':x.shape[1],'n_endpoint_annotations':int(c[['OS_TIME','OS_EVENT']].notna().all(axis=1).sum()),'expression_quantiles':np.nanquantile(x.to_numpy(),[0,.01,.5,.99,1]).tolist(),'analysis_plan_sha256':sha(WORK/'external_analysis_plan.json'),'curated_data_audit':'pdacR parse_Puleo_array.R sums probe values and reverses the SDRF event into a censor indicator. For scoring, use mean probe values from the deposited matrix; for inference, use primary SDRF death-event definitions.','platform_annotation_sha256':sha(SRC/'GPL13667_platform_only.soft.gz')}
(OUT/'provenance.json').write_text(json.dumps(p,indent=2),encoding='utf-8')
print(json.dumps(p,indent=2),flush=True)
