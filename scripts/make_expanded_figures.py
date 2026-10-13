from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'revision_20261012'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser(description='Regenerate Figure 7 from the archived external results.')
parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/revision_figures')
OUT=parser.parse_args().output_dir;OUT.mkdir(parents=True,exist_ok=True)
PROJECT=ROOT/'pdac_project'
a=json.loads((WORK/'expanded_external/audit_results.json').read_text(encoding='utf-8'))
ind=[r for r in a['external'] if r['role']=='independent primary OS']; fe,re=a['meta']
BLUE='#326B8E';GREEN='#86AFA1';ROSE='#8C5C65'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(13.3,10),gridspec_kw={'height_ratios':[1.05,1]})
ax=axs[0,0];forest=ind+[{'cohort':'Fixed-effect pool',**fe},{'cohort':'Random-effects pool',**re}]
for i,r in enumerate(forest):
    y=len(forest)-1-i
    ax.errorbar(r['hr_per_sd'],y,xerr=[[r['hr_per_sd']-r['ci_low']],[r['ci_high']-r['hr_per_sd']]],fmt='s' if i>=5 else 'o',color=ROSE if i==6 else BLUE,capsize=3,markersize=5)
    ax.text(2.38,y,f"{r['hr_per_sd']:.2f} ({r['ci_low']:.2f}-{r['ci_high']:.2f})",ha='left',va='center',fontsize=8.4)
ax.axvline(1,color='gray',linestyle='--');ax.set_yticks(range(7),[r['cohort'] for r in forest[::-1]],fontsize=9)
ax.set_xlim(.72,3.35);ax.set_xticks([1,1.5,2]);ax.set_xlabel('HR per cohort SD (95% CI)')
ax.set_title('A  All eligible independent cohort effects',loc='left',fontweight='bold')
ax=axs[0,1];ax.bar(np.arange(5),[r['c_index'] for r in ind],color=[BLUE,GREEN,BLUE,GREEN,BLUE])
ax.set_xticks(np.arange(5),[r['cohort'] for r in ind],rotation=23,ha='right',fontsize=9)
ax.set_ylim(.45,.68);ax.axhline(.5,color='gray',linestyle='--');ax.set_ylabel('Harrell C-index')
for i,r in enumerate(ind):ax.text(i,r['c_index']+.007,f"{r['c_index']:.3f}",ha='center',fontsize=9)
ax.set_title('B  Cohort-level discrimination',loc='left',fontweight='bold')
ax=axs[1,0]
for i,cohort in enumerate([r['cohort'] for r in ind]):
    rows=[r for r in a['leave_one_out'] if r['omitted_cohort']==cohort]
    for j,r in enumerate(rows):
        y=4-i+(.12 if j==0 else -.12)
        ax.errorbar(r['hr_per_sd'],y,xerr=[[r['hr_per_sd']-r['ci_low']],[r['ci_high']-r['hr_per_sd']]],fmt='o' if j==0 else 's',color=BLUE if j==0 else ROSE,capsize=2,markersize=4,label=['Fixed effect','Random effects'][j] if i==0 else None)
    ax.text(1.54,4-i,f"P={rows[0]['p']:.4f} / {rows[1]['p']:.4f}",va='center',fontsize=8)
ax.set_yticks(range(5),[f"Omit {r['cohort']}" for r in ind[::-1]],fontsize=9)
ax.axvline(1,color='gray',linestyle='--');ax.set_xlim(.94,1.91);ax.set_xticks([1,1.2,1.4,1.6]);ax.set_xlabel('HR per SD after omission (95% CI)')
ax.legend(frameon=False,fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.17),ncol=2);ax.set_title('C  Every leave-one-cohort-out result',loc='left',fontweight='bold')
ax=axs[1,1];ax.axis('off');ax.set_title('D  Endpoint and independence audit',loc='left',fontweight='bold')
lines=['Five cohorts: 618 eligible patients / 412 deaths',* [f"{r['cohort']}: {r['n_survival']} patients / {r['events']} deaths" for r in ind],'GSE71729: channel-2 OS fields recovered','GSE28735: overlapping cancer-death sensitivity','GSE21501: excluded for unresolved independence',f"Heterogeneity: I2={fe['I2']:.1f}%; tau2={fe['tau2']:.4f}"]
for i,line in enumerate(lines):ax.text(.02,.94-i*.088,line,transform=ax.transAxes,va='top',fontsize=9.8)
fig.tight_layout(pad=2.2,w_pad=2.5,h_pad=2.4);fig.savefig(OUT/'Figure_07_revised.png',dpi=300,facecolor='white');fig.savefig(OUT/'Figure_07_revised.pdf');plt.close(fig)


print('Reproduced current Figure 7.')
