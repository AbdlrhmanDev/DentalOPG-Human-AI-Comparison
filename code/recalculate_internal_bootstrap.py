#!/usr/bin/env python3
"""Recalculate the NEW internal paired 20,000-replicate bootstrap (seed 20261112).

Input: data/derived/per_image_detection_counts.csv from this repository.
The reference-image counts are the units of resampling. Does not claim to
reproduce the unknown historical random seed, retrain models, or re-run inference.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data' / 'derived' / 'per_image_detection_counts.csv'
OUT_MAIN = ROOT / 'results' / 'main_tables' / 'Table_4_Paired_Bootstrap.csv'
OUT_DETAIL = ROOT / 'results' / 'statistical_audit' / 'Bonferroni_Adjusted_Bootstrap_CIs.csv'
SEED, B, N = 20261112, 20000, 154
EXPECTED_SHA256 = '0088a4a374fc9a80fa5ea748a7ddbfa8df8ed436e4df7fa63b63a8544f0fbbee'
PARTICIPANTS = ['YOLOv8s','YOLO11s','RT-DETR-L','Dentist 01','Dentist 02']


def f1(totals):
    tp, fp, fn = totals[...,0], totals[...,1], totals[...,2]
    den = 2.0*tp + fp + fn
    return np.divide(200.0*tp, den,
                     out=np.zeros_like(den,dtype=float),where=den!=0)


def load_primary():
    h = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if h != EXPECTED_SHA256:
        raise ValueError(f'Input SHA-256 mismatch; got {h}, expected {EXPECTED_SHA256}')
    with SOURCE.open(newline='',encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    keys=set(rows[0])
    assert {'case_id','participant','threshold','TP','FP','FN'}.issubset(keys)
    assert len(rows) == N*5*2, len(rows)
    caseids=sorted({row['case_id'] for row in rows})
    assert len(caseids)==N
    bykey={}
    for row in rows:
        key=(row['case_id'],row['participant'],float(row['threshold']))
        if key in bykey: raise ValueError(f'Duplicate row {key}')
        vals=tuple(int(row[k]) for k in ('TP','FP','FN'))
        if min(vals)<0: raise ValueError('Negative detection count')
        bykey[key]=vals
    assert set(row['participant'] for row in rows)==set(PARTICIPANTS)
    assert set(float(row['threshold']) for row in rows)=={0.25,0.5}
    matrices={}
    for threshold in (0.25,0.5):
        matrices[threshold]=np.array([
            [bykey[(case,p,threshold)] for p in PARTICIPANTS]
            for case in caseids
        ],dtype=np.int64)
        assert matrices[threshold].shape == (N,5,3)
        for i in range(5):
            assert (matrices[threshold][:,i,0]+matrices[threshold][:,i,2]).sum()==588
        assert np.all((matrices[threshold][:,:,0]+matrices[threshold][:,:,2]) ==
                      (matrices[threshold][:,[0],0]+matrices[threshold][:,[0],2]))
    assert np.all(matrices[0.25][:,:,0] >= matrices[0.5][:,:,0])
    return matrices


def write_table(path, names, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(names);w.writerows(values)


def main():
    matrices=load_primary()
    m=matrices[0.5]
    obs=f1(m.sum(axis=0))
    rng=np.random.default_rng(SEED)
    boot=np.empty((B,5),dtype=float)
    for start in range(0,B,500):
        size=min(500,B-start)
        indices=rng.integers(0,N,size=(size,N))
        boot[start:start+size]=f1(m[indices].sum(axis=1))
    detailed=[];main=[]
    for ai in PARTICIPANTS[:3]:
        for dentist in PARTICIPANTS[3:]:
            i,j=PARTICIPANTS.index(ai),PARTICIPANTS.index(dentist)
            diff=boot[:,i]-boot[:,j]
            nominal=np.percentile(diff,[2.5,97.5])
            adj=np.percentile(diff,[100*(.05/6/2),100*(1-.05/6/2)])
            point=obs[i]-obs[j]
            excluded=bool(adj[0]>0 or adj[1]<0)
            detailed.append([ai,dentist,f'{point:.2f}',f'{nominal[0]:.2f}',
                    f'{nominal[1]:.2f}',f'{adj[0]:.2f}',f'{adj[1]:.2f}',excluded])
            main.append([f'{ai} vs {dentist}',f'{point:.2f}',
                         f'{adj[0]:.2f}',f'{adj[1]:.2f}',excluded])
    write_table(OUT_MAIN,
       ['Comparison','AI minus dentist F1 (pp)','Bonferroni CI lower (pp)',
        'Bonferroni CI upper (pp)','Adjusted CI excludes zero'],main)
    write_table(OUT_DETAIL,
       ['AI model','Dentist','Difference (pp)','Nominal 95% lower',
        'Nominal 95% upper','Bonferroni CI lower','Bonferroni CI upper',
        'Adjusted CI excludes zero'],detailed)
    print('Internal paired bootstrap complete; replicates:',B,'seed:',SEED)
    print('Wrote:',OUT_MAIN)
    for row in main: print(row)

if __name__=='__main__': main()
