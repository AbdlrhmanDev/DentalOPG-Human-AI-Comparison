from pathlib import Path
from zipfile import ZipFile
from lxml import etree
from collections import Counter, defaultdict
from scipy.optimize import linear_sum_assignment
import numpy as np, pandas as pd, re, json

PROJECT=Path(__file__).resolve().parent
ROOT=PROJECT/'Inputs'
OUT=PROJECT/'Results'
OUT.mkdir(exist_ok=True)
CLASS_NAMES=['Caries','Missing Teeth','Root Canal Treatment','Filling','Crown','Unerupted Wisdom Teeth']
CASES=[f'CASE_{x:04d}' for x in range(1,155)]

def parse_zip(zipname,reader):
    out=[]; meta=[]; bad=[]; sizes=Counter(); all_labels=Counter(); geom=Counter(); duplicates=[]
    with ZipFile(ROOT/zipname) as z:
      root=etree.fromstring(z.read('annotations.xml'))
      for image in root.findall('image'):
        nm=image.get('name'); match=re.search(r'CASE_\d{4}',nm or '')
        if not match: bad.append(('missing_case_name',nm)); continue
        case=match.group(); W=int(image.get('width'));H=int(image.get('height'))
        assert W>0 and H>0
        sizes[(W,H)]+=1
        img_obj=[]
        for elem in image:
          geom[elem.tag]+=1
          if elem.tag!='box':continue
          label=elem.get('label'); all_labels[label]+=1
          if label not in CLASS_NAMES: bad.append(('unexpected_class',nm,label)); continue
          x1,y1,x2,y2=(float(elem.get(k)) for k in ('xtl','ytl','xbr','ybr'))
          if not (0<=x1<x2<=W and 0<=y1<y2<=H):bad.append(('invalid_box',nm,label,x1,y1,x2,y2))
          b=(x1/W,y1/H,x2/W,y2/H)
          if any(all(abs(a-bb)<1e-8 for a,bb in zip(b,old[:4])) and label==old[4] for old in img_obj):duplicates.append(nm)
          img_obj.append((*b,label))
          out.append(dict(reader=reader,case_id=case,cvat_filename=nm,class_id=CLASS_NAMES.index(label),class_name=label,x1=b[0],y1=b[1],x2=b[2],y2=b[3],width_px=x2-x1,height_px=y2-y1,source=elem.get('source')))
        meta.append(dict(reader=reader,case_id=case,cvat_filename=nm,width=W,height=H,boxes=len(img_obj)))
    return pd.DataFrame(out),pd.DataFrame(meta),bad,geom,sizes,all_labels,duplicates

D1,M1,b1,g1,s1,c1,du1=parse_zip('Dentist01_CVAT_Annotations.zip','Dentist 01')
D2,M2,b2,g2,s2,c2,du2=parse_zip('Dentist02_CVAT_Annotations.zip.zip','Dentist 02')
ref=pd.read_csv(ROOT/'reference_annotations.csv')
ref['case_id']=ref.cvat_filename.str.extract(r'(CASE_\d{4})')[0]
assert ref.case_id.notna().all()
R=ref.rename(columns={'x_center':'xc','y_center':'yc','width':'bw','height':'bh'})
R['x1']=R.xc-R.bw/2;R['x2']=R.xc+R.bw/2;R['y1']=R.yc-R.bh/2;R['y2']=R.yc+R.bh/2
ref_box=R[['case_id','class_id','x1','y1','x2','y2']]
# Validate cardinality and case uniqueness.
assert len(M1)==154 and len(M2)==154 and M1.case_id.nunique()==M2.case_id.nunique()==154
assert set(M1.case_id)==set(M2.case_id)==set(CASES)
assert set(ref.case_id).issubset(set(CASES))
assert len(ref)==588 and len(D1)==565 and len(D2)==606
assert not b1 and not b2

# Max-cardinality maximum-weight bipartite match at threshold.
def iou(a,b):
    xx1=max(a[0],b[0]); yy1=max(a[1],b[1]); xx2=min(a[2],b[2]); yy2=min(a[3],b[3])
    inter=max(0,xx2-xx1)*max(0,yy2-yy1)
    if inter<=0:return 0.0
    ar1=max(0,a[2]-a[0])*max(0,a[3]-a[1]);ar2=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/(ar1+ar2-inter)

by_d1={(case,cls):grp.reset_index(drop=True) for (case,cls),grp in D1.groupby(['case_id','class_id'])}
by_d2={(case,cls):grp.reset_index(drop=True) for (case,cls),grp in D2.groupby(['case_id','class_id'])}
by_ref={(case,cls):grp.reset_index(drop=True) for (case,cls),grp in ref_box.groupby(['case_id','class_id'])}


def match_data(group_a,group_b,thresh,who,versus):
    records=[]; matched=[]
    for case in CASES:
        for cls in range(6):
            da=group_a.get((case,cls));db=group_b.get((case,cls));
            aa=[] if da is None else da[['x1','y1','x2','y2']].to_numpy(dtype=float).tolist()
            bb=[] if db is None else db[['x1','y1','x2','y2']].to_numpy(dtype=float).tolist()
            n,m=len(aa),len(bb)
            hits=[]
            if n and m:
                mat=np.array([[iou(a,b) for b in bb] for a in aa])
                # Max-cardinality first, higher IoU second. Dummy pairs 0.
                weights=np.zeros((n+m,n+m),dtype=float)
                weights[:n,:m]=np.where(mat+1e-12>=thresh,2.0+mat,0)
                ii,jj=linear_sum_assignment(weights,maximize=True)
                hits=[(int(i),int(j),float(mat[i,j])) for i,j in zip(ii,jj) if i<n and j<m and mat[i,j]+1e-12>=thresh]
            tp=len(hits)
            records.append(dict(comparison=f'{who} vs {versus}',threshold=thresh,case_id=case,class_id=cls,class_name=CLASS_NAMES[cls],TP=tp,FP=n-tp,FN=m-tp,detections=n,reference_boxes=m))
            for i,j,score in hits:
                matched.append(dict(comparison=f'{who} vs {versus}',threshold=thresh,case_id=case,class_id=cls,class_name=CLASS_NAMES[cls],detection_index=i,reference_index=j,iou=score))
    return pd.DataFrame(records),pd.DataFrame(matched)

rows=[];matches=[]
for thresh in [0.50,0.25]:
 for d,tag in [(by_d1,'Dentist 01'),(by_d2,'Dentist 02')]:
    rd,ma=match_data(d,by_ref,thresh,tag,'Dataset reference')
    rows.append(rd);matches.append(ma)
 rd,ma=match_data(by_d1,by_d2,thresh,'Dentist 01','Dentist 02')
 rows.append(rd);matches.append(ma)
allrows=pd.concat(rows,ignore_index=True)
allmatches=pd.concat(matches,ignore_index=True)

def metric_rows(x,group_keys):
   s=x.groupby(group_keys,as_index=False)[['TP','FP','FN','detections','reference_boxes']].sum()
   s['precision']=s.TP/(s.TP+s.FP).replace({0:np.nan})
   s['recall']=s.TP/(s.TP+s.FN).replace({0:np.nan})
   s['F1']=2*s.TP/(2*s.TP+s.FP+s.FN).replace({0:np.nan})
   return s

summary=metric_rows(allrows,['comparison','threshold'])
per_class=metric_rows(allrows,['comparison','threshold','class_id','class_name'])
per_image=metric_rows(allrows,['comparison','threshold','case_id'])

# Reader1 prior values and reproducibility test
r1=summary[(summary.comparison=='Dentist 01 vs Dataset reference')]
print('REPRODUCED DENTIST 01',r1[['threshold','TP','FP','FN','precision','recall','F1']].to_string(index=False))
assert r1[r1.threshold==.50].TP.iloc[0]==321
assert r1[r1.threshold==.25].TP.iloc[0]==498

# Annotator QC
qc=[]
for name,df,md,bad,geom,sizes,cc,duplicates in [('Dentist 01',D1,M1,b1,g1,s1,c1,du1),('Dentist 02',D2,M2,b2,g2,s2,c2,du2)]:
    qc.append(dict(reader=name,images=len(md),images_with_boxes=(md.boxes>0).sum(),images_without_boxes=(md.boxes==0).sum(),boxes=len(df),masks=geom['mask'],polygons=geom['polygon'],invalid_boxes=len(bad),exact_duplicate_boxes=len(duplicates),thin_boxes_under_5px=int(((df.width_px<5)|(df.height_px<5)).sum()),image_dimensions=str(dict(sizes)),sources=str(df.source.value_counts().to_dict())))
qc=pd.DataFrame(qc)
cls_counts=pd.DataFrame({'class_id':range(6),'class_name':CLASS_NAMES,'dataset_reference':[int((ref.class_id==i).sum()) for i in range(6)],'dentist_01':[int((D1.class_id==i).sum()) for i in range(6)],'dentist_02':[int((D2.class_id==i).sum()) for i in range(6)]})

# Paired (resample 154 images simultaneously) bootstrap CI for F1 differences
rng=np.random.default_rng(20261014)
bootstrap=[]
for threshold in (.5,.25):
  set1=allrows[(allrows.comparison=='Dentist 01 vs Dataset reference')&(allrows.threshold==threshold)].groupby('case_id')[['TP','FP','FN']].sum().reindex(CASES).to_numpy()
  set2=allrows[(allrows.comparison=='Dentist 02 vs Dataset reference')&(allrows.threshold==threshold)].groupby('case_id')[['TP','FP','FN']].sum().reindex(CASES).to_numpy()
  def f1(t):return 2*t[...,0]/np.maximum(1,2*t[...,0]+t[...,1]+t[...,2])
  b_idx=rng.integers(0,154,size=(4000,154))
  s1=set1[b_idx].sum(axis=1);s2=set2[b_idx].sum(axis=1)
  dist=f1(s2)-f1(s1)
  obs=float(f1(set2.sum(0))-f1(set1.sum(0)))
  bootstrap.append(dict(threshold=threshold,dentist02_minus_dentist01_F1=obs,ci_low=float(np.quantile(dist,.025)),ci_high=float(np.quantile(dist,.975)),bootstrap_images=154,bootstrap_replicates=len(dist)))
bootstrap=pd.DataFrame(bootstrap)

# Write plain files
for name,df in [('QC_Summary',qc),('Class_Counts',cls_counts),('Summary_All_Thresholds',summary),('Per_Class_All_Thresholds',per_class),('Per_Image_All_Thresholds',per_image),('Matched_Box_Pairs',allmatches),('Doctor01_All_Boxes',D1),('Doctor02_All_Boxes',D2),('Bootstrap_F1_Differences',bootstrap)]:
 df.to_csv(OUT/f'{name}.csv',index=False)

print('QC\n',qc.to_string(index=False))
print('COUNTS\n',cls_counts.to_string(index=False))
print('SUMMARY\n',summary.to_string(index=False,float_format=lambda v:f'{v:.5f}'))
print('PERCLASS IOU 0.5\n',per_class[(per_class.threshold==.5)&(per_class.comparison!='Dentist 01 vs Dentist 02')].to_string(index=False,float_format=lambda v:f'{v:.5f}'))
print('PAIRED BOOTSTRAP\n',bootstrap.to_string(index=False,float_format=lambda v:f'{v:.5f}'))

# Extra investigate large matching overlap between doctors, total 6 classes.
# reference labels may include images not present in source; preserved for all CASE IDs.

