#!/usr/bin/env python3
"""Reproduce MOPG-7 four-class external bounding-box evaluation from archived detections.

Inputs:
  * MOPG-7 Version 2 OFFICIAL test COCO annotations (instances_test.json).
  * Original model predictions saved at raw confidence 0.01.

Definitions: locked confidence thresholds 0.40/0.30/0.50; four class mapping;
class-consistent max-cardinality one-to-one matching, then maximal total IoU.
No model training or inference, no alteration of reference annotations.

IMPORTANT: This script verifies aggregate/individual matches ONLY when official COCO
reference is provided. Without it, previous aggregate results are not independent
bounding-box-level verification.
"""
import argparse, json, csv, math, sys, hashlib
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment

EXT_NAMES={0:'Missing Teeth',1:'Dental Crown',2:'Root Canal',3:'Caries'}
MODEL_TO_EXT={0:3,1:0,2:2,4:1}
CONF={'YOLOv8s':0.40,'YOLO11s':0.30,'RT-DETR-L':0.50}
PRED_FILES={'YOLOv8s':'YOLOv8s_predictions.json','YOLO11s':'YOLO11s_predictions.json','RT-DETR-L':'RT_DETR_L_predictions.json'}
THRESHOLDS=(0.50,0.25)
EXPECTED_REF={0:268,1:217,2:244,3:147}
EXPECTED_CLASS={
('YOLOv8s',.50,0):(0,213,268),('YOLOv8s',.50,1):(176,74,41),('YOLOv8s',.50,2):(99,97,145),('YOLOv8s',.50,3):(27,32,120),
('YOLO11s',.50,0):(0,340,268),('YOLO11s',.50,1):(193,110,24),('YOLO11s',.50,2):(46,58,198),('YOLO11s',.50,3):(34,62,113),
('RT-DETR-L',.50,0):(0,334,268),('RT-DETR-L',.50,1):(195,71,22),('RT-DETR-L',.50,2):(133,123,111),('RT-DETR-L',.50,3):(56,64,91),
('YOLOv8s',.25,0):(22,191,246),('YOLOv8s',.25,1):(191,59,26),('YOLOv8s',.25,2):(164,32,80),('YOLOv8s',.25,3):(39,20,108),
('YOLO11s',.25,0):(31,309,237),('YOLO11s',.25,1):(207,96,10),('YOLO11s',.25,2):(93,11,151),('YOLO11s',.25,3):(51,45,96),
('RT-DETR-L',.25,0):(57,277,211),('RT-DETR-L',.25,1):(202,64,15),('RT-DETR-L',.25,2):(204,52,40),('RT-DETR-L',.25,3):(78,42,69),
}

def basename(s): return str(s).replace('\\','/').rsplit('/',1)[-1].strip().lower()

def validate_box(b, w, h, context):
    if len(b)!=4 or not all(math.isfinite(float(v)) for v in b): raise ValueError('Invalid box '+context)
    x1,y1,x2,y2=map(float,b)
    if x2<=x1 or y2<=y1: raise ValueError('Nonpositive box '+context)
    # A few negligible numerical overflow pixels may be present in model outputs.
    if x1 < -1 or y1 < -1 or x2 > w+1 or y2 > h+1:
        raise ValueError('Box out of image bounds '+context+f' {(x1,y1,x2,y2)} vs {(w,h)}')
    return [max(0.,x1),max(0.,y1),min(float(w),x2),min(float(h),y2)]

def iou_array(a,b):
    a=np.asarray(a,dtype=float).reshape(-1,4)
    b=np.asarray(b,dtype=float).reshape(-1,4)
    if len(a)==0 or len(b)==0: return np.zeros((len(a),len(b)),dtype=float)
    xy1=np.maximum(a[:,None,:2],b[None,:,:2]); xy2=np.minimum(a[:,None,2:],b[None,:,2:])
    inter=np.prod(np.maximum(xy2-xy1,0),axis=2)
    area_a=np.prod(a[:,2:]-a[:,:2],axis=1)
    area_b=np.prod(b[:,2:]-b[:,:2],axis=1)
    return inter/(area_a[:,None]+area_b[None,:]-inter+1e-12)

def match_counts(pred,gt,iou_threshold):
    """Exact lexicographic optimization: maximize number of matches, then sum IoUs."""
    n,m=len(pred),len(gt)
    if n==0: return 0,0,m
    if m==0: return 0,n,0
    iou=iou_array(pred,gt)
    valid=iou>=iou_threshold-1e-12
    big=min(n,m)+1.0 # 1 valid match outranks maximal total IoU of all remaining matches
    w=np.zeros((n,m+n),dtype=float)
    w[:,:m]=valid*(big+iou)
    rows,cols=linear_sum_assignment(w,maximize=True)
    tp=sum(1 for r,c in zip(rows,cols) if c<m and valid[r,c])
    return tp,n-tp,m-tp

def category_map(categories, class_definitions_csv):
    """Validate COCO placeholder IDs against official MOPG-7 v2 metadata; no guessing.

    The official COCO file calls categories class_0 .. class_6. The separate
    class_definitions.csv supplies their semantic identities. Refuse to run
    if the metadata does not agree with the locked pre-registered mapping.
    """
    with open(class_definitions_csv, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not {'class_id','class_name'}.issubset(rows[0].keys() if rows else set()):
        raise ValueError('Official class_definitions.csv lacks class_id/class_name')
    source_names = {int(row['class_id']): ' '.join(row['class_name'].lower().replace('_',' ').replace('-',' ').split())
                    for row in rows}
    approved = {
        0: {'missing teeth','missing tooth'},
        1: {'dental crown','crown'},
        2: {'root canal','root canal treatment'},
        3: {'caries','dental caries'},
    }
    for cid, names in approved.items():
        if source_names.get(cid) not in names:
            raise ValueError(f'Official MOPG-7 class definitions conflict at id={cid}: {source_names.get(cid)!r}')
    coco_ids = {int(item['id']) for item in categories}
    if coco_ids != set(source_names):
        raise ValueError('COCO category IDs differ from official class definitions')
    for item in categories:
        cid = int(item['id'])
        name = ' '.join(str(item.get('name','')).lower().replace('_',' ').replace('-',' ').split())
        if name not in {f'class {cid}', f'class_{cid}',source_names[cid]}:
            raise ValueError(f'Unrecognized COCO category label for class {cid}: {item.get("name")}')
    return {0:0, 1:1, 2:2, 3:3}

def load_reference(path, class_definitions_csv):
    data=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    images=data.get('images'); annotations=data.get('annotations'); cats=data.get('categories')
    if not isinstance(images,list) or not isinstance(annotations,list) or not isinstance(cats,list):
        raise ValueError('Official COCO test JSON must include images, annotations and categories arrays')
    if len(images)!=210: raise ValueError(f'Expected 210 official test images, got {len(images)}. STOP and verify partition/version.')
    cmap=category_map(cats, class_definitions_csv)
    byid={int(im['id']):im for im in images}
    byname={}
    for im in images:
        key=basename(im['file_name'])
        if key in byname: raise ValueError('Ambiguous duplicate image basename '+key)
        byname[key]=im
    reference=defaultdict(lambda:defaultdict(list)); numbers=Counter()
    for item in annotations:
        if int(item['image_id']) not in byid: raise ValueError('Orphan annotation image_id')
        if int(item['category_id']) not in cmap: continue
        image=byid[int(item['image_id'])]
        W,H=float(image['width']),float(image['height'])
        x,y,w,h=map(float,item['bbox']);
        rect=validate_box([x,y,x+w,y+h],W,H,'gt:'+str(item.get('id')))
        cl=cmap[int(item['category_id'])]; numbers[cl]+=1
        reference[basename(image['file_name'])][cl].append(rect)
    actual=dict(numbers)
    if actual!=EXPECTED_REF:
        raise ValueError('Official reference count mismatch: expected '+repr(EXPECTED_REF)+'; got '+repr(actual)
                         +'. Verify MOPG-7 VERSION 2 official test split and class definitions.')
    if len(annotations)!=1088: print(f'NOTE: Original protocol says 1,088 all-class annotations; this COCO has {len(annotations)}.',file=sys.stderr)
    return byname,reference,cmap,len(annotations)

def load_prediction(path,model):
    payload=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    config=payload.get('configuration',{})
    if config.get('model')!=model: raise ValueError('Wrong model predictions file')
    if len(payload.get('images',[]))!=210: raise ValueError('Expected 210 prediction images')
    if abs(float(config.get('locked_evaluation_confidence',-1))-CONF[model])>1e-9: raise ValueError('Confidence threshold mismatch')
    pred={}
    for im in payload['images']:
        key=basename(im['image_file'])
        if key in pred: raise ValueError('Duplicate predicted image '+key)
        pred[key]=im
    return config,pred

def f1(tp,fp,fn): return 100*2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.

def write_csv(path,fieldnames,records):
    with open(path,'w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=fieldnames);wr.writeheader();wr.writerows(records)

def run(coco,pred_dir,outdir,class_definitions_csv,check_expected=True):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    gt_images,gt_boxes,cmap,all_n=load_reference(coco, class_definitions_csv)
    perclass=[];perimage=[];summary=[];mismatches=[];hashes={}
    for model,fn in PRED_FILES.items():
        path=Path(pred_dir)/fn;config,pred_images=load_prediction(path,model)
        if set(pred_images)!=set(gt_images):
            missing=sorted(set(gt_images)-set(pred_images))[:4]
            extra=sorted(set(pred_images)-set(gt_images))[:4]
            raise ValueError(f'Image filename alignment failed for {model}. missing={missing} extra={extra}')
        hashes[model]={'prediction_json_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                       'checkpoint_sha256_reported':config.get('checkpoint_sha256'),
                       'selected_confidence':CONF[model]}
        for threshold in THRESHOLDS:
            for name,src in sorted(gt_images.items(),key=lambda x:int(x[1]['id'])):
                W,H=int(src['width']),int(src['height']); im=pred_images[name]
                if (W,H)!=(int(im['width']),int(im['height'])):
                    raise ValueError(f'Image dimension mismatch: {model}, {name}: GT={(W,H)} vs pred={(im["width"],im["height"])}')
                pb=defaultdict(list)
                for item in im.get('detections',[]):
                    cid=int(item['model_class_id']);conf=float(item['confidence'])
                    if cid in MODEL_TO_EXT and conf>=CONF[model]:
                        rect=validate_box(item['xyxy'],W,H,f'{model}:{name}')
                        pb[MODEL_TO_EXT[cid]].append(rect)
                tot_tp=tot_fp=tot_fn=0
                for cls in EXT_NAMES:
                    tp,fp,fn=match_counts(pb[cls],gt_boxes[name][cls],threshold)
                    perclass.append(dict(model=model,iou_threshold=threshold,image_id=int(im['image_id']),external_class_id=cls,
                                         class_name=EXT_NAMES[cls],TP=tp,FP=fp,FN=fn))
                    tot_tp+=tp;tot_fp+=fp;tot_fn+=fn
                perimage.append(dict(model=model,iou_threshold=threshold,image_id=int(im['image_id']),TP=tot_tp,FP=tot_fp,FN=tot_fn))
            for cls in EXT_NAMES:
                vals=[r for r in perclass if r['model']==model and r['iou_threshold']==threshold and r['external_class_id']==cls]
                tp=sum(v['TP'] for v in vals); fp=sum(v['FP'] for v in vals);fn=sum(v['FN'] for v in vals)
                candidate=(tp,fp,fn);expected=EXPECTED_CLASS[(model,threshold,cls)]
                if candidate!=expected: mismatches.append({'model':model,'iou':threshold,'class':EXT_NAMES[cls],
                                                            'calculated':candidate,'previously_reported':expected})
            vals=[r for r in perimage if r['model']==model and r['iou_threshold']==threshold]
            tp=sum(v['TP'] for v in vals); fp=sum(v['FP'] for v in vals);fn=sum(v['FN'] for v in vals)
            summary.append({'model':model,'iou_threshold':threshold,'TP':tp,'FP':fp,'FN':fn,'micro_F1_percent':round(f1(tp,fp,fn),5)})
    write_csv(outdir/'external_per_image_class_recalculated.csv',['model','iou_threshold','image_id','external_class_id','class_name','TP','FP','FN'],perclass)
    write_csv(outdir/'external_per_image_recalculated.csv',['model','iou_threshold','image_id','TP','FP','FN'],perimage)
    write_csv(outdir/'external_micro_f1_recalculated.csv',['model','iou_threshold','TP','FP','FN','micro_F1_percent'],summary)
    info={'dataset':'MOPG-7 Version 2 official COCO test','reference_file_sha256':hashlib.sha256(Path(coco).read_bytes()).hexdigest(),
          'n_images':len(gt_images),'all_reference_boxes':all_n,'eligible_reference_boxes':sum(EXPECTED_REF.values()),
          'category_id_mapping_validated_against_official_class_definitions':cmap,'predictions':hashes,
          'comparison_against_archived_class_counts':'PASS' if not mismatches else 'FAIL',
          'mismatches':mismatches,'matching':'max-cardinality one-to-one followed by maximum summed IoU',
          'note':'Failure to reproduce may indicate a different dataset version, matching implementation or postprocessing; do not silently change thresholds.'}
    (outdir/'external_reproduction_audit.json').write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(info,indent=2,ensure_ascii=False));print('\n',*summary,sep='\n')
    if mismatches and check_expected: raise AssertionError(f'External reproduction mismatch: {len(mismatches)} class-level rows differ. See audit JSON.')
    return summary,info

def selftest():
    # Two candidate boxes for two references: cardinality must be preferred to higher single IoU.
    a=[[0,0,10,10],[21,0,30,10]];b=[[0,0,10,10],[20,0,30,10]]
    assert match_counts(a,b,.50)==(2,0,0)
    assert match_counts([],b,.50)==(0,0,2)
    assert match_counts(a,[],.50)==(0,2,0)
    assert match_counts([[0,0,10,10]],[[20,20,30,30]],.25)==(0,1,1)
    assert abs(f1(384,592,492)-41.468)<.01
    print('SELFTEST PASS')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--coco',type=Path,help='Path to official MOPG-7 Version 2 instances_test.json')
    ap.add_argument('--class-definitions',required=False,type=Path,help='Official metadata/class_definitions.csv from MOPG-7 Version 2')
    ap.add_argument('--predictions',type=Path,default=Path(__file__).resolve().parents[1]/'data')
    ap.add_argument('--out',type=Path,default=Path('external_reproduced_results'))
    ap.add_argument('--selftest',action='store_true')
    ap.add_argument('--allow-discrepancies',action='store_true')
    a=ap.parse_args()
    if a.selftest: selftest();return
    if a.coco is None: ap.error('--coco is required (official reference boxes)')
    if a.class_definitions is None: ap.error('--class-definitions is required (official MOPG-7 metadata)')
    run(a.coco,a.predictions,a.out,a.class_definitions,not a.allow_discrepancies)
if __name__=='__main__': main()
