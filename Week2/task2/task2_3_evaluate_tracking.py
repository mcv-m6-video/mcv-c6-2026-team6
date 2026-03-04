"""
task2/task2_3_evaluate_tracking.py  (upgraded)

Key upgrades:
  1. Uses TrackEval for proper HOTA/IDF1 — matches what other teams report.
     Falls back to the internal implementation only if TrackEval unavailable.
  2. Reads best configs from task2_1/task2_2 results to evaluate tuned trackers.
  3. Evaluates ALL track files: provided detectors + task1_1 + task1_2,
     both IoU-tracker and SORT variants.
  4. Comprehensive plots: IDF1/HOTA bars, DetA vs AssA scatter,
     full metrics heatmap, ID-switch comparison, HOTA vs alpha curve.

Install TrackEval (recommended):
    pip install trackeval --break-system-packages
"""

import os
import sys
import json
import shutil
import tempfile
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, defaultdict

DATA_ROOT   = os.environ.get('DATA_ROOT', 'data/AICity_data/train/S03/c010')
ANN_PATH    = os.path.join(DATA_ROOT, 'annotations.xml')
GT_FALLBACK = os.path.join(DATA_ROOT, 'AICity_data/train/S03/c010/gt', 'gt.txt')
RESULTS_DIR = 'results/task2_3'
PLOTS_DIR   = 'plots/try2'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# GT loaders
# ═══════════════════════════════════════════════════════════════════

def parse_cvat_xml(xml_path):
    tree=ET.parse(xml_path); root=tree.getroot(); gt={}
    for track in root.findall('track'):
        label=track.attrib.get('label','').lower()
        if label not in ('car','vehicle'): continue
        for box in track.findall('box'):
            if box.attrib.get('outside','0')=='1': continue
            fid=int(box.attrib['frame'])+1
            gt.setdefault(fid,[]).append([
                float(box.attrib['xtl']),float(box.attrib['ytl']),
                float(box.attrib['xbr']),float(box.attrib['ybr']),0])  # gt_id placeholder
    # Assign stable GT track IDs from track element
    gt2={}
    tid=0
    for track in root.findall('track'):
        label=track.attrib.get('label','').lower()
        if label not in ('car','vehicle'): continue
        tid+=1
        for box in track.findall('box'):
            if box.attrib.get('outside','0')=='1': continue
            fid=int(box.attrib['frame'])+1
            gt2.setdefault(fid,[]).append([
                float(box.attrib['xtl']),float(box.attrib['ytl']),
                float(box.attrib['xbr']),float(box.attrib['ybr']),tid])
    return gt2


def parse_annotations_mot(gt_path):
    """Parse MOT gt.txt with track IDs → {fid: [[x1,y1,x2,y2,tid], ...]}"""
    gt={}
    with open(gt_path) as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith('#'): continue
            parts=line.split(',')
            if len(parts)<6: continue
            fid=int(parts[0]); tid=int(parts[1])
            x,y,w,h=float(parts[2]),float(parts[3]),float(parts[4]),float(parts[5])
            conf=float(parts[6]) if len(parts)>6 else 1.0
            if conf==0: continue
            gt.setdefault(fid,[]).append([x,y,x+w,y+h,tid])
    return gt


def load_gt():
    if os.path.exists(ANN_PATH):
        print(f"  [GT] Parsing CVAT XML: {ANN_PATH}")
        return parse_cvat_xml(ANN_PATH)
    if os.path.exists(GT_FALLBACK):
        print(f"  [GT] Parsing MOT txt: {GT_FALLBACK}")
        return parse_annotations_mot(GT_FALLBACK)
    raise FileNotFoundError(f"No GT found at {ANN_PATH} or {GT_FALLBACK}")


def parse_track_file(path):
    """MOT track file → {fid: [[x1,y1,x2,y2,tid], ...]}"""
    tracks={}
    if not os.path.exists(path): return tracks
    with open(path) as f:
        for line in f:
            parts=line.strip().split(',')
            if len(parts)<6: continue
            fid=int(parts[0]); tid=int(parts[1])
            x1,y1=float(parts[2]),float(parts[3])
            w,h=float(parts[4]),float(parts[5])
            tracks.setdefault(fid,[]).append([x1,y1,x1+w,y1+h,tid])
    return tracks


def compute_iou_matrix(boxes_a, boxes_b):
    boxes_a=np.array(boxes_a,dtype=np.float32)
    boxes_b=np.array(boxes_b,dtype=np.float32)
    N,M=len(boxes_a),len(boxes_b)
    if N==0 or M==0: return np.zeros((N,M),dtype=np.float32)
    a=boxes_a[:,np.newaxis,:]; b=boxes_b[np.newaxis,:,:]
    xi1=np.maximum(a[...,0],b[...,0]); yi1=np.maximum(a[...,1],b[...,1])
    xi2=np.minimum(a[...,2],b[...,2]); yi2=np.minimum(a[...,3],b[...,3])
    inter=np.maximum(0.,xi2-xi1)*np.maximum(0.,yi2-yi1)
    area_a=(boxes_a[:,2]-boxes_a[:,0])*(boxes_a[:,3]-boxes_a[:,1])
    area_b=(boxes_b[:,2]-boxes_b[:,0])*(boxes_b[:,3]-boxes_b[:,1])
    union=area_a[:,np.newaxis]+area_b[np.newaxis,:]-inter+1e-6
    return inter/union


# ═══════════════════════════════════════════════════════════════════
# TrackEval-based evaluation (proper HOTA/IDF1)
# ═══════════════════════════════════════════════════════════════════

def _write_trackeval_gt(gt_dict, out_dir, seq_name='c010'):
    """Write GT in MOT format for TrackEval."""
    gt_seq_dir=os.path.join(out_dir,'gt','MOT17-train',seq_name,'gt')
    os.makedirs(gt_seq_dir,exist_ok=True)
    seqinfo=os.path.join(out_dir,'gt','MOT17-train',seq_name,'seqinfo.ini')
    all_fids=sorted(gt_dict.keys())
    n_frames=max(all_fids) if all_fids else 1
    with open(seqinfo,'w') as f:
        f.write(f"[Sequence]\nname={seq_name}\nimDir=img1\n"
                f"frameRate=10\nseqLength={n_frames}\n"
                f"imWidth=1920\nimHeight=1080\nimExt=.jpg\n")
    with open(os.path.join(gt_seq_dir,'gt.txt'),'w') as f:
        for fid in sorted(gt_dict.keys()):
            for box in gt_dict[fid]:
                x1,y1,x2,y2,tid=box[0],box[1],box[2],box[3],int(box[4])
                w,h=x2-x1,y2-y1
                f.write(f"{fid},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,1,1\n")
    # seqmap
    seqmap_dir=os.path.join(out_dir,'gt','seqmaps')
    os.makedirs(seqmap_dir,exist_ok=True)
    with open(os.path.join(seqmap_dir,'MOT17-train.txt'),'w') as f:
        f.write('name\n'+seq_name+'\n')
    return gt_seq_dir


def _write_trackeval_pred(tracks_dict, out_dir, tracker_name, seq_name='c010'):
    pred_dir=os.path.join(out_dir,'trackers','mot_challenge',
                          'MOT17-train',tracker_name,'data')
    os.makedirs(pred_dir,exist_ok=True)
    with open(os.path.join(pred_dir,f'{seq_name}.txt'),'w') as f:
        for fid in sorted(tracks_dict.keys()):
            for box in tracks_dict[fid]:
                x1,y1,x2,y2,tid=box[0],box[1],box[2],box[3],int(box[4])
                w,h=x2-x1,y2-y1
                f.write(f"{fid},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")


def evaluate_with_trackeval(gt_dict, tracks_dict, tracker_name):
    """
    Run TrackEval HOTA + IDF1 evaluation.
    Returns dict with HOTA, DetA, AssA, IDF1, IDP, IDR, MOTA, IDSW
    or None if TrackEval not available.
    """
    try:
        import trackeval
    except ImportError:
        return None

    tmpdir=tempfile.mkdtemp(prefix='trackeval_')
    try:
        seq_name='c010'
        _write_trackeval_gt(gt_dict, tmpdir, seq_name)
        _write_trackeval_pred(tracks_dict, tmpdir, tracker_name, seq_name)

        eval_config=trackeval.Evaluator.get_default_eval_config()
        eval_config['DISPLAY_LESS_PROGRESS']=True
        eval_config['PRINT_RESULTS']=False
        eval_config['PRINT_ONLY_COMBINED']=True
        eval_config['TIME_PROGRESS']=False

        dataset_config=trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        dataset_config['GT_FOLDER']     =os.path.join(tmpdir,'gt')
        dataset_config['TRACKERS_FOLDER']=os.path.join(tmpdir,'trackers')
        dataset_config['BENCHMARK']     ='MOT17'
        dataset_config['SPLIT_TO_EVAL'] ='train'
        dataset_config['TRACKERS_TO_EVAL']=[tracker_name]
        dataset_config['CLASSES_TO_EVAL']=['pedestrian']  # MOT17 uses 'pedestrian' class
        dataset_config['DO_PREPROC']    =False

        metrics_config={'METRICS':['HOTA','CLEAR','Identity'],'THRESHOLD':0.5}

        evaluator=trackeval.Evaluator(eval_config)
        dataset_list=[trackeval.datasets.MotChallenge2DBox(dataset_config)]
        metrics_list=[trackeval.metrics.HOTA(metrics_config),
                      trackeval.metrics.CLEAR(metrics_config),
                      trackeval.metrics.Identity(metrics_config)]

        res,_=evaluator.evaluate(dataset_list, metrics_list)

        # Extract results
        r=res['MotChallenge2DBox'][tracker_name]['c010']['pedestrian']
        hota_res=r.get('HOTA',{})
        clear_res=r.get('CLEAR',{})
        id_res=r.get('Identity',{})

        hota = float(np.mean(hota_res.get('HOTA', [0])))
        deta = float(np.mean(hota_res.get('DetA', [0])))
        assa = float(np.mean(hota_res.get('AssA', [0])))
        hota_alpha = {round(a,2): float(v) for a,v in
                      zip(hota_res.get('Alpha',[]), hota_res.get('HOTA',[]))}

        return {
            'HOTA':  round(hota, 4),
            'DetA':  round(deta, 4),
            'AssA':  round(assa, 4),
            'IDF1':  round(float(id_res.get('IDF1', 0)), 4),
            'IDP':   round(float(id_res.get('IDP',  0)), 4),
            'IDR':   round(float(id_res.get('IDR',  0)), 4),
            'MOTA':  round(float(clear_res.get('MOTA', 0)), 4),
            'IDSW':  int(clear_res.get('IDSW', 0)),
            'FP':    int(clear_res.get('FP', 0)),
            'FN':    int(clear_res.get('FN', 0)),
            'hota_alpha': hota_alpha,   # for HOTA vs alpha plot
            '_source': 'trackeval',
        }
    except Exception as e:
        print(f"    [TrackEval error] {e} — falling back to internal")
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════
# Internal fallback metrics (IDF1, MOTA, HOTA)
# ═══════════════════════════════════════════════════════════════════

def _greedy_match(gt_boxes, pred_boxes, gt_ids, pred_ids, iou_thr):
    if not gt_boxes or not pred_boxes:
        return []
    iou_mat=compute_iou_matrix(np.array([b[:4] for b in pred_boxes]),
                                np.array([b[:4] for b in gt_boxes]))
    matched_p,matched_g,pairs=set(),set(),[]
    for flat in np.argsort(-iou_mat,axis=None):
        pi,gi=np.unravel_index(flat,iou_mat.shape)
        if pi in matched_p or gi in matched_g: continue
        if iou_mat[pi,gi]<iou_thr: break
        pairs.append((pi,gi)); matched_p.add(pi); matched_g.add(gi)
    return pairs


def compute_idf1_internal(gt_dict, pred_dict, iou_thr=0.5):
    gt_to_pred=defaultdict(list)
    total_gt  =sum(len(v) for v in gt_dict.values())
    total_pred=sum(len(v) for v in pred_dict.values())
    for fid in sorted(set(gt_dict)|set(pred_dict)):
        gts=gt_dict.get(fid,[]); preds=pred_dict.get(fid,[])
        if not gts or not preds: continue
        pairs=_greedy_match(gts,preds,
                            [b[4] for b in gts],[b[4] for b in preds],iou_thr)
        for pi,gi in pairs:
            gt_to_pred[int(gts[gi][4])].append(int(preds[pi][4]))
    idtp=sum(Counter(v).most_common(1)[0][1] for v in gt_to_pred.values() if v)
    idfp=total_pred-idtp; idfn=total_gt-idtp
    idf1=(2*idtp)/(2*idtp+idfp+idfn+1e-9)
    idp=idtp/(idtp+idfp+1e-9); idr=idtp/(idtp+idfn+1e-9)
    return {'IDF1':round(idf1,4),'IDP':round(idp,4),'IDR':round(idr,4)}


def compute_mota_internal(gt_dict, pred_dict, iou_thr=0.5):
    total_fp=0; total_fn=0; total_idsw=0; total_gt=0; prev={}
    for fid in sorted(set(gt_dict)|set(pred_dict)):
        gts=gt_dict.get(fid,[]); preds=pred_dict.get(fid,[])
        total_gt+=len(gts)
        if not gts: total_fp+=len(preds); continue
        if not preds: total_fn+=len(gts); continue
        pairs=_greedy_match(gts,preds,
                            [b[4] for b in gts],[b[4] for b in preds],iou_thr)
        curr={int(preds[pi][4]):int(gts[gi][4]) for pi,gi in pairs}
        total_fp+=len(preds)-len(pairs); total_fn+=len(gts)-len(pairs)
        for pid,gid in curr.items():
            if pid in prev and prev[pid]!=gid: total_idsw+=1
        prev=curr
    mota=1-(total_fp+total_fn+total_idsw)/(total_gt+1e-9)
    return {'MOTA':round(mota,4),'FP':total_fp,'FN':total_fn,'IDSW':total_idsw}


def compute_hota_internal(gt_dict, pred_dict, alphas=None):
    """
    Full HOTA computed across multiple alpha thresholds (like TrackEval).
    HOTA = mean over alpha of sqrt(DetA(alpha) * AssA(alpha))
    """
    if alphas is None:
        alphas=np.arange(0.05,0.99,0.05)

    hota_per_alpha=[]; deta_per_alpha=[]; assa_per_alpha=[]

    for alpha in alphas:
        total_tp=0; total_fp=0; total_fn=0
        gt_traj=defaultdict(list); pred_traj=defaultdict(list)

        for fid in sorted(set(gt_dict)|set(pred_dict)):
            gts=gt_dict.get(fid,[]); preds=pred_dict.get(fid,[])
            if not gts and not preds: continue
            if not gts: total_fp+=len(preds); continue
            if not preds: total_fn+=len(gts); continue
            pairs=_greedy_match(gts,preds,
                                [b[4] for b in gts],[b[4] for b in preds],alpha)
            total_tp+=len(pairs)
            total_fp+=len(preds)-len(pairs)
            total_fn+=len(gts)-len(pairs)
            for pi,gi in pairs:
                gid=int(gts[gi][4]); pid=int(preds[pi][4])
                gt_traj[gid].append(pid); pred_traj[pid].append(gid)

        det_a=total_tp/(total_tp+total_fp+total_fn+1e-9)

        # AssA: for each GT track, find best matching pred track
        ass_scores=[]
        for gid,pids in gt_traj.items():
            best_pid,cnt=Counter(pids).most_common(1)[0]
            total_frames_this=len(pids)
            # frames this best pred appears total
            total_pred_frames=len(pred_traj.get(best_pid,[]))
            ass_scores.append(cnt/(total_frames_this+total_pred_frames-cnt+1e-9))

        ass_a=float(np.mean(ass_scores)) if ass_scores else 0.0
        h=float(np.sqrt(det_a*ass_a))
        hota_per_alpha.append(h); deta_per_alpha.append(det_a); assa_per_alpha.append(ass_a)

    hota_alpha={round(a,2):round(h,4) for a,h in zip(alphas,hota_per_alpha)}
    return {
        'HOTA': round(float(np.mean(hota_per_alpha)),4),
        'DetA': round(float(np.mean(deta_per_alpha)),4),
        'AssA': round(float(np.mean(assa_per_alpha)),4),
        'hota_alpha': hota_alpha,
        '_source': 'internal',
    }


def evaluate_tracker(gt_dict, pred_dict, tracker_name):
    """Full evaluation: try TrackEval first, fall back to internal."""
    print(f"  Evaluating {tracker_name} ...", end=' ', flush=True)

    result=evaluate_with_trackeval(gt_dict, pred_dict, tracker_name)
    if result is not None:
        print(f"[TrackEval]  IDF1={result['IDF1']:.4f}  HOTA={result['HOTA']:.4f}  MOTA={result['MOTA']:.4f}")
        return result

    # Internal fallback
    idf1=compute_idf1_internal(gt_dict, pred_dict)
    mota=compute_mota_internal(gt_dict, pred_dict)
    hota=compute_hota_internal(gt_dict, pred_dict)
    result={**idf1, **mota, **hota}
    print(f"[internal]  IDF1={result['IDF1']:.4f}  HOTA={result['HOTA']:.4f}  MOTA={result['MOTA']:.4f}")
    return result


# ═══════════════════════════════════════════════════════════════════
# Build track file registry
# ═══════════════════════════════════════════════════════════════════

def build_track_files():
    """Collect all track files from task2_1 and task2_2."""
    track_files={}

    # Provided detectors — IoU tracker (task2_1)
    for short,det in [('MaskRCNN','MaskRCNN_provided'),
                      ('SSD','SSD512_provided'),
                      ('YOLO','YOLOv3_provided')]:
        for suffix,label in [('_iou_tracker','_IoU'),
                              ('_iou_hungarian_tracker','_IoU_Hungarian'),
                              ('_sort_tracks','_SORT'),]:
            p=f'results/task2_1/{det}{suffix}.txt'
            if os.path.exists(p): track_files[f'{short}{label}']=p
        for suffix,label in [('_sort_tracks','_SORT'),
                              ('_kalman_tracker','_Kalman')]:
            p=f'results/task2_2/{det}{suffix}.txt'
            if os.path.exists(p): track_files[f'{short}{label}']=p

    # task1_1 best model
    t11='results/task1_1/best_model.json'
    if os.path.exists(t11):
        d=json.load(open(t11)); n=d.get('best_model_name','')
        if n:
            for suffix,label in [('_iou_tracker','_IoU'),
                                  ('_iou_hungarian_tracker','_IoU_Hungarian')]:
                p=f'results/task2_1/{n}{suffix}.txt'
                if os.path.exists(p): track_files[f'{n}{label}']=p
            for suffix,label in [('_sort_tracks','_SORT'),
                                  ('_kalman_tracker','_Kalman')]:
                p=f'results/task2_2/{n}{suffix}.txt'
                if os.path.exists(p): track_files[f'{n}{label}']=p

    # task1_2 fine-tuned
    t12='results/task1_2/best_config.json'
    if os.path.exists(t12):
        d=json.load(open(t12)); ft=d.get('model_name','')+('_finetuned')
        if ft:
            for suffix,label in [('_iou_tracker','_IoU'),
                                  ('_iou_hungarian_tracker','_IoU_Hungarian')]:
                p=f'results/task2_1/{ft}{suffix}.txt'
                if os.path.exists(p): track_files[f'{ft}{label}']=p
            for suffix,label in [('_sort_tracks','_SORT'),
                                  ('_kalman_tracker','_Kalman')]:
                p=f'results/task2_2/{ft}{suffix}.txt'
                if os.path.exists(p): track_files[f'{ft}{label}']=p

    return track_files


# ═══════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════

def save_plots(df, all_hota_alpha):
    # 1) IDF1 and HOTA bar charts side by side
    fig,axes=plt.subplots(1,2,figsize=(16,6))
    for ax,(metric,color) in zip(axes,[('IDF1','steelblue'),('HOTA','darkorange')]):
        sdf=df.sort_values(metric,ascending=False)
        bars=ax.barh(sdf['Tracker'],sdf[metric],color=color,edgecolor='black',linewidth=0.6)
        ax.bar_label(bars,fmt='%.4f',padding=3,fontsize=8)
        ax.set_xlabel(metric); ax.set_title(f'{metric} per Tracker',fontsize=12)
        ax.set_xlim(0,1.05); ax.grid(True,alpha=0.3,axis='x')
    plt.suptitle('Task 2.3 — Tracking Evaluation S03/C010',fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR,'task2_3_idf1_hota.png'),dpi=150); plt.close()

    # 2) DetA vs AssA scatter (key plot from report)
    fig,ax=plt.subplots(figsize=(8,7))
    colors=plt.cm.tab10(np.linspace(0,1,len(df)))
    for i,row in df.iterrows():
        ax.scatter(float(row['DetA']),float(row['AssA']),s=150,color=colors[i],zorder=3)
        ax.annotate(row['Tracker'],(float(row['DetA']),float(row['AssA'])),
                    textcoords='offset points',xytext=(5,5),fontsize=7)
    ax.set_xlabel('DetA (Detection Accuracy)',fontsize=12)
    ax.set_ylabel('AssA (Association Accuracy)',fontsize=12)
    ax.set_title('HOTA Decomposition: DetA vs AssA',fontsize=12)
    ax.plot([0,1],[0,1],'k--',alpha=0.3,label='DetA=AssA')
    ax.set_xlim(0,1.05); ax.set_ylim(0,1.05)
    ax.legend(fontsize=9); ax.grid(True,alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR,'task2_3_deta_assa.png'),dpi=150); plt.close()

    # 3) HOTA vs alpha curves (like Team 1 report)
    if all_hota_alpha:
        fig,ax=plt.subplots(figsize=(9,6))
        colors=plt.cm.tab10(np.linspace(0,1,len(all_hota_alpha)))
        for (name,alpha_dict),col in zip(all_hota_alpha.items(),colors):
            alphas=sorted(alpha_dict.keys())
            vals=[alpha_dict[a] for a in alphas]
            mean_hota=np.mean(list(alpha_dict.values()))
            ax.plot(alphas,vals,'-o',ms=4,color=col,
                    label=f'{name} ({mean_hota:.3f})',linewidth=1.5)
        ax.set_xlabel('Alpha (IoU threshold for matching)',fontsize=11)
        ax.set_ylabel('HOTA',fontsize=11)
        ax.set_title('HOTA vs Alpha — all trackers',fontsize=12)
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.legend(fontsize=7,bbox_to_anchor=(1.01,1),loc='upper left')
        ax.grid(True,alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR,'task2_3_hota_alpha.png'),
                    dpi=150,bbox_inches='tight'); plt.close()

    # 4) Full metrics heatmap
    heatmap_cols=['IDF1','IDP','IDR','HOTA','DetA','AssA','MOTA']
    cols_available=[c for c in heatmap_cols if c in df.columns]
    if cols_available:
        hm=df[cols_available].astype(float).values
        fig,ax=plt.subplots(figsize=(max(10,len(cols_available)*1.5),max(4,len(df)*0.6)))
        im=ax.imshow(hm,cmap='YlOrRd',aspect='auto',vmin=0,vmax=1)
        plt.colorbar(im,ax=ax)
        ax.set_xticks(range(len(cols_available))); ax.set_xticklabels(cols_available)
        ax.set_yticks(range(len(df))); ax.set_yticklabels(df['Tracker'].tolist(),fontsize=8)
        for i in range(len(df)):
            for j in range(len(cols_available)):
                v=hm[i,j]
                ax.text(j,i,f'{v:.3f}',ha='center',va='center',
                        fontsize=7,color='black' if v<0.7 else 'white')
        ax.set_title('Full MOT Metrics Heatmap',fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR,'task2_3_heatmap.png'),dpi=150); plt.close()

    # 5) ID Switches comparison
    if 'IDSW' in df.columns:
        fig,ax=plt.subplots(figsize=(max(8,len(df)*1.2),5))
        sdf=df.sort_values('IDSW')
        med=sdf['IDSW'].median()
        colors_idsw=['seagreen' if v<=med else 'salmon' for v in sdf['IDSW']]
        bars=ax.bar(sdf['Tracker'],sdf['IDSW'],color=colors_idsw,edgecolor='black')
        ax.bar_label(bars,padding=3,fontsize=9)
        ax.set_ylabel('# ID Switches'); ax.set_title('ID Switches per Tracker (lower=better)')
        plt.xticks(rotation=30,ha='right'); ax.grid(True,alpha=0.3,axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR,'task2_3_idsw.png'),dpi=150); plt.close()

    # 6) IoU tracker vs SORT grouped comparison
    iou_trackers =df[df['Tracker'].str.contains('IoU')  & ~df['Tracker'].str.contains('Hungarian')]
    sort_trackers=df[df['Tracker'].str.contains('SORT|Kalman',regex=True)]
    if not iou_trackers.empty and not sort_trackers.empty:
        # Match by detector prefix
        detectors=[]
        iou_idf1=[]; sort_idf1=[]
        for _,irow in iou_trackers.iterrows():
            prefix=irow['Tracker'].replace('_IoU','')
            smatch=sort_trackers[sort_trackers['Tracker'].str.startswith(prefix)]
            if not smatch.empty:
                detectors.append(prefix)
                iou_idf1.append(float(irow['IDF1']))
                sort_idf1.append(float(smatch.iloc[0]['IDF1']))
        if detectors:
            x=np.arange(len(detectors)); width=0.35
            fig,ax=plt.subplots(figsize=(max(8,len(detectors)*2),5))
            b1=ax.bar(x-width/2,iou_idf1,width,label='IoU Tracker',
                      color='steelblue',edgecolor='black')
            b2=ax.bar(x+width/2,sort_idf1,width,label='SORT (Kalman)',
                      color='darkorange',edgecolor='black')
            ax.bar_label(b1,fmt='%.3f',padding=3,fontsize=8)
            ax.bar_label(b2,fmt='%.3f',padding=3,fontsize=8)
            ax.set_xticks(x); ax.set_xticklabels(detectors,rotation=20,ha='right')
            ax.set_ylabel('IDF1'); ax.set_ylim(0,1.1)
            ax.set_title('IoU Tracker vs SORT (Kalman) — IDF1 Comparison')
            ax.legend(); ax.grid(True,alpha=0.3,axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR,'task2_3_iou_vs_sort.png'),dpi=150)
            plt.close()

    print(f"Plots saved to {PLOTS_DIR}/")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("="*60)
    print("TASK 2.3: IDF1 / HOTA Evaluation")
    print("="*60)

    # Check TrackEval
    try:
        import trackeval
        print("  [✓] TrackEval available — using official HOTA/IDF1")
    except ImportError:
        print("  [!] TrackEval not found — using internal implementation")
        print("      Install with: pip install trackeval --break-system-packages")

    gt_dict=load_gt()
    print(f"  GT: {len(gt_dict)} frames, "
          f"{len({int(b[4]) for v in gt_dict.values() for b in v})} unique GT IDs")

    track_files=build_track_files()
    if not track_files:
        print("[ERROR] No track files found. Run task2_1 and task2_2 first.")
        return
    print(f"\n  Found {len(track_files)} track files to evaluate:")
    for name,path in track_files.items():
        print(f"    {name}: {path}")

    rows=[]; all_hota_alpha={}

    for tracker_name,track_path in track_files.items():
        pred_dict=parse_track_file(track_path)
        if not pred_dict:
            print(f"  [skip] {tracker_name}: empty or unreadable")
            continue

        print(f"\n--- {tracker_name} ---")
        result=evaluate_tracker(gt_dict, pred_dict,
                                tracker_name.replace('/','_').replace(' ','_'))

        n_ids=len({int(b[4]) for v in pred_dict.values() for b in v})
        row={'Tracker':tracker_name, 'N_IDs':n_ids, **{
            k:v for k,v in result.items()
            if k not in ('hota_alpha','_source')}}
        rows.append(row)

        if 'hota_alpha' in result and result['hota_alpha']:
            all_hota_alpha[tracker_name]=result['hota_alpha']

        # Print key metrics
        print(f"  IDF1={result.get('IDF1',0):.4f}  "
              f"HOTA={result.get('HOTA',0):.4f}  "
              f"DetA={result.get('DetA',0):.4f}  "
              f"AssA={result.get('AssA',0):.4f}  "
              f"MOTA={result.get('MOTA',0):.4f}  "
              f"IDSW={result.get('IDSW',0)}")

    if not rows:
        print("No results produced."); return

    df=pd.DataFrame(rows)
    csv_path=os.path.join(RESULTS_DIR,'full_metrics_table.csv')
    df.to_csv(csv_path,index=False)
    print(f"\nSaved → {csv_path}")

    print_cols=['Tracker','IDF1','HOTA','DetA','AssA','MOTA','IDSW']
    print_cols=[c for c in print_cols if c in df.columns]
    print("\n"+df[print_cols].to_string(index=False))

    # Save HOTA vs alpha data
    if all_hota_alpha:
        with open(os.path.join(RESULTS_DIR,'hota_alpha.json'),'w') as f:
            json.dump(all_hota_alpha,f,indent=2)

    save_plots(df, all_hota_alpha)
    print("\n✓ Task 2.3 complete.")


if __name__ == '__main__':
    main()