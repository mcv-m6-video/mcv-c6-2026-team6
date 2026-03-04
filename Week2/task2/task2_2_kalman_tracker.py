"""
task2/task2_2_kalman_tracker.py  (upgraded)

Upgrades over previous version:
  1. Grid search over iou_threshold × max_age × min_confidence × matcher
     → saves best config per detector
  2. Greedy matching alternative alongside existing Hungarian (SORT used Hungarian only before)
  3. Confidence threshold filtering with ablation
  4. Runs ALL detectors (provided + task1_1 + task1_2) so task2_3 has
     SORT track files for every detector
  5. Output filenames aligned with what task2_3 expects:
       {det_name}_sort_tracks.txt      ← best config
       {det_name}_kalman_tracker.txt   ← alias (same file, for compatibility)
  6. Ablation plots: IoU, max_age, min_hits, min_conf, greedy vs Hungarian
"""

import os
import sys
import cv2
import json
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import product as iterproduct
from pathlib import Path
from scipy.optimize import linear_sum_assignment

_DATA_BASE  = os.environ.get('DATA_ROOT', 'data')
if os.path.basename(_DATA_BASE.rstrip('/')) == 'c010':
    C010_ROOT = _DATA_BASE
else:
    C010_ROOT = os.path.join(_DATA_BASE, 'AICity_data', 'train', 'S03', 'c010')
DATA_ROOT   = C010_ROOT
ANN_PATH    = os.path.join(C010_ROOT, 'annotations.xml')
GT_FALLBACK = os.path.join(C010_ROOT, 'gt', 'gt.txt')
DET_DIR     = os.path.join(C010_ROOT, 'det')
VIDEO_PATH  = os.environ.get('VIDEO_PATH', os.path.join(C010_ROOT, 'vdo.avi'))
RESULTS_DIR = 'results/task2_2'
PLOTS_DIR   = 'plots/try2'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)
os.makedirs('qualitative', exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# Inlined utilities  (same as task2_1)
# ═══════════════════════════════════════════════════════════════════

def parse_cvat_xml(xml_path):
    tree = ET.parse(xml_path); root = tree.getroot()
    gt = {}
    for track in root.findall('track'):
        label = track.attrib.get('label','').lower()
        if label not in ('car','vehicle'):
            continue
        for box in track.findall('box'):
            if box.attrib.get('outside','0') == '1':
                continue
            fid = int(box.attrib['frame']) + 1
            gt.setdefault(fid,[]).append([
                float(box.attrib['xtl']), float(box.attrib['ytl']),
                float(box.attrib['xbr']), float(box.attrib['ybr']),
            ])
    return gt


def parse_annotations_mot(gt_path):
    """MOT format: frame, track_id, x, y, w, h, conf, class, visibility"""
    gt = {}
    with open(gt_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split(',')
            if len(parts) < 6: continue
            fid = int(parts[0])
            tid = int(parts[1])          # track_id — was missing!
            x,y,w,h = float(parts[2]),float(parts[3]),float(parts[4]),float(parts[5])
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            if conf == 0: continue
            gt.setdefault(fid,[]).append([x,y,x+w,y+h,tid])
    return gt


def load_gt():
    if os.path.exists(ANN_PATH):
        print(f"  [GT] Parsing CVAT XML: {ANN_PATH}")
        return parse_cvat_xml(ANN_PATH)
    if os.path.exists(GT_FALLBACK):
        print(f"  [GT] Parsing MOT txt: {GT_FALLBACK}")
        return parse_annotations_mot(GT_FALLBACK)
    print("  [GT] WARNING: no GT found"); return {}


def parse_detections_mot(det_path):
    dets = {}
    with open(det_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(',')
            if len(parts) < 6: continue
            fid = int(parts[0])
            x,y,w,h = float(parts[2]),float(parts[3]),float(parts[4]),float(parts[5])
            score = float(parts[6]) if len(parts) > 6 else 1.0
            dets.setdefault(fid,[]).append([x,y,x+w,y+h,score])
    return dets


def save_tracks_mot(tracks_dict, out_path):
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    with open(out_path,'w') as f:
        for fid in sorted(tracks_dict.keys()):
            for t in tracks_dict[fid]:
                x1,y1,x2,y2,tid = t[0],t[1],t[2],t[3],int(t[4])
                f.write(f"{fid},{tid},{x1:.2f},{y1:.2f},"
                        f"{x2-x1:.2f},{y2-y1:.2f},1,-1,-1,-1\n")


def compute_iou_matrix(boxes_a, boxes_b):
    boxes_a = np.array(boxes_a, dtype=np.float32)
    boxes_b = np.array(boxes_b, dtype=np.float32)
    N,M = len(boxes_a),len(boxes_b)
    if N==0 or M==0: return np.zeros((N,M),dtype=np.float32)
    a = boxes_a[:,np.newaxis,:]
    b = boxes_b[np.newaxis,:,:]
    xi1=np.maximum(a[...,0],b[...,0]); yi1=np.maximum(a[...,1],b[...,1])
    xi2=np.minimum(a[...,2],b[...,2]); yi2=np.minimum(a[...,3],b[...,3])
    inter = np.maximum(0.,xi2-xi1)*np.maximum(0.,yi2-yi1)
    area_a=(boxes_a[:,2]-boxes_a[:,0])*(boxes_a[:,3]-boxes_a[:,1])
    area_b=(boxes_b[:,2]-boxes_b[:,0])*(boxes_b[:,3]-boxes_b[:,1])
    union=area_a[:,np.newaxis]+area_b[np.newaxis,:]-inter+1e-6
    return inter/union


class VideoFrameLoader:
    def __init__(self, video_path=VIDEO_PATH):
        self.video_path = video_path; self._cap = None
    def _open(self):
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.video_path)
    def get_frame(self, frame_idx_0based):
        self._open()
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_0based)
        ret, frame = self._cap.read()
        return frame if ret else None
    def __del__(self):
        if self._cap is not None: self._cap.release()


_COLORS=[(255,85,0),(0,170,255),(170,0,255),(0,255,85),(255,0,170),
         (85,255,0),(0,85,255),(255,170,0),(0,255,170),(170,255,0)]
def _track_color(tid): return _COLORS[int(tid)%len(_COLORS)]

def draw_tracks(frame, tracks, trails=None, gt_boxes=None, frame_id=None):
    img=frame.copy()
    if gt_boxes:
        for box in gt_boxes:
            cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,200,0),1)
    if trails:
        for tid,pts in trails.items():
            col=_track_color(tid)
            for k in range(1,len(pts)): cv2.line(img,pts[k-1],pts[k],col,1)
    for trk in tracks:
        x1,y1,x2,y2=int(trk[0]),int(trk[1]),int(trk[2]),int(trk[3]); tid=int(trk[4])
        col=_track_color(tid); cv2.rectangle(img,(x1,y1),(x2,y2),col,2)
        cv2.putText(img,f"ID:{tid}",(x1,max(y1-4,10)),cv2.FONT_HERSHEY_SIMPLEX,0.45,col,1)
    if frame_id is not None:
        cv2.putText(img,f"f{frame_id}",(6,18),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    return img

def save_qualitative_grid(panels, out_path, nrows=2, ncols=3):
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    fig,axes=plt.subplots(nrows,ncols,figsize=(ncols*5,nrows*3.2))
    axes=np.array(axes).flatten()
    for i,ax in enumerate(axes):
        if i<len(panels):
            img,title=panels[i]; ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)); ax.set_title(title,fontsize=9)
        ax.axis('off')
    plt.tight_layout(); plt.savefig(out_path,dpi=130,bbox_inches='tight'); plt.close()


# ═══════════════════════════════════════════════════════════════════
# Kalman Filter (SORT-style)
# ═══════════════════════════════════════════════════════════════════

class KalmanBoxTracker:
    _id_counter = 0

    def __init__(self, box):
        KalmanBoxTracker._id_counter += 1
        self.id = KalmanBoxTracker._id_counter
        self.age = 0; self.hits = 1; self.time_since_update = 0
        self.kf = self._build_kf()
        self.kf['x'][:4] = self._box_to_z(box)

    @staticmethod
    def _build_kf():
        F=np.eye(7,dtype=np.float32); F[0,4]=F[1,5]=F[2,6]=1.0
        H=np.zeros((4,7),dtype=np.float32); H[0,0]=H[1,1]=H[2,2]=H[3,3]=1.0
        Q=np.eye(7,dtype=np.float32); Q[4:,4:]*=0.01
        R=np.eye(4,dtype=np.float32); R[2:,2:]*=10.0
        P=np.eye(7,dtype=np.float32); P[4:,4:]*=1000.0
        return {'F':F,'H':H,'Q':Q,'R':R,'P':P,'x':np.zeros((7,1),dtype=np.float32)}

    @staticmethod
    def _box_to_z(box):
        w=box[2]-box[0]; h=box[3]-box[1]
        cx=box[0]+w/2; cy=box[1]+h/2
        s=w*h; r=w/(h+1e-6)
        return np.array([[cx],[cy],[s],[r]],dtype=np.float32)

    @staticmethod
    def _z_to_box(z):
        cx,cy,s,r=float(z[0,0]),float(z[1,0]),float(z[2,0]),float(z[3,0])
        s=max(s,1.0); w=np.sqrt(s*abs(r)); h=s/(w+1e-6)
        return [cx-w/2,cy-h/2,cx+w/2,cy+h/2]

    def predict(self):
        kf=self.kf; kf['x']=kf['F']@kf['x']; kf['P']=kf['F']@kf['P']@kf['F'].T+kf['Q']
        self.time_since_update+=1; self.age+=1
        return self._z_to_box(kf['x'][:4])

    def update(self, box):
        kf=self.kf; z=self._box_to_z(box)
        S=kf['H']@kf['P']@kf['H'].T+kf['R']; K=kf['P']@kf['H'].T@np.linalg.inv(S)
        kf['x']=kf['x']+K@(z-kf['H']@kf['x'])
        kf['P']=(np.eye(7)-K@kf['H'])@kf['P']
        self.time_since_update=0; self.hits+=1

    def get_state(self): return self._z_to_box(self.kf['x'][:4])
    def to_mot(self):
        box=self.get_state(); return [box[0],box[1],box[2],box[3],self.id]


# ═══════════════════════════════════════════════════════════════════
# SORT tracker  (now supports both greedy and Hungarian)
# ═══════════════════════════════════════════════════════════════════

class SORTTracker:
    def __init__(self, iou_threshold=0.3, max_age=5, min_hits=3,
                 use_hungarian=True):
        self.iou_threshold = iou_threshold
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.use_hungarian = use_hungarian
        self.trackers      = []
        self.frame_count   = 0
        KalmanBoxTracker._id_counter = 0

    def reset(self):
        self.trackers=[]; self.frame_count=0; KalmanBoxTracker._id_counter=0

    def update(self, detections):
        self.frame_count += 1
        trk_boxes=[]; to_del=[]
        for i,t in enumerate(self.trackers):
            box=t.predict()
            if np.any(np.isnan(box)): to_del.append(i)
            else: trk_boxes.append(box)
        for i in reversed(to_del): self.trackers.pop(i)

        trk_arr=np.array(trk_boxes) if trk_boxes else np.empty((0,4))
        det_arr=np.array([d[:4] for d in detections]) if detections else np.empty((0,4))

        if self.use_hungarian:
            matched,unmatched_dets,unmatched_trks=self._hungarian(det_arr,trk_arr)
        else:
            matched,unmatched_dets,unmatched_trks=self._greedy(det_arr,trk_arr)

        for di,ti in matched: self.trackers[ti].update(detections[di][:4])
        for di in unmatched_dets: self.trackers.append(KalmanBoxTracker(detections[di][:4]))
        self.trackers=[t for t in self.trackers if t.time_since_update<=self.max_age]

        return [t.to_mot() for t in self.trackers
                if t.time_since_update==0 and
                   (t.hits>=self.min_hits or self.frame_count<=self.min_hits)]

    def _greedy(self, det_arr, trk_arr):
        if len(trk_arr)==0: return [], list(range(len(det_arr))), []
        if len(det_arr)==0: return [], [], list(range(len(trk_arr)))
        iou_mat=compute_iou_matrix(det_arr,trk_arr)
        matched_d,matched_t,pairs=set(),set(),[]
        for flat in np.argsort(-iou_mat,axis=None):
            di,ti=np.unravel_index(flat,iou_mat.shape)
            if di in matched_d or ti in matched_t: continue
            if iou_mat[di,ti] < self.iou_threshold: break
            pairs.append((int(di),int(ti))); matched_d.add(di); matched_t.add(ti)
        return pairs,[i for i in range(len(det_arr)) if i not in matched_d], \
                     [i for i in range(len(trk_arr)) if i not in matched_t]

    def _hungarian(self, det_arr, trk_arr):
        if len(trk_arr)==0: return [], list(range(len(det_arr))), []
        if len(det_arr)==0: return [], [], list(range(len(trk_arr)))
        iou_mat=compute_iou_matrix(det_arr,trk_arr)
        row_ind,col_ind=linear_sum_assignment(-iou_mat)
        matched_d,matched_t,pairs=set(),set(),[]
        for di,ti in zip(row_ind,col_ind):
            if iou_mat[di,ti]>=self.iou_threshold:
                pairs.append((int(di),int(ti))); matched_d.add(di); matched_t.add(ti)
        return pairs,[i for i in range(len(det_arr)) if i not in matched_d], \
                     [i for i in range(len(trk_arr)) if i not in matched_t]


def _nms(dets, nms_iou=0.5):
    if len(dets) <= 1: return dets
    dets_s = sorted(dets, key=lambda d: d[4] if len(d)>4 else 1.0, reverse=True)
    keep = []
    for d in dets_s:
        suppressed = False
        for k in keep:
            xi1=max(d[0],k[0]); yi1=max(d[1],k[1])
            xi2=min(d[2],k[2]); yi2=min(d[3],k[3])
            inter=max(0,xi2-xi1)*max(0,yi2-yi1)
            if inter==0: continue
            iou=inter/((d[2]-d[0])*(d[3]-d[1])+(k[2]-k[0])*(k[3]-k[1])-inter+1e-6)
            if iou>=nms_iou: suppressed=True; break
        if not suppressed: keep.append(d)
    return keep


def run_tracker(det_dict, tracker, min_confidence=0.0, nms_iou=0.5, min_area=0):
    tracker.reset(); tracks_dict={}
    for fid in sorted(det_dict.keys()):
        dets=[d for d in det_dict[fid] if len(d)<5 or d[4]>=min_confidence]
        if min_area > 0:
            dets = [d for d in dets if (d[2]-d[0])*(d[3]-d[1]) >= min_area]
        dets = _nms(dets, nms_iou=nms_iou)
        tracks_dict[fid]=tracker.update(dets)
    return tracks_dict


# ═══════════════════════════════════════════════════════════════════
# Quick IDF1 proxy (same as task2_1 — for grid search)
# ═══════════════════════════════════════════════════════════════════

def align_det_to_gt(det_dict, gt_dict):
    """Align detection frame IDs to GT frame IDs (positional remap if no overlap)."""
    gt_fids  = sorted(gt_dict.keys())
    det_fids = sorted(det_dict.keys())
    if not gt_fids or not det_fids:
        return det_dict
    overlap = len(set(gt_fids) & set(det_fids))
    if overlap > 0.1 * min(len(gt_fids), len(det_fids)):
        return det_dict
    n = min(len(det_fids), len(gt_fids))
    remap = {det_fids[i]: gt_fids[i] for i in range(n)}
    return {remap.get(fid, fid): boxes for fid, boxes in det_dict.items()}


def quick_idf1(gt_dict, tracks_dict, iou_thr=0.5):
    from collections import Counter, defaultdict
    gt_to_pred=defaultdict(list)
    total_gt  =sum(len(v) for v in gt_dict.values())
    total_pred=sum(len(tracks_dict.get(fid,[])) for fid in gt_dict)
    for fid in sorted(set(gt_dict)|set(tracks_dict)):
        gts=gt_dict.get(fid,[]); preds=tracks_dict.get(fid,[])
        if not gts or not preds: continue
        gt_boxes  =np.array([b[:4] for b in gts])
        pred_boxes=np.array([b[:4] for b in preds])
        gt_ids    =[int(b[4]) if len(b)>4 else 0 for b in gts]
        pred_ids  =[int(b[4]) for b in preds]
        iou_mat=compute_iou_matrix(pred_boxes,gt_boxes)
        matched_p,matched_g=set(),set()
        for flat in np.argsort(-iou_mat,axis=None):
            pi,gi=np.unravel_index(flat,iou_mat.shape)
            if pi in matched_p or gi in matched_g: continue
            if iou_mat[pi,gi]<iou_thr: break
            matched_p.add(pi); matched_g.add(gi)
            gt_to_pred[gt_ids[gi]].append(pred_ids[pi])
    idtp=sum(Counter(v).most_common(1)[0][1] for v in gt_to_pred.values() if v)
    idfp=total_pred-idtp; idfn=total_gt-idtp
    return (2*idtp)/(2*idtp+idfp+idfn+1e-9)


# ═══════════════════════════════════════════════════════════════════
# Grid search
# ═══════════════════════════════════════════════════════════════════

def grid_search(gt_dict, det_dict, det_name):
    iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    max_ages       = [1, 3, 5, 10, 15]
    min_confs      = [0.0, 0.25, 0.4, 0.5, 0.6, 0.7]
    min_hits_list  = [1, 2, 3]
    min_areas      = [0, 5000, 15000, 30000, 50000]
    matchers       = ['greedy', 'hungarian']

    rows=[]; best_idf1=-1; best_cfg={}
    total=len(iou_thresholds)*len(max_ages)*len(min_confs)*len(min_hits_list)*len(min_areas)*len(matchers)
    print(f"  Grid search: {total} configs ...")

    for iou_thr,max_age,min_conf,min_hits,min_area,matcher in iterproduct(
            iou_thresholds,max_ages,min_confs,min_hits_list,min_areas,matchers):
        tracker=SORTTracker(iou_threshold=iou_thr, max_age=max_age,
                            min_hits=min_hits, use_hungarian=(matcher=='hungarian'))
        tracks=run_tracker(det_dict, tracker, min_confidence=min_conf, nms_iou=0.5, min_area=min_area)
        idf1=quick_idf1(gt_dict, tracks)
        n_ids=len({int(t[4]) for fid in tracks for t in tracks[fid]})
        rows.append({'iou_threshold':iou_thr,'max_age':max_age,'min_conf':min_conf,
                     'min_hits':min_hits,'matcher':matcher,
                     'IDF1':round(idf1,4),'n_ids':n_ids})
        if idf1>best_idf1:
            best_idf1=idf1
            best_cfg={'iou_threshold':iou_thr,'max_age':max_age,'min_conf':min_conf,
                      'min_hits':min_hits,'matcher':matcher,'IDF1':round(idf1,4)}

    df=pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR,f'{det_name}_grid_search.csv'),index=False)
    print(f"  Best: matcher={best_cfg['matcher']}  "
          f"iou={best_cfg['iou_threshold']}  age={best_cfg['max_age']}  "
          f"min_hits={best_cfg['min_hits']}  conf={best_cfg['min_conf']}  "
          f"IDF1={best_cfg['IDF1']:.4f}")
    return best_cfg, df


# ═══════════════════════════════════════════════════════════════════
# Ablations
# ═══════════════════════════════════════════════════════════════════

def ablation_iou(gt_dict, det_dict, max_age=5, min_hits=3, min_conf=0.0):
    rows=[]
    for thr in np.arange(0.1,0.7,0.1):
        tracker=SORTTracker(iou_threshold=float(thr),max_age=max_age,min_hits=min_hits)
        tracks=run_tracker(det_dict,tracker,min_confidence=min_conf)
        n_ids=len({int(t[4]) for fid in tracks for t in tracks[fid]})
        idf1=quick_idf1(gt_dict,tracks)
        rows.append({'iou_threshold':round(float(thr),2),'n_ids':n_ids,'IDF1':round(idf1,4)})
    return pd.DataFrame(rows)

def ablation_age(gt_dict, det_dict, iou_thr=0.3, min_hits=3, min_conf=0.0):
    rows=[]
    for age in [1,2,3,5,7,10,15,20]:
        tracker=SORTTracker(iou_threshold=iou_thr,max_age=age,min_hits=min_hits)
        tracks=run_tracker(det_dict,tracker,min_confidence=min_conf)
        n_ids=len({int(t[4]) for fid in tracks for t in tracks[fid]})
        idf1=quick_idf1(gt_dict,tracks)
        rows.append({'max_age':age,'n_ids':n_ids,'IDF1':round(idf1,4)})
    return pd.DataFrame(rows)

def ablation_conf(gt_dict, det_dict, iou_thr=0.3, max_age=5, min_hits=3):
    rows=[]
    for conf in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7]:
        tracker=SORTTracker(iou_threshold=iou_thr,max_age=max_age,min_hits=min_hits)
        tracks=run_tracker(det_dict,tracker,min_confidence=conf)
        n_ids=len({int(t[4]) for fid in tracks for t in tracks[fid]})
        idf1=quick_idf1(gt_dict,tracks)
        rows.append({'min_conf':conf,'n_ids':n_ids,'IDF1':round(idf1,4)})
    return pd.DataFrame(rows)

def ablation_min_hits(gt_dict, det_dict, iou_thr=0.3, max_age=5, min_conf=0.0):
    rows=[]
    for mh in [1,2,3,5,7]:
        tracker=SORTTracker(iou_threshold=iou_thr,max_age=max_age,min_hits=mh)
        tracks=run_tracker(det_dict,tracker,min_confidence=min_conf)
        n_ids=len({int(t[4]) for fid in tracks for t in tracks[fid]})
        idf1=quick_idf1(gt_dict,tracks)
        rows.append({'min_hits':mh,'n_ids':n_ids,'IDF1':round(idf1,4)})
    return pd.DataFrame(rows)

def compare_matchers(gt_dict, det_dict, iou_thr=0.3, max_age=5, min_hits=3, min_conf=0.0):
    rows=[]
    for matcher in ['greedy','hungarian']:
        tracker=SORTTracker(iou_threshold=iou_thr,max_age=max_age,min_hits=min_hits,
                            use_hungarian=(matcher=='hungarian'))
        tracks=run_tracker(det_dict,tracker,min_confidence=min_conf)
        n_ids=len({int(t[4]) for fid in tracks for t in tracks[fid]})
        idf1=quick_idf1(gt_dict,tracks)
        rows.append({'matcher':matcher,'n_ids':n_ids,'IDF1':round(idf1,4)})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# Qualitative
# ═══════════════════════════════════════════════════════════════════

def build_trails(tracks_dict,trail_length=20):
    trails={}
    for fid in sorted(tracks_dict.keys()):
        for trk in tracks_dict[fid]:
            tid=int(trk[4]); cx=int((trk[0]+trk[2])/2); cy=int((trk[1]+trk[3])/2)
            trails.setdefault(tid,[]).append((fid,(cx,cy)))
    return trails

def get_trail_at_frame(trails_full,fid,trail_length=20):
    active={}
    for tid,pts in trails_full.items():
        pts_before=[(f,p) for f,p in pts if f<=fid]
        if pts_before: active[tid]=[p for _,p in pts_before[-trail_length:]]
    return active

def save_qualitative(loader, tracks_dict, gt_dict, det_name, suffix=''):
    all_fids=sorted(tracks_dict.keys())
    if not all_fids: return
    sample=all_fids[::max(1,len(all_fids)//6)][:6]
    trails_full=build_trails(tracks_dict); panels=[]
    for fid in sample:
        frame=loader.get_frame(fid-1)
        if frame is None: continue
        trail_at=get_trail_at_frame(trails_full,fid)
        ann=draw_tracks(frame,tracks_dict.get(fid,[]),trails=trail_at,
                        gt_boxes=gt_dict.get(fid,[]),frame_id=fid)
        panels.append((ann,f"Frame {fid}"))
    save_qualitative_grid(panels,f'qualitative/task2_2_{det_name}{suffix}.png',nrows=2,ncols=3)


# ═══════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════

def save_ablation_plots(det_name, iou_df, age_df, conf_df, hits_df, match_df, grid_df):
    fig,axes=plt.subplots(2,3,figsize=(18,10))

    for ax,df,xcol,xlabel,title,color in [
        (axes[0,0], iou_df,  'iou_threshold', 'IoU Threshold',    'IoU Threshold', 'b'),
        (axes[0,1], age_df,  'max_age',        'Max Age (frames)', 'Max Age',       'g'),
        (axes[0,2], conf_df, 'min_conf',        'Min Confidence',  'Confidence',    'm'),
        (axes[1,0], hits_df, 'min_hits',        'Min Hits',        'Min Hits',      'c'),
    ]:
        ax.plot(df[xcol], df['IDF1'], f'{color}-o', ms=5, label='IDF1')
        ax2=ax.twinx()
        ax2.plot(df[xcol], df['n_ids'], 'r--s', ms=4, label='N IDs')
        ax2.set_ylabel('# Unique IDs', color='red')
        ax.set_xlabel(xlabel); ax.set_ylabel('IDF1')
        ax.set_title(f'{title}\n{det_name}')
        ax.grid(True,alpha=0.3); ax.set_ylim(0,1)

    # Greedy vs Hungarian bar
    ax=axes[1,1]
    colors=['steelblue','darkorange']
    bars=ax.bar(match_df['matcher'],match_df['IDF1'],color=colors,edgecolor='black')
    ax.bar_label(bars,fmt='%.4f',padding=3)
    ax.set_ylabel('IDF1'); ax.set_ylim(0,1)
    ax.set_title(f'Greedy vs Hungarian\n{det_name}'); ax.grid(True,alpha=0.3,axis='y')

    # Heatmap: iou × max_age for best matcher
    ax=axes[1,2]
    best_matcher=grid_df.groupby('matcher')['IDF1'].mean().idxmax()
    sub=grid_df[(grid_df['matcher']==best_matcher)&(grid_df['min_conf']==0.0)&(grid_df['min_hits']==3)]
    if not sub.empty:
        pivot=sub.pivot_table(index='max_age',columns='iou_threshold',values='IDF1',aggfunc='mean')
        im=ax.imshow(pivot.values,cmap='YlGn',vmin=0,vmax=1,aspect='auto')
        plt.colorbar(im,ax=ax)
        ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels([f'{v:.1f}' for v in pivot.columns],fontsize=8)
        ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index,fontsize=8)
        ax.set_xlabel('IoU Threshold'); ax.set_ylabel('Max Age')
        ax.set_title(f'IDF1 Heatmap ({best_matcher})\n{det_name}')
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                ax.text(j,i,f'{pivot.values[i,j]:.2f}',ha='center',va='center',fontsize=7)

    plt.suptitle(f'Task 2.2 SORT Tracker Analysis — {det_name}',fontsize=13)
    plt.tight_layout()
    fname=f'task2_2_analysis_{det_name.replace("/","_")}.png'
    plt.savefig(os.path.join(PLOTS_DIR,fname),dpi=150,bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {fname}")


def save_summary_plot(summary_rows):
    df=pd.DataFrame(summary_rows)
    if df.empty: return
    detectors=df['detector'].unique()
    x=np.arange(len(detectors)); width=0.35
    fig,ax=plt.subplots(figsize=(max(8,len(detectors)*2),5))
    g_vals=[df[(df['detector']==d)&(df['tracker']=='greedy_best')]['IDF1'].values for d in detectors]
    h_vals=[df[(df['detector']==d)&(df['tracker']=='hungarian_best')]['IDF1'].values for d in detectors]
    g_vals=[v[0] if len(v) else 0 for v in g_vals]
    h_vals=[v[0] if len(v) else 0 for v in h_vals]
    b1=ax.bar(x-width/2,g_vals,width,label='Greedy (best)',color='steelblue',edgecolor='black')
    b2=ax.bar(x+width/2,h_vals,width,label='Hungarian (best)',color='darkorange',edgecolor='black')
    ax.bar_label(b1,fmt='%.3f',padding=3,fontsize=8)
    ax.bar_label(b2,fmt='%.3f',padding=3,fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(detectors,rotation=20,ha='right')
    ax.set_ylabel('IDF1 (grid-search best)'); ax.set_ylim(0,1.1)
    ax.set_title('Task 2.2 — Best IDF1: Greedy vs Hungarian per Detector')
    ax.legend(); ax.grid(True,alpha=0.3,axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR,'task2_2_summary.png'),dpi=150); plt.close()


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("="*60)
    print("TASK 2.2: Tracking with Kalman Filter (SORT)")
    print("="*60)

    gt_dict = load_gt()
    loader  = VideoFrameLoader(video_path=VIDEO_PATH)

    # ── Detection sources ─────────────────────────────────────────
    det_files={}
    for name,fname in [
        ('MaskRCNN_provided','det_mask_rcnn.txt'),
        ('SSD512_provided',  'det_ssd512.txt'),
        ('YOLOv3_provided',  'det_yolo3.txt'),
    ]:
        p=os.path.join(DET_DIR,fname)
        if os.path.exists(p): det_files[name]=p

    t11_path='results/task1_1/best_model.json'
    if os.path.exists(t11_path):
        t11=json.load(open(t11_path))
        best_name=t11.get('best_model_name',t11.get('model_name',''))
        det_txt=t11.get('det_txt','')
        if not det_txt or not os.path.exists(det_txt):
            det_txt=os.path.join('results','task1_1','detections',f'{best_name}.txt')
        if best_name and os.path.exists(det_txt):
            det_files[best_name]=det_txt
            print(f"  [task1_1] Loaded best: {best_name}")

    t12_path='results/task1_2/best_config.json'
    if os.path.exists(t12_path):
        t12=json.load(open(t12_path))
        ft_model=t12.get('model_name','')
        ft_name=f'{ft_model}_finetuned'
        ft_det=os.path.join('results','task1_2',f'{ft_name}_detections.txt')
        if os.path.exists(ft_det):
            det_files[ft_name]=ft_det
            print(f"  [task1_2] Loaded fine-tuned: {ft_name}")

    if not det_files:
        print("[ERROR] No detection files found."); return

    # ── Per-detector processing ───────────────────────────────────
    all_best_configs={}; summary_rows=[]; metrics_rows=[]

    for det_name,det_path in det_files.items():
        if not os.path.exists(det_path): continue
        print(f"\n{'='*50}\nDetector: {det_name}\n{'='*50}")
        det_dict=parse_detections_mot(det_path)
        pre  = len(set(det_dict.keys()) & set(gt_dict.keys()))
        det_dict = align_det_to_gt(det_dict, gt_dict)
        post = len(set(det_dict.keys()) & set(gt_dict.keys()))
        print(f"  Frame alignment: overlap {pre}→{post} / {len(gt_dict)} GT frames")

        # Grid search
        best_cfg,grid_df=grid_search(gt_dict,det_dict,det_name)
        all_best_configs[det_name]=best_cfg

        # Run best greedy and best hungarian configs
        best_greedy=grid_df[grid_df['matcher']=='greedy'].nlargest(1,'IDF1').iloc[0]
        best_hung  =grid_df[grid_df['matcher']=='hungarian'].nlargest(1,'IDF1').iloc[0]

        for cfg_row,label in [(best_greedy,'greedy_best'),(best_hung,'hungarian_best')]:
            tracker=SORTTracker(
                iou_threshold=cfg_row['iou_threshold'],
                max_age=int(cfg_row['max_age']),
                min_hits=int(cfg_row['min_hits']),
                use_hungarian=(cfg_row['matcher']=='hungarian'))
            tracks=run_tracker(det_dict,tracker,min_confidence=float(cfg_row['min_conf']))
            n_ids=len({int(t[4]) for fid in tracks for t in tracks[fid]})
            idf1=quick_idf1(gt_dict,tracks)
            print(f"  [{label}] {n_ids} IDs  IDF1={idf1:.4f}")
            metrics_rows.append({'detector':det_name,'tracker':label,
                                  'n_ids':n_ids,'IDF1':round(idf1,4)})
            summary_rows.append({'detector':det_name,'tracker':label,'IDF1':round(idf1,4)})

        # Save primary track file using overall best config (for task2_3)
        overall_best = best_cfg
        tracker=SORTTracker(
            iou_threshold=overall_best['iou_threshold'],
            max_age=int(overall_best['max_age']),
            min_hits=int(overall_best['min_hits']),
            use_hungarian=(overall_best['matcher']=='hungarian'))
        best_tracks=run_tracker(det_dict,tracker,
                                min_confidence=float(overall_best['min_conf']))

        # Save under both naming conventions task2_3 expects
        sort_path   =os.path.join(RESULTS_DIR,f'{det_name}_sort_tracks.txt')
        kalman_path =os.path.join(RESULTS_DIR,f'{det_name}_kalman_tracker.txt')
        save_tracks_mot(best_tracks, sort_path)
        save_tracks_mot(best_tracks, kalman_path)
        print(f"  Saved tracks → {sort_path}")

        # Ablation curves
        print("  Running ablation curves ...")
        iou_df  =ablation_iou( gt_dict,det_dict,max_age=int(best_greedy['max_age']),
                               min_hits=int(best_greedy['min_hits']),min_conf=float(best_greedy['min_conf']))
        age_df  =ablation_age( gt_dict,det_dict,iou_thr=float(best_greedy['iou_threshold']),
                               min_hits=int(best_greedy['min_hits']),min_conf=float(best_greedy['min_conf']))
        conf_df =ablation_conf(gt_dict,det_dict,iou_thr=float(best_greedy['iou_threshold']),
                               max_age=int(best_greedy['max_age']),min_hits=int(best_greedy['min_hits']))
        hits_df =ablation_min_hits(gt_dict,det_dict,iou_thr=float(best_greedy['iou_threshold']),
                               max_age=int(best_greedy['max_age']),min_conf=float(best_greedy['min_conf']))
        match_df=compare_matchers(gt_dict,det_dict,
                               iou_thr=float(best_greedy['iou_threshold']),
                               max_age=int(best_greedy['max_age']),
                               min_hits=int(best_greedy['min_hits']),
                               min_conf=float(best_greedy['min_conf']))

        for df_abl,fname in [
            (iou_df,  f'{det_name}_iou_abl.csv'),
            (age_df,  f'{det_name}_age_abl.csv'),
            (conf_df, f'{det_name}_conf_abl.csv'),
            (hits_df, f'{det_name}_hits_abl.csv'),
            (match_df,f'{det_name}_matcher_abl.csv'),
        ]:
            df_abl.to_csv(os.path.join(RESULTS_DIR,fname),index=False)

        save_ablation_plots(det_name,iou_df,age_df,conf_df,hits_df,match_df,grid_df)
        save_qualitative(loader,best_tracks,gt_dict,det_name,suffix='_best')

    # ── Save ─────────────────────────────────────────────────────
    with open(os.path.join(RESULTS_DIR,'best_configs.json'),'w') as f:
        json.dump(all_best_configs,f,indent=2)

    metrics_df=pd.DataFrame(metrics_rows)
    metrics_df.to_csv(os.path.join(RESULTS_DIR,'summary.csv'),index=False)
    print(f"\nSaved → {RESULTS_DIR}/")
    print(metrics_df.to_string(index=False))

    save_summary_plot(summary_rows)
    print(f"\nPlots saved to {PLOTS_DIR}/")
    print("\n✓ Task 2.2 complete.")


if __name__ == '__main__':
    main()