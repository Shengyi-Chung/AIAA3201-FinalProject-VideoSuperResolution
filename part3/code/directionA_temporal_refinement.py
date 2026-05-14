#!/usr/bin/env python3
import argparse, csv
from pathlib import Path
from collections import defaultdict
import cv2, numpy as np

def read_rgb(p):
    im=cv2.imread(str(p),cv2.IMREAD_COLOR)
    if im is None: raise FileNotFoundError(p)
    return cv2.cvtColor(im,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0

def save_rgb(p,img):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    u8=np.clip(img*255,0,255).round().astype(np.uint8)
    cv2.imwrite(str(p),cv2.cvtColor(u8,cv2.COLOR_RGB2BGR))

def gray(img):
    return cv2.cvtColor(np.clip(img*255,0,255).astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0

def colorize(x):
    lo,hi=np.percentile(x,2),np.percentile(x,98)
    y=np.clip((x-lo)/(hi-lo+1e-8),0,1)
    c=cv2.applyColorMap((y*255).astype(np.uint8),cv2.COLORMAP_JET)
    return cv2.cvtColor(c,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0

def psnr(a,b):
    mse=float(np.mean((a-b)**2))
    return 99.0 if mse<1e-12 else float(20*np.log10(1.0/np.sqrt(mse)))

def ssim_rgb(a,b):
    c1,c2=0.01**2,0.03**2; vals=[]
    for ch in range(3):
        x,y=a[...,ch].astype(np.float32),b[...,ch].astype(np.float32)
        mx,my=cv2.GaussianBlur(x,(11,11),1.5),cv2.GaussianBlur(y,(11,11),1.5)
        vx=cv2.GaussianBlur(x*x,(11,11),1.5)-mx*mx
        vy=cv2.GaussianBlur(y*y,(11,11),1.5)-my*my
        vxy=cv2.GaussianBlur(x*y,(11,11),1.5)-mx*my
        vals.append(float(np.mean(((2*mx*my+c1)*(2*vxy+c2))/((mx*mx+my*my+c1)*(vx+vy+c2)+1e-12))))
    return float(np.mean(vals))

def sharp(img): return float(np.var(cv2.Laplacian(gray(img),cv2.CV_32F)))

def tde(imgs,gts):
    if len(imgs)<2: return 0.0
    return float(np.mean([np.mean(np.abs((imgs[i]-imgs[i-1])-(gts[i]-gts[i-1]))) for i in range(1,len(imgs))]))

def motion(gts):
    if len(gts)<2: return 0.0
    return float(np.mean([np.mean(np.abs(gts[i]-gts[i-1])) for i in range(1,len(gts))]))

def warp_ref_to_cur(ref,cur):
    flow=cv2.calcOpticalFlowFarneback(gray(cur),gray(ref),None,0.5,3,15,3,5,1.2,0)
    h,w=flow.shape[:2]; xx,yy=np.meshgrid(np.arange(w),np.arange(h))
    mx=(xx+flow[...,0]).astype(np.float32); my=(yy+flow[...,1]).astype(np.float32)
    warped=cv2.remap(ref,mx,my,interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    mag=np.sqrt(flow[...,0]**2+flow[...,1]**2)
    return warped,mag

def noflow_avg(frames,i,beta):
    cur=frames[i]; acc=cur.copy(); w=1.0
    if i>0: acc+=beta*frames[i-1]; w+=beta
    if i+1<len(frames): acc+=beta*frames[i+1]; w+=beta
    return np.clip(acc/w,0,1)

def flow_avg(frames,i,beta):
    cur=frames[i]; acc=cur.copy(); w=1.0
    if i>0:
        wp,_=warp_ref_to_cur(frames[i-1],cur); acc+=beta*wp; w+=beta
    if i+1<len(frames):
        wn,_=warp_ref_to_cur(frames[i+1],cur); acc+=beta*wn; w+=beta
    return np.clip(acc/w,0,1)

def matr(frames,i,beta,tau,motion_tau):
    cur=frames[i]; acc=cur.copy(); weight=np.ones(cur.shape[:2],np.float32); gates=[]
    for j in (i-1,i+1):
        if j<0 or j>=len(frames): continue
        warped,mag=warp_ref_to_cur(frames[j],cur)
        err=np.mean(np.abs(warped-cur),axis=2)
        gate=np.exp(-err/tau)*np.exp(-mag/motion_tau)
        gate=np.clip(cv2.GaussianBlur(gate,(0,0),1.0),0,1)
        w=beta*gate; acc+=w[...,None]*warped; weight+=w; gates.append(gate)
    return np.clip(acc/weight[...,None],0,1), (np.mean(gates,axis=0) if gates else np.zeros(cur.shape[:2],np.float32))

def collect(input_dir):
    root=Path(input_dir)/'frames'; seqs={}
    for sd in sorted(root.iterdir()):
        if not sd.is_dir(): continue
        items=[]
        for fp in sorted(sd.glob('*_fugr.png')):
            fr=fp.name.replace('_fugr.png',''); gt=sd/f'{fr}_gt.png'; basic=sd/f'{fr}_basic.png'
            if gt.exists(): items.append((fr,fp,gt,basic if basic.exists() else fp))
        if items: seqs[sd.name]=items
    return seqs

def eval_seq(frames,gts):
    return {'psnr':float(np.mean([psnr(x,y) for x,y in zip(frames,gts)])),
            'ssim':float(np.mean([ssim_rgb(x,y) for x,y in zip(frames,gts)])),
            'laplacian_sharpness':float(np.mean([sharp(x) for x in frames])),
            'tde':tde(frames,gts)}

def resize_h(img,h=220):
    H,W=img.shape[:2]; return cv2.resize(img,(int(W*h/H),h),interpolation=cv2.INTER_AREA)

def err_map(img,gt): return colorize(np.mean(np.abs(img-gt),axis=2))

def make_panel(path,imgs,titles):
    imgs=[resize_h(x) for x in imgs]; th=34; H=imgs[0].shape[0]; W=sum(x.shape[1] for x in imgs)
    canvas=np.ones((H+th,W,3),np.uint8)*255; x0=0
    for im,title in zip(imgs,titles):
        w=im.shape[1]; canvas[th:,x0:x0+w]=np.clip(im*255,0,255).round().astype(np.uint8)
        cv2.putText(canvas,title,(x0+6,24),cv2.FONT_HERSHEY_SIMPLEX,0.52,(0,0,0),2,cv2.LINE_AA); x0+=w
    path.parent.mkdir(parents=True,exist_ok=True); cv2.imwrite(str(path),cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input_dir',required=True); ap.add_argument('--out_dir',required=True)
    ap.add_argument('--betas',nargs='+',type=float,default=[0.03,0.05,0.08,0.10,0.15,0.20])
    ap.add_argument('--taus',nargs='+',type=float,default=[0.02,0.04,0.06,0.08])
    ap.add_argument('--motion_taus',nargs='+',type=float,default=[0.5,1.0,2.0,4.0])
    ap.add_argument('--panel_limit',type=int,default=12)
    args=ap.parse_args(); out=Path(args.out_dir); mdir=out/'metrics'; pdir=out/'panels'; fdir=out/'best_frames'
    mdir.mkdir(parents=True,exist_ok=True); pdir.mkdir(parents=True,exist_ok=True); fdir.mkdir(parents=True,exist_ok=True)
    raw=collect(args.input_dir); print('Sequences:',sorted(raw.keys()),flush=True)
    data={}
    for seq,items in raw.items():
        data[seq]={'frames':[x[0] for x in items],'fugr':[read_rgb(x[1]) for x in items],'gt':[read_rgb(x[2]) for x in items],'basic':[read_rgb(x[3]) for x in items]}
    seq_rows=[]; frame_rows=[]
    def add_method(name,outputs):
        for seq,outs in outputs.items():
            gts=data[seq]['gt']; met=eval_seq(outs,gts)
            seq_rows.append({'sequence':seq,'method':name,'num_frames':len(outs),'motion':motion(gts),**met})
            for fr,im,gt in zip(data[seq]['frames'],outs,gts):
                frame_rows.append({'sequence':seq,'frame':fr,'method':name,'psnr':psnr(im,gt),'ssim':ssim_rgb(im,gt),'laplacian_sharpness':sharp(im)})
    add_method('BasicVSR',{s:data[s]['basic'] for s in data}); add_method('FUGR-C',{s:data[s]['fugr'] for s in data})
    for beta in args.betas:
        add_method(f'NoFlowAvg-b{beta:.2f}',{s:[noflow_avg(data[s]['fugr'],i,beta) for i in range(len(data[s]['fugr']))] for s in data})
        add_method(f'FlowAvg-b{beta:.2f}',{s:[flow_avg(data[s]['fugr'],i,beta) for i in range(len(data[s]['fugr']))] for s in data})
    total=len(args.betas)*len(args.taus)*len(args.motion_taus); k=0
    for beta in args.betas:
        for tau in args.taus:
            for mtau in args.motion_taus:
                k+=1; name=f'MATR-b{beta:.2f}-t{tau:.2f}-m{mtau:.1f}'; print(f'[{k}/{total}] {name}',flush=True)
                add_method(name,{s:[matr(data[s]['fugr'],i,beta,tau,mtau)[0] for i in range(len(data[s]['fugr']))] for s in data})
    with (mdir/'directionA_sequence_metrics.csv').open('w',newline='') as f:
        fields=['sequence','method','num_frames','motion','psnr','ssim','laplacian_sharpness','tde']; w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(seq_rows)
    with (mdir/'directionA_frame_metrics.csv').open('w',newline='') as f:
        fields=['sequence','frame','method','psnr','ssim','laplacian_sharpness']; w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(frame_rows)
    methods=sorted(set(r['method'] for r in seq_rows)); summary=[]; base_rows=[r for r in seq_rows if r['method']=='FUGR-C']
    base_psnr=float(np.mean([r['psnr'] for r in base_rows])); base_tde=float(np.mean([r['tde'] for r in base_rows]))
    for method in methods:
        rows=[r for r in seq_rows if r['method']==method]
        rec={'method':method,'num_sequences':len(rows),'num_frames':int(sum(r['num_frames'] for r in rows)),'motion':float(np.mean([r['motion'] for r in rows])),'psnr':float(np.mean([r['psnr'] for r in rows])),'ssim':float(np.mean([r['ssim'] for r in rows])),'laplacian_sharpness':float(np.mean([r['laplacian_sharpness'] for r in rows])),'tde':float(np.mean([r['tde'] for r in rows]))}
        rec['delta_psnr_vs_fugr']=rec['psnr']-base_psnr; rec['delta_tde_vs_fugr']=rec['tde']-base_tde; rec['tde_reduction_pct']=100.0*(base_tde-rec['tde'])/base_tde if base_tde>0 else 0.0; summary.append(rec)
    summary.sort(key=lambda r:(r['tde'],-r['psnr']))
    with (mdir/'directionA_summary_metrics.csv').open('w',newline='') as f:
        fields=['method','num_sequences','num_frames','motion','psnr','ssim','laplacian_sharpness','tde','delta_psnr_vs_fugr','delta_tde_vs_fugr','tde_reduction_pct']; w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(summary)
    candidates=[r for r in summary if r['method']!='FUGR-C' and r['delta_psnr_vs_fugr']>-0.05]; best=candidates[0] if candidates else summary[0]
    with (mdir/'directionA_best_summary.txt').open('w') as f:
        f.write('Direction A: Motion-Adaptive Temporal Refinement\n\n'); f.write(f'FUGR-C baseline: PSNR={base_psnr:.6f}, TDE={base_tde:.8f}\n\n'); f.write('Top 20 methods by TDE:\n')
        for r in summary[:20]: f.write(f"{r['method']},{r['num_frames']},{r['psnr']:.6f},{r['ssim']:.6f},{r['laplacian_sharpness']:.8f},{r['tde']:.8f},dPSNR={r['delta_psnr_vs_fugr']:.6f},TDEred={r['tde_reduction_pct']:.3f}%\n")
        f.write('\nSelected best under PSNR-loss constraint:\n'); f.write(str(best)+'\n')
    print((mdir/'directionA_best_summary.txt').read_text(),flush=True)
    best_name=best['method']
    def best_outputs(seq):
        frames=data[seq]['fugr']
        if best_name=='BasicVSR': return data[seq]['basic'],[np.zeros(frames[0].shape[:2]) for _ in frames]
        if best_name=='FUGR-C': return data[seq]['fugr'],[np.zeros(frames[0].shape[:2]) for _ in frames]
        if best_name.startswith('NoFlowAvg'):
            beta=float(best_name.split('-b')[1]); return [noflow_avg(frames,i,beta) for i in range(len(frames))],[np.zeros(frames[0].shape[:2]) for _ in frames]
        if best_name.startswith('FlowAvg'):
            beta=float(best_name.split('-b')[1]); return [flow_avg(frames,i,beta) for i in range(len(frames))],[np.zeros(frames[0].shape[:2]) for _ in frames]
        parts=best_name.split('-'); beta=float(parts[1][1:]); tau=float(parts[2][1:]); mtau=float(parts[3][1:]); outs=[]; gates=[]
        for i in range(len(frames)):
            o,g=matr(frames,i,beta,tau,mtau); outs.append(o); gates.append(g)
        return outs,gates
    pc=0
    for seq in data:
        outs,gates=best_outputs(seq)
        for fr,im,gate,fugr,gt in zip(data[seq]['frames'],outs,gates,data[seq]['fugr'],data[seq]['gt']):
            save_rgb(fdir/seq/f'{fr}_directionA.png',im); save_rgb(fdir/seq/f'{fr}_gate.png',colorize(gate))
            if pc<args.panel_limit:
                make_panel(pdir/f'panel_{seq}_{fr}.png',[fugr,im,gt,err_map(fugr,gt),err_map(im,gt),colorize(gate)],['FUGR-C','DirectionA','GT','FUGR Err','A Err','Gate']); pc+=1
    print('Saved:',out,flush=True)
if __name__=='__main__': main()
