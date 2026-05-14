#!/usr/bin/env python3
import argparse, csv
from pathlib import Path
from collections import defaultdict
import cv2, numpy as np

def read(p):
    im=cv2.imread(str(p),cv2.IMREAD_COLOR)
    if im is None: raise FileNotFoundError(p)
    return cv2.cvtColor(im,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0

def save(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    u=np.clip(x*255,0,255).round().astype(np.uint8)
    cv2.imwrite(str(p),cv2.cvtColor(u,cv2.COLOR_RGB2BGR))

def g(x):
    return cv2.cvtColor(np.clip(x*255,0,255).astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0

def psnr(a,b):
    m=float(np.mean((a-b)**2))
    return 99.0 if m<1e-12 else float(20*np.log10(1/np.sqrt(m)))

def ssim(a,b):
    c1,c2=0.01**2,0.03**2; vals=[]
    for ch in range(3):
        x,y=a[...,ch].astype(np.float32),b[...,ch].astype(np.float32)
        mx,my=cv2.GaussianBlur(x,(11,11),1.5),cv2.GaussianBlur(y,(11,11),1.5)
        vx=cv2.GaussianBlur(x*x,(11,11),1.5)-mx*mx
        vy=cv2.GaussianBlur(y*y,(11,11),1.5)-my*my
        vxy=cv2.GaussianBlur(x*y,(11,11),1.5)-mx*my
        vals.append(float(np.mean(((2*mx*my+c1)*(2*vxy+c2))/((mx*mx+my*my+c1)*(vx+vy+c2)+1e-12))))
    return float(np.mean(vals))

def sharp(x):
    return float(np.var(cv2.Laplacian(g(x),cv2.CV_32F)))

def tde(xs,ys):
    if len(xs)<2: return 0.0
    return float(np.mean([np.mean(np.abs((xs[i]-xs[i-1])-(ys[i]-ys[i-1]))) for i in range(1,len(xs))]))

def mot(ys):
    if len(ys)<2: return 0.0
    return float(np.mean([np.mean(np.abs(ys[i]-ys[i-1])) for i in range(1,len(ys))]))

def flow(cur,ref):
    return cv2.calcOpticalFlowFarneback(g(cur),g(ref),None,0.5,3,15,3,5,1.2,0)

def warp(ref,fl):
    h,w=fl.shape[:2]; xx,yy=np.meshgrid(np.arange(w),np.arange(h))
    return cv2.remap(ref,(xx+fl[...,0]).astype(np.float32),(yy+fl[...,1]).astype(np.float32),
                     cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)

def c01(x):
    lo,hi=np.percentile(x,2),np.percentile(x,98)
    y=np.clip((x-lo)/(hi-lo+1e-8),0,1)
    cm=cv2.applyColorMap((y*255).astype(np.uint8),cv2.COLORMAP_JET)
    return cv2.cvtColor(cm,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0

def err(a,b):
    return c01(np.mean(np.abs(a-b),axis=2))

def resize(x,H=210):
    h,w=x.shape[:2]
    return cv2.resize(x,(int(w*H/h),H),interpolation=cv2.INTER_AREA)

def panel(path,imgs,names):
    imgs=[resize(x) for x in imgs]; th=32; H=imgs[0].shape[0]; W=sum(x.shape[1] for x in imgs)
    can=np.ones((H+th,W,3),np.uint8)*255; x0=0
    for im,n in zip(imgs,names):
        w=im.shape[1]; can[th:,x0:x0+w]=np.clip(im*255,0,255).round().astype(np.uint8)
        cv2.putText(can,n,(x0+5,23),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA); x0+=w
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(path),cv2.cvtColor(can,cv2.COLOR_RGB2BGR))

def collect(root):
    root=Path(root)/"frames"; data={}
    for sd in sorted(root.iterdir()):
        if not sd.is_dir(): continue
        items=[]
        for fp in sorted(sd.glob("*_fugr.png")):
            fr=fp.name.replace("_fugr.png",""); bp=sd/f"{fr}_basic.png"; gp=sd/f"{fr}_gt.png"
            if bp.exists() and gp.exists(): items.append((fr,bp,fp,gp))
        if items: data[sd.name]=items
    return data

def residual(basic,fugr):
    return [f-b for b,f in zip(basic,fugr)]

def rts_noflow(basic,fugr,i,beta):
    r=residual(basic,fugr); acc=r[i].copy(); w=1.0
    if i>0: acc+=beta*r[i-1]; w+=beta
    if i+1<len(r): acc+=beta*r[i+1]; w+=beta
    return np.clip(basic[i]+acc/w,0,1), np.zeros(basic[i].shape[:2],np.float32)

def rts_flow(basic,fugr,i,beta):
    r=residual(basic,fugr); cur=basic[i]; acc=r[i].copy(); w=1.0; gates=[]
    for j in (i-1,i+1):
        if j<0 or j>=len(r): continue
        fl=flow(cur,basic[j]); acc+=beta*warp(r[j],fl); w+=beta; gates.append(np.ones(cur.shape[:2],np.float32))
    return np.clip(cur+acc/w,0,1), (np.mean(gates,axis=0) if gates else np.zeros(cur.shape[:2],np.float32))

def rts_gate(basic,fugr,i,beta,tau,mtau):
    r=residual(basic,fugr); cur=basic[i]
    acc=r[i].copy(); wt=np.ones(cur.shape[:2],np.float32); gates=[]
    for j in (i-1,i+1):
        if j<0 or j>=len(r): continue
        fl=flow(cur,basic[j])
        wb=warp(basic[j],fl); wr=warp(r[j],fl)
        mag=np.sqrt(fl[...,0]**2+fl[...,1]**2)
        e=np.mean(np.abs(wb-cur),axis=2)
        gate=np.exp(-e/tau)*np.exp(-mag/mtau)
        gate=np.clip(cv2.GaussianBlur(gate,(0,0),1.0),0,1)
        w=beta*gate; acc+=w[...,None]*wr; wt+=w; gates.append(gate)
    return np.clip(cur+acc/wt[...,None],0,1), (np.mean(gates,axis=0) if gates else np.zeros(cur.shape[:2],np.float32))

def eval_seq(xs,ys):
    return dict(psnr=float(np.mean([psnr(x,y) for x,y in zip(xs,ys)])),
                ssim=float(np.mean([ssim(x,y) for x,y in zip(xs,ys)])),
                laplacian_sharpness=float(np.mean([sharp(x) for x in xs])),
                tde=tde(xs,ys))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input_dir",required=True); ap.add_argument("--out_dir",required=True)
    ap.add_argument("--betas",nargs="+",type=float,default=[0.03,0.05,0.08,0.10,0.15,0.20,0.30,0.50])
    ap.add_argument("--taus",nargs="+",type=float,default=[0.02,0.04,0.06,0.08])
    ap.add_argument("--motion_taus",nargs="+",type=float,default=[0.5,1.0,2.0,4.0])
    ap.add_argument("--panel_limit",type=int,default=12)
    args=ap.parse_args()
    out=Path(args.out_dir); md=out/"metrics"; pd=out/"panels"; fd=out/"best_frames"
    md.mkdir(parents=True,exist_ok=True); pd.mkdir(parents=True,exist_ok=True); fd.mkdir(parents=True,exist_ok=True)
    raw=collect(args.input_dir); print("Sequences:",sorted(raw.keys()),flush=True)
    data={}
    for s,items in raw.items():
        data[s]={"frames":[x[0] for x in items],"basic":[read(x[1]) for x in items],
                 "fugr":[read(x[2]) for x in items],"gt":[read(x[3]) for x in items]}
    seq_rows=[]; frame_rows=[]
    def add(name,outs):
        for s,xs in outs.items():
            ys=data[s]["gt"]; met=eval_seq(xs,ys)
            seq_rows.append({"sequence":s,"method":name,"num_frames":len(xs),"motion":mot(ys),**met})
            for fr,x,y in zip(data[s]["frames"],xs,ys):
                frame_rows.append({"sequence":s,"frame":fr,"method":name,"psnr":psnr(x,y),"ssim":ssim(x,y),"laplacian_sharpness":sharp(x)})
    add("BasicVSR",{s:data[s]["basic"] for s in data})
    add("FUGR-C",{s:data[s]["fugr"] for s in data})
    for b in args.betas:
        print("Evaluating",f"RTS-NoFlow-b{b:.2f}",flush=True)
        add(f"RTS-NoFlow-b{b:.2f}",{s:[rts_noflow(data[s]["basic"],data[s]["fugr"],i,b)[0] for i in range(len(data[s]["fugr"]))] for s in data})
        print("Evaluating",f"RTS-Flow-b{b:.2f}",flush=True)
        add(f"RTS-Flow-b{b:.2f}",{s:[rts_flow(data[s]["basic"],data[s]["fugr"],i,b)[0] for i in range(len(data[s]["fugr"]))] for s in data})
    total=len(args.betas)*len(args.taus)*len(args.motion_taus); k=0
    for b in args.betas:
        for t in args.taus:
            for m in args.motion_taus:
                k+=1; name=f"RTS-Gated-b{b:.2f}-t{t:.2f}-m{m:.1f}"
                print(f"[{k}/{total}] {name}",flush=True)
                add(name,{s:[rts_gate(data[s]["basic"],data[s]["fugr"],i,b,t,m)[0] for i in range(len(data[s]["fugr"]))] for s in data})
    with (md/"directionA2_sequence_metrics.csv").open("w",newline="") as f:
        fields=["sequence","method","num_frames","motion","psnr","ssim","laplacian_sharpness","tde"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(seq_rows)
    with (md/"directionA2_frame_metrics.csv").open("w",newline="") as f:
        fields=["sequence","frame","method","psnr","ssim","laplacian_sharpness"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(frame_rows)
    base=[r for r in seq_rows if r["method"]=="FUGR-C"]
    bps=float(np.mean([r["psnr"] for r in base])); bt=float(np.mean([r["tde"] for r in base]))
    summary=[]
    for meth in sorted(set(r["method"] for r in seq_rows)):
        rows=[r for r in seq_rows if r["method"]==meth]
        rec={"method":meth,"num_sequences":len(rows),"num_frames":int(sum(r["num_frames"] for r in rows)),
             "motion":float(np.mean([r["motion"] for r in rows])),
             "psnr":float(np.mean([r["psnr"] for r in rows])),
             "ssim":float(np.mean([r["ssim"] for r in rows])),
             "laplacian_sharpness":float(np.mean([r["laplacian_sharpness"] for r in rows])),
             "tde":float(np.mean([r["tde"] for r in rows]))}
        rec["delta_psnr_vs_fugr"]=rec["psnr"]-bps
        rec["delta_tde_vs_fugr"]=rec["tde"]-bt
        rec["tde_reduction_pct"]=100*(bt-rec["tde"])/bt if bt>0 else 0
        summary.append(rec)
    summary.sort(key=lambda r:(r["tde"],-r["psnr"]))
    with (md/"directionA2_summary_metrics.csv").open("w",newline="") as f:
        fields=["method","num_sequences","num_frames","motion","psnr","ssim","laplacian_sharpness","tde","delta_psnr_vs_fugr","delta_tde_vs_fugr","tde_reduction_pct"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(summary)
    cand=[r for r in summary if r["method"]!="FUGR-C" and r["delta_psnr_vs_fugr"]>-0.05]
    best=cand[0] if cand else summary[0]
    txt=md/"directionA2_best_summary.txt"
    with txt.open("w") as f:
        f.write("Direction A2: Residual Temporal Stabilization\n\n")
        f.write(f"FUGR-C baseline: PSNR={bps:.6f}, TDE={bt:.8f}\n\n")
        f.write("Top 25 methods by TDE:\n")
        for r in summary[:25]:
            f.write(f"{r['method']},{r['num_frames']},{r['psnr']:.6f},{r['ssim']:.6f},{r['laplacian_sharpness']:.8f},{r['tde']:.8f},dPSNR={r['delta_psnr_vs_fugr']:.6f},TDEred={r['tde_reduction_pct']:.3f}%\n")
        f.write("\nSelected best under PSNR-loss constraint:\n"+str(best)+"\n")
    print(txt.read_text(),flush=True)
    bestn=best["method"]
    def best_out(s):
        ba,fu=data[s]["basic"],data[s]["fugr"]; outs=[]; gates=[]
        if bestn=="BasicVSR": return ba,[np.zeros(ba[0].shape[:2],np.float32) for _ in ba]
        if bestn=="FUGR-C": return fu,[np.zeros(ba[0].shape[:2],np.float32) for _ in ba]
        if bestn.startswith("RTS-NoFlow"):
            b=float(bestn.split("-b")[1])
            for i in range(len(fu)):
                o,gg=rts_noflow(ba,fu,i,b); outs.append(o); gates.append(gg)
            return outs,gates
        if bestn.startswith("RTS-Flow"):
            b=float(bestn.split("-b")[1])
            for i in range(len(fu)):
                o,gg=rts_flow(ba,fu,i,b); outs.append(o); gates.append(gg)
            return outs,gates
        parts=bestn.split("-"); b=float(parts[2][1:]); t=float(parts[3][1:]); m=float(parts[4][1:])
        for i in range(len(fu)):
            o,gg=rts_gate(ba,fu,i,b,t,m); outs.append(o); gates.append(gg)
        return outs,gates
    pc=0
    for s in data:
        outs,gates=best_out(s)
        for fr,o,gg,ba,fu,gt in zip(data[s]["frames"],outs,gates,data[s]["basic"],data[s]["fugr"],data[s]["gt"]):
            save(fd/s/f"{fr}_directionA2.png",o); save(fd/s/f"{fr}_gate.png",c01(gg))
            if pc<args.panel_limit:
                panel(pd/f"panel_{s}_{fr}.png",[ba,fu,o,gt,err(fu,gt),err(o,gt),c01(gg)],["Basic","FUGR-C","A2-RTS","GT","FUGR Err","A2 Err","Gate"]); pc+=1
    print("Saved:",out,flush=True)

if __name__=="__main__":
    main()
