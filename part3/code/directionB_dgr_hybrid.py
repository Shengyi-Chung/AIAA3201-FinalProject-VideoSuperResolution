#!/usr/bin/env python3
import argparse, csv
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np

def read_rgb(p):
    im=cv2.imread(str(p),cv2.IMREAD_COLOR)
    if im is None: raise FileNotFoundError(p)
    return cv2.cvtColor(im,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0

def save_rgb(p,img):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    u8=np.clip(img*255,0,255).round().astype(np.uint8)
    cv2.imwrite(str(p),cv2.cvtColor(u8,cv2.COLOR_RGB2BGR))

def gray(x):
    return cv2.cvtColor(np.clip(x*255,0,255).astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0

def norm01(x,eps=1e-8):
    lo,hi=np.percentile(x,2),np.percentile(x,98)
    return np.clip((x-lo)/(hi-lo+eps),0,1)

def colorize(x):
    u8=np.clip(norm01(x)*255,0,255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(u8,cv2.COLORMAP_JET),cv2.COLOR_BGR2RGB).astype(np.float32)/255.0

def hp(x,sigma):
    return x-cv2.GaussianBlur(x,(0,0),sigma)

def texmap(x):
    g=gray(x)
    gx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)
    return norm01(cv2.GaussianBlur(np.sqrt(gx*gx+gy*gy),(0,0),1.2))

def dgr(fugr,cn,alpha,sigma,tau):
    dis=cv2.GaussianBlur(np.mean(np.abs(fugr-cn),axis=2),(0,0),1.5)
    mask=np.clip(cv2.GaussianBlur(texmap(fugr)*np.exp(-dis/tau),(0,0),1.0),0,1)
    detail=hp(cn,sigma)-hp(fugr,sigma)
    out=np.clip(fugr+alpha*mask[...,None]*detail,0,1)
    return out,mask

def psnr(a,b):
    mse=float(np.mean((a-b)**2))
    return 99.0 if mse<1e-12 else float(20*np.log10(1.0/np.sqrt(mse)))

def ssim(a,b):
    c1,c2=0.01**2,0.03**2
    vals=[]
    for ch in range(3):
        x=a[...,ch].astype(np.float32); y=b[...,ch].astype(np.float32)
        mx=cv2.GaussianBlur(x,(11,11),1.5); my=cv2.GaussianBlur(y,(11,11),1.5)
        vx=cv2.GaussianBlur(x*x,(11,11),1.5)-mx*mx
        vy=cv2.GaussianBlur(y*y,(11,11),1.5)-my*my
        vxy=cv2.GaussianBlur(x*y,(11,11),1.5)-mx*my
        vals.append(float(np.mean(((2*mx*my+c1)*(2*vxy+c2))/((mx*mx+my*my+c1)*(vx+vy+c2)+1e-12))))
    return float(np.mean(vals))

def sharp(x):
    return float(np.var(cv2.Laplacian(gray(x),cv2.CV_32F)))

def tde(imgs,gts):
    if len(imgs)<2: return 0.0
    return float(np.mean([np.mean(np.abs((imgs[i]-imgs[i-1])-(gts[i]-gts[i-1]))) for i in range(1,len(imgs))]))

def err(a,b):
    return colorize(np.mean(np.abs(a-b),axis=2))

def resize_h(x,h=220):
    H,W=x.shape[:2]
    return cv2.resize(x,(int(W*h/H),h),interpolation=cv2.INTER_AREA)

def panel(path,imgs,titles):
    imgs=[resize_h(x) for x in imgs]; th=34; h=imgs[0].shape[0]; w=sum(x.shape[1] for x in imgs)
    can=np.ones((h+th,w,3),np.uint8)*255; x0=0
    for im,t in zip(imgs,titles):
        ww=im.shape[1]; can[th:,x0:x0+ww]=np.clip(im*255,0,255).round().astype(np.uint8)
        cv2.putText(can,t,(x0+6,24),cv2.FONT_HERSHEY_SIMPLEX,0.52,(0,0,0),2,cv2.LINE_AA); x0+=ww
    path.parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(path),cv2.cvtColor(can,cv2.COLOR_RGB2BGR))

def find_samples(input_dir):
    root=Path(input_dir)/"frames"; samples=[]
    for fp in sorted(root.glob("*/*_fugr.png")):
        seq=fp.parent.name; stem=fp.name.replace("_fugr.png","")
        cn=fp.parent/f"{stem}_controlnet_fugr.png"; gt=fp.parent/f"{stem}_gt.png"; basic=fp.parent/f"{stem}_basic.png"
        if cn.exists() and gt.exists(): samples.append((seq,stem,basic if basic.exists() else fp,fp,cn,gt))
    return samples

def metrics(seq,frame,method,img,gt,a="",s="",tau="",mask_mean=""):
    return dict(sequence=seq,frame=frame,method=method,alpha=a,sigma=s,tau=tau,psnr=psnr(img,gt),ssim=ssim(img,gt),laplacian_sharpness=sharp(img),mask_mean=mask_mean)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input_dir",required=True)
    ap.add_argument("--out_dir",required=True)
    ap.add_argument("--alphas",nargs="+",type=float,default=[0.02,0.05,0.10,0.15,0.20])
    ap.add_argument("--sigmas",nargs="+",type=float,default=[1.0,1.6,2.4])
    ap.add_argument("--taus",nargs="+",type=float,default=[0.05,0.08,0.12])
    ap.add_argument("--panel_limit",type=int,default=12)
    args=ap.parse_args()
    out=Path(args.out_dir); md=out/"metrics"; bd=out/"best_frames"; pd=out/"panels"
    md.mkdir(parents=True,exist_ok=True); bd.mkdir(parents=True,exist_ok=True); pd.mkdir(parents=True,exist_ok=True)

    raw=[]
    for seq,stem,basicp,fugrp,cnp,gtp in find_samples(args.input_dir):
        raw.append(dict(seq=seq,frame=stem,basic=read_rgb(basicp),fugr=read_rgb(fugrp),cn=read_rgb(cnp),gt=read_rgb(gtp)))
    print("Found samples:",len(raw),flush=True)
    if not raw: raise RuntimeError("No samples found.")

    rows=[]
    for r in raw:
        rows += [metrics(r["seq"],r["frame"],"BasicVSR",r["basic"],r["gt"]),
                 metrics(r["seq"],r["frame"],"FUGR-C",r["fugr"],r["gt"]),
                 metrics(r["seq"],r["frame"],"ControlNet-FUGR",r["cn"],r["gt"])]

    cfg_imgs={}
    for a in args.alphas:
        for s in args.sigmas:
            for tau in args.taus:
                m=f"DGR-a{a:.3f}-s{s:.1f}-t{tau:.2f}"
                cfg_imgs[m]=[]
                for r in raw:
                    im,mask=dgr(r["fugr"],r["cn"],a,s,tau)
                    cfg_imgs[m].append((r["seq"],r["frame"],im,r["gt"],mask,r))
                    rows.append(metrics(r["seq"],r["frame"],m,im,r["gt"],a,s,tau,float(mask.mean())))

    frame_csv=md/"dgr_frame_metrics.csv"
    with frame_csv.open("w",newline="") as f:
        fields=["sequence","frame","method","alpha","sigma","tau","psnr","ssim","laplacian_sharpness","mask_mean"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

    summary=[]
    for m in sorted(set(x["method"] for x in rows)):
        sub=[x for x in rows if x["method"]==m]
        tdes=[]
        for seq in sorted(set(r["seq"] for r in raw)):
            items=sorted([r for r in raw if r["seq"]==seq],key=lambda z:z["frame"])
            if m=="BasicVSR": imgs=[z["basic"] for z in items]
            elif m=="FUGR-C": imgs=[z["fugr"] for z in items]
            elif m=="ControlNet-FUGR": imgs=[z["cn"] for z in items]
            else:
                imgs=[z[2] for z in sorted([z for z in cfg_imgs[m] if z[0]==seq],key=lambda z:z[1])]
            gts=[z["gt"] for z in items]
            tdes.append(tde(imgs,gts))
        summary.append(dict(method=m,num_frames=len(sub),psnr=float(np.mean([x["psnr"] for x in sub])),
                            ssim=float(np.mean([x["ssim"] for x in sub])),
                            laplacian_sharpness=float(np.mean([x["laplacian_sharpness"] for x in sub])),
                            tde=float(np.mean(tdes)),
                            mask_mean=float(np.mean([float(x["mask_mean"]) for x in sub if x["mask_mean"]!=""])) if any(x["mask_mean"]!="" for x in sub) else ""))

    summary.sort(key=lambda x:(x["psnr"],x["ssim"]),reverse=True)
    with (md/"dgr_config_summary.csv").open("w",newline="") as f:
        fields=["method","num_frames","psnr","ssim","laplacian_sharpness","tde","mask_mean"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(summary)

    dgrs=[x for x in summary if x["method"].startswith("DGR-")]
    best=dgrs[0]
    with (md/"dgr_best_summary.txt").open("w") as f:
        f.write("Direction B10: Diffusion-Guided Residual Hybrid\n")
        f.write(f"input_dir: {args.input_dir}\nnum_samples: {len(raw)}\n\n")
        f.write("Top 15 methods/configs by PSNR:\n")
        for r in summary[:15]:
            f.write(f"{r['method']},{r['num_frames']},{r['psnr']:.6f},{r['ssim']:.6f},{r['laplacian_sharpness']:.8f},{r['tde']:.8f},{r['mask_mean']}\n")
        f.write("\nBest DGR:\n"+str(best)+"\n")
    print((md/"dgr_best_summary.txt").read_text(),flush=True)

    best_m=best["method"]; count=0
    for seq,frame,im,gt,mask,r in cfg_imgs[best_m]:
        save_rgb(bd/seq/f"{frame}_dgr.png",im); save_rgb(bd/seq/f"{frame}_mask.png",colorize(mask))
        if count<args.panel_limit:
            panel(pd/f"panel_{seq}_{frame}.png",
                  [r["fugr"],r["cn"],im,gt,err(r["fugr"],gt),err(r["cn"],gt),err(im,gt),colorize(mask)],
                  ["FUGR-C","CN-FUGR","DGR","GT","FUGR Err","CN Err","DGR Err","DGR Mask"])
            count+=1
    print("Saved:",out,flush=True)

if __name__=="__main__":
    main()
