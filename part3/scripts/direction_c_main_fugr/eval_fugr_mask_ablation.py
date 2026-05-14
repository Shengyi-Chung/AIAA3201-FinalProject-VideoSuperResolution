#!/usr/bin/env python3
import argparse, csv, sys, time
from pathlib import Path
import cv2, numpy as np, torch
PART2_DIR = Path(__file__).resolve().parents[1] / 'Part2'
if not PART2_DIR.exists():
    PART2_DIR = Path.cwd().parents[0] / 'Part2'
sys.path.insert(0, str(PART2_DIR))
from model_basicvsr import BasicVSR

def read_rgb(p):
    img=cv2.imread(str(p),cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(p)
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0

def gray(x): return cv2.cvtColor(np.clip(x*255,0,255).astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0

def norm01(x,eps=1e-8):
    lo,hi=np.percentile(x,2),np.percentile(x,98)
    return np.clip((x-lo)/(hi-lo+eps),0,1)

def psnr(x,y):
    mse=float(np.mean((x-y)**2)); return 99.0 if mse<1e-12 else float(20*np.log10(1/np.sqrt(mse)))

def ssim_rgb(x,y):
    c1,c2=0.01**2,0.03**2; vals=[]
    for ch in range(3):
        a=x[...,ch].astype(np.float32); b=y[...,ch].astype(np.float32)
        ma=cv2.GaussianBlur(a,(11,11),1.5); mb=cv2.GaussianBlur(b,(11,11),1.5)
        va=cv2.GaussianBlur(a*a,(11,11),1.5)-ma*ma; vb=cv2.GaussianBlur(b*b,(11,11),1.5)-mb*mb
        vab=cv2.GaussianBlur(a*b,(11,11),1.5)-ma*mb
        vals.append(float(np.mean(((2*ma*mb+c1)*(2*vab+c2))/((ma*ma+mb*mb+c1)*(va+vb+c2)+1e-12))))
    return float(np.mean(vals))

def sharp(x): return float(np.var(cv2.Laplacian(gray(x),cv2.CV_32F)))
def tde(outs,gts):
    return 0.0 if len(outs)<2 else float(np.mean([np.mean(np.abs((outs[i]-outs[i-1])-(gts[i]-gts[i-1]))) for i in range(1,len(outs))]))
def motion(gts):
    return 0.0 if len(gts)<2 else float(np.mean([np.mean(np.abs(gts[i]-gts[i-1])) for i in range(1,len(gts))]))
def hp(x,s): return x-cv2.GaussianBlur(x,(0,0),s)
def texmap(b):
    g=gray(b); gx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)
    return norm01(cv2.GaussianBlur(np.sqrt(gx*gx+gy*gy),(0,0),1.2))
def dismap(b,g): return cv2.GaussianBlur(np.mean(np.abs(b-g),axis=2),(0,0),1.5)
def trisk(rp,rc,rn):
    if rp is None or rn is None: return np.zeros(rc.shape[:2],np.float32)
    return cv2.GaussianBlur(np.mean(np.abs(rc-0.5*(rp+rn)),axis=2),(0,0),1.2)
def fugr(b,g,a,sig,strength):
    return np.clip(b+strength*a[...,None]*(hp(g,sig)-hp(b,sig)),0,1)
def rgb(b,g,a): return np.clip((1-a[...,None])*b+a[...,None]*g,0,1)

def metrics(outs,gts):
    return dict(psnr=float(np.mean([psnr(o,gts[i]) for i,o in enumerate(outs)])),
                ssim=float(np.mean([ssim_rgb(o,gts[i]) for i,o in enumerate(outs)])),
                laplacian_sharpness=float(np.mean([sharp(o) for o in outs])),
                tde=tde(outs,gts))

def load_model(path,device,spynet):
    m=BasicVSR(spynet_path=spynet).to(device); ck=torch.load(str(path),map_location='cpu')
    st=ck['model_state_dict'] if isinstance(ck,dict) and 'model_state_dict' in ck else ck
    m.load_state_dict(st,strict=True); m.eval(); return m

def load_lr(d,names):
    return torch.stack([torch.from_numpy(read_rgb(d/n).transpose(2,0,1)).float() for n in names],0).unsqueeze(0)
@torch.no_grad()
def infer(m,x,device,amp=False):
    x=x.to(device)
    if device.type=='cuda' and amp:
        with torch.cuda.amp.autocast(): y=m(x)
    else: y=m(x)
    y=torch.clamp(y[0].detach().cpu(),0,1).permute(0,2,3,1).numpy().astype(np.float32)
    if device.type=='cuda': torch.cuda.empty_cache()
    return [y[i] for i in range(y.shape[0])]

def write_csv(path,rows,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--val_root',default='/home/schung760/shared_data/project1/val')
    ap.add_argument('--basic_ckpt',default='/home/schung760/my_storage_1T/AIAA3201-FinalProject-VideoSuperResolution/Part2/weights/basicvsr_stage1.pth')
    ap.add_argument('--gan_ckpt',default='/home/schung760/my_storage_1T/AIAA3201-FinalProject-VideoSuperResolution/Part2/weights/basicvsr_gan.pth')
    ap.add_argument('--spynet_path',default='/home/schung760/my_storage_1T/AIAA3201-FinalProject-VideoSuperResolution/Part2/weights/spynet.pth')
    ap.add_argument('--out_dir',required=True); ap.add_argument('--frame_mode',choices=['center7','all'],default='center7'); ap.add_argument('--seq_limit',type=int,default=0)
    ap.add_argument('--max_alpha',type=float,default=0.25); ap.add_argument('--tau_dis',type=float,default=0.08); ap.add_argument('--tau_temp',type=float,default=0.04)
    ap.add_argument('--hp_sigma',type=float,default=1.6); ap.add_argument('--detail_strength',type=float,default=1.2); ap.add_argument('--amp',action='store_true')
    args=ap.parse_args(); start=time.time(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    val=Path(args.val_root); hr=val/'val_sharp'; lrroot=val/'val_sharp_bicubic'/'X4'
    seqs=sorted([p.name for p in hr.iterdir() if p.is_dir() and (lrroot/p.name).is_dir()]);
    if args.seq_limit>0: seqs=seqs[:args.seq_limit]
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('Mask ablation',len(seqs),'seqs',args.frame_mode,dev,flush=True)
    basic=load_model(Path(args.basic_ckpt),dev,args.spynet_path); gan=load_model(Path(args.gan_ckpt),dev,args.spynet_path)
    seq_rows=[]
    for si,seq in enumerate(seqs):
        t0=time.time(); hrd=hr/seq; lrd=lrroot/seq; names=sorted([p.name for p in hrd.glob('*.png') if (lrd/p.name).exists()])
        if args.frame_mode=='center7': names=names[:7]
        print(f'[{si+1}/{len(seqs)}] {seq}: {len(names)} frames',flush=True)
        gts=[read_rgb(hrd/n) for n in names]; mot=motion(gts)
        b=infer(basic,load_lr(lrd,names),dev,args.amp); g=infer(gan,load_lr(lrd,names),dev,args.amp)
        res=[hp(g[i],args.hp_sigma)-hp(b[i],args.hp_sigma) for i in range(len(names))]
        outs={k:[] for k in ['BasicVSR','VSRGAN','RGB-Hybrid','FUGR-constant-alpha','FUGR-texture-only','FUGR-uncertainty-only','FUGR-full-no-temporal','FUGR-full-temporal']}
        amean={k:[] for k in outs}; outs['BasicVSR']=b; outs['VSRGAN']=g
        for i in range(len(names)):
            rp=res[i-1] if i>0 else None; rc=res[i]; rn=res[i+1] if i+1<len(names) else None
            tex=texmap(b[i]); dis=dismap(b[i],g[i]); rel=np.exp(-dis/args.tau_dis); temp=np.exp(-trisk(rp,rc,rn)/args.tau_temp)
            a_const=np.ones_like(tex)*args.max_alpha
            a_tex=np.clip(args.max_alpha*tex,0,args.max_alpha)
            a_unc=np.clip(args.max_alpha*rel,0,args.max_alpha)
            a_full=np.clip(cv2.GaussianBlur(args.max_alpha*tex*rel,(0,0),1.0),0,args.max_alpha)
            a_temp=np.clip(cv2.GaussianBlur(args.max_alpha*tex*rel*temp,(0,0),1.0),0,args.max_alpha)
            pairs=[('RGB-Hybrid',rgb(b[i],g[i],a_full),a_full),('FUGR-constant-alpha',fugr(b[i],g[i],a_const,args.hp_sigma,args.detail_strength),a_const),('FUGR-texture-only',fugr(b[i],g[i],a_tex,args.hp_sigma,args.detail_strength),a_tex),('FUGR-uncertainty-only',fugr(b[i],g[i],a_unc,args.hp_sigma,args.detail_strength),a_unc),('FUGR-full-no-temporal',fugr(b[i],g[i],a_full,args.hp_sigma,args.detail_strength),a_full),('FUGR-full-temporal',fugr(b[i],g[i],a_temp,args.hp_sigma,args.detail_strength),a_temp)]
            for k,o,a in pairs: outs[k].append(o); amean[k].append(float(a.mean()))
        for k,o in outs.items():
            md=metrics(o,gts); seq_rows.append(dict(sequence=seq,method=k,motion=mot,alpha_mean=float(np.mean(amean[k])) if amean[k] else 0.0,num_frames=len(names),**md))
        by={r['method']:r for r in seq_rows if r['sequence']==seq}
        print(f"  Basic={by['BasicVSR']['psnr']:.4f}, Full={by['FUGR-full-no-temporal']['psnr']:.4f}, Const={by['FUGR-constant-alpha']['psnr']:.4f}, time={time.time()-t0:.1f}s",flush=True)
    fields=['sequence','method','motion','alpha_mean','num_frames','psnr','ssim','laplacian_sharpness','tde']; write_csv(out/'mask_ablation_per_sequence.csv',seq_rows,fields)
    summary=[]
    for m in sorted(set(r['method'] for r in seq_rows)):
        sub=[r for r in seq_rows if r['method']==m]
        summary.append(dict(method=m,psnr=np.mean([r['psnr'] for r in sub]),ssim=np.mean([r['ssim'] for r in sub]),laplacian_sharpness=np.mean([r['laplacian_sharpness'] for r in sub]),tde=np.mean([r['tde'] for r in sub]),motion=np.mean([r['motion'] for r in sub]),alpha_mean=np.mean([r['alpha_mean'] for r in sub]),num_sequences=len(sub)))
    summary.sort(key=lambda x:(x['psnr'],x['ssim']),reverse=True); sfields=['method','psnr','ssim','laplacian_sharpness','tde','motion','alpha_mean','num_sequences']; write_csv(out/'mask_ablation_summary.csv',summary,sfields)
    with (out/'mask_ablation_summary.txt').open('w') as f:
        f.write('FUGR mask-component ablation\n'); f.write(f'num_sequences: {len(seqs)}\nframe_mode: {args.frame_mode}\nmax_alpha: {args.max_alpha}\ntau_dis: {args.tau_dis}\ntau_temp: {args.tau_temp}\nhp_sigma: {args.hp_sigma}\ndetail_strength: {args.detail_strength}\ntotal_time_sec: {time.time()-start:.2f}\n\n')
        f.write('method,psnr,ssim,laplacian_sharpness,tde,motion,alpha_mean,num_sequences\n')
        for r in summary: f.write(f"{r['method']},{r['psnr']:.6f},{r['ssim']:.6f},{r['laplacian_sharpness']:.8f},{r['tde']:.8f},{r['motion']:.8f},{r['alpha_mean']:.8f},{r['num_sequences']}\n")
    print('\n===== Mask ablation summary ====='); print((out/'mask_ablation_summary.txt').read_text()); print('Saved to:',out)
if __name__=='__main__': main()
