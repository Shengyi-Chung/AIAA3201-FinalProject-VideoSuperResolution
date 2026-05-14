#!/usr/bin/env python3
import argparse, sys
from pathlib import Path
import cv2, numpy as np, torch
PART2_DIR=Path(__file__).resolve().parents[1]/'Part2'
if not PART2_DIR.exists(): PART2_DIR=Path.cwd().parents[0]/'Part2'
sys.path.insert(0,str(PART2_DIR))
from model_basicvsr import BasicVSR

def read_rgb(p):
    img=cv2.imread(str(p),cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(p)
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0

def norm01(x,eps=1e-8):
    lo,hi=np.percentile(x,2),np.percentile(x,98); return np.clip((x-lo)/(hi-lo+eps),0,1)
def gray(x): return cv2.cvtColor(np.clip(x*255,0,255).astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
def hp(x,s): return x-cv2.GaussianBlur(x,(0,0),s)
def texmap(b):
    g=gray(b); gx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)
    return norm01(cv2.GaussianBlur(np.sqrt(gx*gx+gy*gy),(0,0),1.2))
def dismap(b,g): return cv2.GaussianBlur(np.mean(np.abs(b-g),axis=2),(0,0),1.5)
def alpha_full(b,g,max_alpha,tau): return np.clip(cv2.GaussianBlur(max_alpha*texmap(b)*np.exp(-dismap(b,g)/tau),(0,0),1.0),0,max_alpha)
def fugr(b,g,a,sig,strength): return np.clip(b+strength*a[...,None]*(hp(g,sig)-hp(b,sig)),0,1)
def rgb(b,g,a): return np.clip((1-a[...,None])*b+a[...,None]*g,0,1)
def colorize(x):
    c=cv2.applyColorMap((norm01(x)*255).astype(np.uint8),cv2.COLORMAP_JET)
    return cv2.cvtColor(c,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
def err(x,gt): return colorize(np.mean(np.abs(x-gt),axis=2))
def resize_h(img,h=240):
    ih,iw=img.shape[:2]; return cv2.resize(img,(int(iw*h/ih),h),interpolation=cv2.INTER_AREA)
def panel(path,imgs,titles):
    imgs=[resize_h(i) for i in imgs]; th=34; w=sum(i.shape[1] for i in imgs); can=np.ones((240+th,w,3),np.uint8)*255; x=0
    for im,t in zip(imgs,titles):
        ww=im.shape[1]; can[th:,x:x+ww]=np.clip(im*255,0,255).round().astype(np.uint8); cv2.putText(can,t,(x+8,24),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,0),2,cv2.LINE_AA); x+=ww
    path.parent.mkdir(parents=True,exist_ok=True); cv2.imwrite(str(path),cv2.cvtColor(can,cv2.COLOR_RGB2BGR))
def load_model(p,dev,spynet):
    m=BasicVSR(spynet_path=spynet).to(dev); ck=torch.load(str(p),map_location='cpu'); st=ck['model_state_dict'] if isinstance(ck,dict) and 'model_state_dict' in ck else ck; m.load_state_dict(st,strict=True); m.eval(); return m
def load_lr(d,names): return torch.stack([torch.from_numpy(read_rgb(d/n).transpose(2,0,1)).float() for n in names],0).unsqueeze(0)
@torch.no_grad()
def infer(m,x,dev,amp=False):
    x=x.to(dev)
    if dev.type=='cuda' and amp:
        with torch.cuda.amp.autocast(): y=m(x)
    else: y=m(x)
    y=torch.clamp(y[0].detach().cpu(),0,1).permute(0,2,3,1).numpy().astype(np.float32)
    return [y[i] for i in range(y.shape[0])]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--val_root',default='/home/schung760/shared_data/project1/val'); ap.add_argument('--basic_ckpt',default='/home/schung760/my_storage_1T/AIAA3201-FinalProject-VideoSuperResolution/Part2/weights/basicvsr_stage1.pth'); ap.add_argument('--gan_ckpt',default='/home/schung760/my_storage_1T/AIAA3201-FinalProject-VideoSuperResolution/Part2/weights/basicvsr_gan.pth'); ap.add_argument('--spynet_path',default='/home/schung760/my_storage_1T/AIAA3201-FinalProject-VideoSuperResolution/Part2/weights/spynet.pth'); ap.add_argument('--out_dir',required=True); ap.add_argument('--seqs',nargs='+',default=['028','003','020']); ap.add_argument('--frame_indices',nargs='+',type=int,default=[25,50,75]); ap.add_argument('--max_alpha',type=float,default=0.25); ap.add_argument('--tau_dis',type=float,default=0.08); ap.add_argument('--hp_sigma',type=float,default=1.6); ap.add_argument('--detail_strength',type=float,default=1.2); ap.add_argument('--amp',action='store_true'); args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True); val=Path(args.val_root); hr=val/'val_sharp'; lrroot=val/'val_sharp_bicubic'/'X4'; dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('Device:',dev)
    bm=load_model(Path(args.basic_ckpt),dev,args.spynet_path); gm=load_model(Path(args.gan_ckpt),dev,args.spynet_path)
    for seq in args.seqs:
        hrd=hr/seq; lrd=lrroot/seq; names=sorted([p.name for p in hrd.glob('*.png') if (lrd/p.name).exists()]); print('Seq',seq,len(names),'frames')
        b=infer(bm,load_lr(lrd,names),dev,args.amp); g=infer(gm,load_lr(lrd,names),dev,args.amp)
        for idx in args.frame_indices:
            idx=min(max(idx,0),len(names)-1); name=names[idx]; lr=read_rgb(lrd/name); gt=read_rgb(hrd/name); a=alpha_full(b[idx],g[idx],args.max_alpha,args.tau_dis); f=fugr(b[idx],g[idx],a,args.hp_sigma,args.detail_strength); r=rgb(b[idx],g[idx],a)
            panel(out/f'qual_seq{seq}_frame{idx:03d}_{Path(name).stem}.png',[lr,b[idx],g[idx],r,f,gt,err(b[idx],gt),err(f,gt),colorize(a)],['LR','Basic','VSRGAN','RGB','FUGR','GT','Basic Err','FUGR Err','Alpha'])
    print('Saved panels to',out)
if __name__=='__main__': main()
