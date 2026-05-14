#!/usr/bin/env python3
import argparse,csv,sys,time
from pathlib import Path
import cv2,numpy as np,torch
PART2_DIR=Path(__file__).resolve().parents[1]/'Part2'
if not PART2_DIR.exists(): PART2_DIR=Path.cwd().parents[0]/'Part2'
sys.path.insert(0,str(PART2_DIR))
from model_basicvsr import BasicVSR

def read_rgb(p):
    img=cv2.imread(str(p),cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(p)
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
def gray(x): return cv2.cvtColor(np.clip(x*255,0,255).astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
def norm01(x,eps=1e-8):
    lo,hi=np.percentile(x,2),np.percentile(x,98); return np.clip((x-lo)/(hi-lo+eps),0,1)
def hp(x,s): return x-cv2.GaussianBlur(x,(0,0),s)
def texmap(b):
    g=gray(b); gx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3); return norm01(cv2.GaussianBlur(np.sqrt(gx*gx+gy*gy),(0,0),1.2))
def dismap(b,g): return cv2.GaussianBlur(np.mean(np.abs(b-g),axis=2),(0,0),1.5)
def post(basics,gans,a,tau,sig,strength):
    outs=[]
    for b,g in zip(basics,gans):
        al=np.clip(cv2.GaussianBlur(a*texmap(b)*np.exp(-dismap(b,g)/tau),(0,0),1.0),0,a); outs.append(np.clip(b+strength*al[...,None]*(hp(g,sig)-hp(b,sig)),0,1))
    return outs
def load_model(p,dev,spynet):
    m=BasicVSR(spynet_path=spynet).to(dev); ck=torch.load(str(p),map_location='cpu'); st=ck['model_state_dict'] if isinstance(ck,dict) and 'model_state_dict' in ck else ck; m.load_state_dict(st,strict=True); m.eval(); return m
def load_lr(d,names): return torch.stack([torch.from_numpy(read_rgb(d/n).transpose(2,0,1)).float() for n in names],0).unsqueeze(0)
@torch.no_grad()
def infer(m,x,dev,amp=False):
    x=x.to(dev)
    if dev.type=='cuda' and amp:
        with torch.cuda.amp.autocast(): y=m(x)
    else: y=m(x)
    y=torch.clamp(y[0].detach().cpu(),0,1).permute(0,2,3,1).numpy().astype(np.float32); return [y[i] for i in range(y.shape[0])]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--val_root',default='/home/schung760/shared_data/project1/val'); ap.add_argument('--basic_ckpt',default='/home/schung760/my_storage_1T/AIAA3201-FinalProject-VideoSuperResolution/Part2/weights/basicvsr_stage1.pth'); ap.add_argument('--gan_ckpt',default='/home/schung760/my_storage_1T/AIAA3201-FinalProject-VideoSuperResolution/Part2/weights/basicvsr_gan.pth'); ap.add_argument('--spynet_path',default='/home/schung760/my_storage_1T/AIAA3201-FinalProject-VideoSuperResolution/Part2/weights/spynet.pth'); ap.add_argument('--out_csv',required=True); ap.add_argument('--seq_limit',type=int,default=5); ap.add_argument('--max_alpha',type=float,default=0.25); ap.add_argument('--tau_dis',type=float,default=0.08); ap.add_argument('--hp_sigma',type=float,default=1.6); ap.add_argument('--detail_strength',type=float,default=1.2); ap.add_argument('--amp',action='store_true'); args=ap.parse_args()
    val=Path(args.val_root); hr=val/'val_sharp'; lrroot=val/'val_sharp_bicubic'/'X4'; seqs=sorted([p.name for p in hr.iterdir() if p.is_dir() and (lrroot/p.name).is_dir()])[:args.seq_limit]
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); bm=load_model(Path(args.basic_ckpt),dev,args.spynet_path); gm=load_model(Path(args.gan_ckpt),dev,args.spynet_path); rows=[]
    for seq in seqs:
        lrd=lrroot/seq; names=sorted([p.name for p in lrd.glob('*.png')]); print('Benchmark',seq,len(names),'frames',flush=True)
        t=time.time(); b=infer(bm,load_lr(lrd,names),dev,args.amp); tb=time.time()-t
        t=time.time(); g=infer(gm,load_lr(lrd,names),dev,args.amp); tg=time.time()-t
        t=time.time(); _=post(b,g,args.max_alpha,args.tau_dis,args.hp_sigma,args.detail_strength); tp=time.time()-t
        rows.append(dict(sequence=seq,num_frames=len(names),basic_time_sec=tb,gan_time_sec=tg,fugr_post_time_sec=tp,basic_sec_per_frame=tb/len(names),gan_sec_per_frame=tg/len(names),fugr_sec_per_frame=tp/len(names)))
    out=Path(args.out_csv); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    ab=np.mean([r['basic_sec_per_frame'] for r in rows]); ag=np.mean([r['gan_sec_per_frame'] for r in rows]); ap=np.mean([r['fugr_sec_per_frame'] for r in rows])
    print('Saved:',out); print(f'Average Basic sec/frame: {ab:.6f}'); print(f'Average GAN sec/frame: {ag:.6f}'); print(f'Average FUGR post sec/frame: {ap:.6f}'); print(f'Overhead vs Basic: {100*ap/ab:.2f}%'); print(f'Overhead vs two-model inference: {100*ap/(ab+ag):.2f}%')
if __name__=='__main__': main()
