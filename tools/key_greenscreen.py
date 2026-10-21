# Green-screen key for Kling output -> ProRes 4444 + alpha.
# ffmpeg -i test.mp4 -vf scale=1920:1080:flags=lanczos -f rawvideo -pix_fmt rgb24 - \
#  | python3 tools/key_greenscreen.py 1920 1080 \
#  | ffmpeg -f rawvideo -pix_fmt rgba -s 1920x1080 -r 24 -i - \
#    -c:v prores_ks -profile:v 4444 -pix_fmt yuva444p10le -alpha_bits 16 -vendor apl0 out.mov
import sys, numpy as np
from scipy import ndimage
W,H=int(sys.argv[1]),int(sys.argv[2])
LO,HI=25.0,60.0             # green dominance: <=LO opaque, >=HI transparent
WM=(slice(980,H),slice(1620,W))  # KlingAI watermark area (bottom-right)
inp,out=sys.stdin.buffer,sys.stdout.buffer
n=0; maxx=0; wm_hits=0
while True:
    buf=inp.read(W*H*3)
    if len(buf)<W*H*3: break
    f=np.frombuffer(buf,np.uint8).reshape(H,W,3).astype(np.float32)
    r,g,b=f[...,0],f[...,1],f[...,2]
    rb=np.maximum(r,b); d=g-rb
    a=np.clip((HI-d)/(HI-LO),0,1)
    # garbage matte: keep only the character (largest blob + anything big)
    solid=ndimage.binary_closing(a>0.5,iterations=2)
    lab,k=ndimage.label(solid)
    if k:
        sz=ndimage.sum(solid,lab,range(1,k+1))
        keep=np.isin(lab,1+np.where(sz>=0.05*sz.max())[0])
        keep=ndimage.binary_fill_holes(keep)
        keep=ndimage.binary_dilation(keep,iterations=3)
        if keep[WM].any(): wm_hits+=1
        a=np.where(keep,a,0)
        xs=np.where(keep.any(0))[0]; maxx=max(maxx,xs.max())
    a[WM]=0
    a=ndimage.grey_erosion(a,size=5)  # choke 2px: drop green/blue mixed fringe
    a=ndimage.gaussian_filter(a,0.6)  # soften stair-stepping on edges
    # despill: no pixel may be greener than its max(r,b)
    g=np.minimum(g,0.5*(r+b))
    o=np.dstack([r,g,b,a*255]).clip(0,255).astype(np.uint8)
    out.write(o.tobytes()); n+=1
sys.stderr.write(f'frames={n} char_max_x={maxx} watermark_overlap_frames={wm_hits}\n')
