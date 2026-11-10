"""
Animated lily: the flower outline is traced first, petals fill one by one,
then the stem, leaves and stamens appear.

Install: pip install numpy matplotlib pillow
Run:     python animated_lily.py
Output:  lily_bloom.gif
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def norm(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) or 1)

def petal(center, direction, length, width, nu=40, nv=21):
    d = norm(direction)
    s = norm(np.cross(d, [0, 0, 1]))
    if np.linalg.norm(s) < 1e-6:
        s = np.array([1., 0., 0.])
    n = norm(np.cross(s, d))
    u, v = np.linspace(0, 1, nu), np.linspace(-1, 1, nv)
    U, V = np.meshgrid(u, v)
    w = width * np.sin(np.pi * U**0.82)**0.72 * V
    h = 0.20*np.sin(np.pi*U) + 0.18*U**2 + 0.07*(1-V**2)*np.sin(np.pi*U)
    P = (np.asarray(center)[:,None,None]
         + d[:,None,None]*(length*U)[None,:,:]
         + s[:,None,None]*w[None,:,:]
         + n[:,None,None]*h[None,:,:])
    # Trace the two edges and the rounded tip.
    outline = [
        np.column_stack([P[:,0,:].T]),
        np.column_stack([P[:,-1,:].T]),
        np.column_stack([P[:,-1,::-1].T]),
    ]
    return P, outline

fig = plt.figure(figsize=(7,8), facecolor="#090313")
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor("#090313")
ax.set_axis_off()
ax.set_xlim(-2,2); ax.set_ylim(-2,2); ax.set_zlim(-2.7,3)
ax.set_box_aspect((1,1,1.45))
ax.view_init(elev=12, azim=-62)

head = np.array([0.,0.,1.18])
colors = ["#d88aff","#c879f5","#e4a0ff","#c77bf0","#e8b0ff","#d58cff"]
petals = []
for i in range(6):
    a = 2*np.pi*i/6 + .12
    petals.append(petal(head,[np.cos(a),np.sin(a),.16],1.62,.48))
for i in range(6):
    a = 2*np.pi*i/6 + np.pi/6
    petals.append(petal(head+[0,0,.035],[np.cos(a),np.sin(a),.05],1.35,.40))

# Each outline segment is drawn progressively.
lines = []
for P, edges in petals:
    for edge in edges:
        ln, = ax.plot([],[],[],color="#f3c5ff",lw=1.35)
        lines.append((ln,edge))
surfaces = [None]*len(petals)

stem = np.array([[0,0,-2.55],[.04,0,-1.8],[.02,.02,-1],[-.04,0,-.15],[0,0,1.15]])
stem_line, = ax.plot([],[],[],color="#278657",lw=5)

def leaf_mesh(start, direction, length=1.3, width=.18):
    d=norm(direction); s=norm(np.cross(d,[0,0,1]))
    if np.linalg.norm(s)<1e-6: s=np.array([1.,0.,0.])
    n=norm(np.cross(s,d))
    u,v=np.linspace(0,1,30),np.linspace(-1,1,15)
    U,V=np.meshgrid(u,v)
    W=width*np.sin(np.pi*U)**.8*V
    H=.12*np.sin(np.pi*U)+.04*(1-V**2)
    return (np.asarray(start)[:,None,None]+d[:,None,None]*(length*U)[None,:,:]
            +s[:,None,None]*W[None,:,:]+n[:,None,None]*H[None,:,:])

leaf_data=[
    ([-.01,0,-1.8],[-1,-.1,.24],1.55),
    ([.02,0,-1.25],[.95,.05,.18],1.35),
    ([0,0,-.72],[-.88,.08,.3],1.3),
    ([-.03,0,-.18],[.92,-.12,.24],1.15)]
leaf_meshes=[leaf_mesh(*x) for x in leaf_data]
leaf_surfaces=[None]*len(leaf_meshes)

filaments=[]
for i in range(6):
    a=2*np.pi*i/6+.3
    p0=head+[0,0,.1]
    p2=head+[.43*np.cos(a),.43*np.sin(a),.95]
    p1=(p0+p2)/2+[0,0,.12]
    pts=np.array([p0,p1,p2])
    ln,=ax.plot([],[],[],color="#f4d5ff",lw=2)
    dot=ax.scatter([],[],[],s=38,color="#f5bd45",depthshade=False)
    filaments.append((ln,dot,pts))
pistil,=ax.plot([],[],[],color="#b9e28b",lw=3)
pistil_dot=ax.scatter([],[],[],s=60,color="#c8ed91",depthshade=False)

# Timing: outline -> petal color -> stem -> leaves -> flower centre.
outline_step=4
outline_end=len(lines)*outline_step
fill_step=3
fill_end=outline_end+len(petals)*fill_step
stem_start=fill_end
leaf_start=stem_start+22
detail_start=leaf_start+len(leaf_meshes)*10
total=detail_start+22+12

def update(frame):
    for i,(ln,pts) in enumerate(lines):
        p=np.clip((frame-i*outline_step)/outline_step,0,1)
        k=max(0,min(len(pts),int(p*len(pts))))
        q=pts[:k]
        ln.set_data(q[:,0],q[:,1]); ln.set_3d_properties(q[:,2])

    if frame>=outline_end:
        count=min(len(petals),(frame-outline_end)//fill_step+1)
        for i in range(int(count)):
            if surfaces[i] is None:
                P,_=petals[i]
                surfaces[i]=ax.plot_surface(P[0],P[1],P[2],
                    color=colors[i%len(colors)],alpha=.95,
                    linewidth=0,shade=True,antialiased=True)

    if frame>=stem_start:
        p=np.clip((frame-stem_start)/22,0,1)
        k=max(2,min(len(stem),int(2+p*(len(stem)-2))))
        q=stem[:k]
        stem_line.set_data(q[:,0],q[:,1]); stem_line.set_3d_properties(q[:,2])

    if frame>=leaf_start:
        count=min(len(leaf_meshes),(frame-leaf_start)//10+1)
        for i in range(int(count)):
            if leaf_surfaces[i] is None:
                P=leaf_meshes[i]
                leaf_surfaces[i]=ax.plot_surface(P[0],P[1],P[2],
                    color="#238b58",linewidth=0,shade=True,antialiased=True)

    if frame>=detail_start:
        p=np.clip((frame-detail_start)/22,0,1)
        for ln,dot,pts in filaments:
            k=max(1,min(3,int(1+2*p)))
            q=pts[:k]
            ln.set_data(q[:,0],q[:,1]); ln.set_3d_properties(q[:,2])
            if p>.75:
                dot._offsets3d=([pts[-1,0]],[pts[-1,1]],[pts[-1,2]])
        top=head+[0,0,1.12*p]
        pistil.set_data([head[0],top[0]],[head[1],top[1]])
        pistil.set_3d_properties([head[2],top[2]])
        if p>.85:
            pistil_dot._offsets3d=([top[0]],[top[1]],[top[2]])
    return []

ani=FuncAnimation(fig,update,frames=total,interval=45,blit=False,repeat=True)
print("Rendering lily_bloom.gif (this may take a little while)...")
ani.save("lily_bloom.gif",writer=PillowWriter(fps=22))
print("Saved: lily_bloom.gif")
plt.show()
