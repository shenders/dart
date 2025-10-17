import numpy as np
import matplotlib.pyplot as plt

def compute_toroidal_volume(R, Z):
    R = np.array(R)
    Z = np.array(Z)
    area = 0.5 * np.abs(np.dot(R, np.roll(Z, 1)) - np.dot(Z, np.roll(R, 1)))
    R_avg = np.mean(R)
    volume = 2 * np.pi * R_avg * area
    return volume

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
fig.subplots_adjust(hspace=0.09,left=0.1,top=0.95, bottom=0.15,right=0.96)
fs = 12
hfsx,hfsy = np.array([0.32,0.42, 0.42,0.32,0.32]),np.array([-1.25,-1.25,1.25,1.25,-1.25])
corx,cory = np.array([0.42, 0.6,1.38,1.38,0.6,0.42,0.42]),np.array([-1.25,-1.25,-0.6,0.6,1.25,1.25,-1.25])
lfsx,lfsy = np.array([1.2,2.0,2.0,1.2,0.83,0.75,1.47,1.47,0.75,0.83,1.2]),np.array([-1.02,-1.02,1.02,1.02,1.44,1.35,0.64,-0.64,-1.35,-1.44,-1.02])
solx,soly = np.array([0.6,0.8,0.9,0.75,1.47, 1.47,0.75,0.9,0.8,0.6,1.38,1.38,0.6]),np.array([-1.25,-1.8,-1.9,-1.35,-0.64,0.64,1.35,1.9,1.8,1.25,0.6,-0.6,-1.25])
slfx,slfy = np.array([0.83,1.2,2.0,2.0,0.87,0.83]),np.array([-1.44,-1.02,-1.02,-1.55,-1.58,-1.44])
pfrx,pfry = np.array([0.32,0.32,0.6, 0.8,0.32,0.32]),np.array([1.31,1.25,1.25,1.8,1.31,1.25])
divx,divy = np.array([0.75,0.83,0.87,1.75,1.75,1.35,1.1,0.9,0.77]),np.array([1.35,1.44,1.58,1.55,1.7,2.1,2.1,1.9,1.44])
sdvx,sdvy = np.array([1.75,2.0,2.0,1.35,1.35,1.75,1.75]),np.array([1.55,1.55,2.2,2.2,2.1,1.7,1.55])
ssdx,ssdy = np.array([1.35,1.35,0.32,0.32,1.1,1.35]),np.array([2.1,2.2,2.2,1.31,2.1,2.1])

print("HFS volume:",compute_toroidal_volume(hfsx,hfsy))
print("COR volume:",compute_toroidal_volume(corx,cory))
print("SOL volume:",compute_toroidal_volume(solx,soly))
print("LFS volume:",compute_toroidal_volume(lfsx,lfsy))
print("SLF volume:",compute_toroidal_volume(slfx,slfy))
print("PFR volume:",compute_toroidal_volume(pfrx,pfry))
print("DIV volume:",compute_toroidal_volume(divx,divy))
print("SDV volume:",compute_toroidal_volume(sdvx,sdvy))
print("SSD volume:",compute_toroidal_volume(ssdx,ssdy))
print("Total volume:",compute_toroidal_volume(hfsx,hfsy)+\
        compute_toroidal_volume(lfsx,lfsy) + \
        compute_toroidal_volume(corx,cory) + \
        compute_toroidal_volume(solx,soly) + \
        compute_toroidal_volume(slfx,slfy)*2 + \
        compute_toroidal_volume(pfrx,pfry) + \
        compute_toroidal_volume(divx,divy)*2 + \
        compute_toroidal_volume(ssdx,ssdy)*2 + \
        compute_toroidal_volume(sdvx,sdvy)*2)

axes.fill(hfsx,hfsy,label='HFS',alpha=0.6,edgecolor='black')
axes.fill(corx,cory,label='COR',alpha=0.6,edgecolor='black')
axes.fill(solx,soly,label='SOL',alpha=0.6,edgecolor='black')
axes.fill(pfrx,-pfry,label='LPFR',alpha=0.6,edgecolor='black')
axes.fill(pfrx,pfry,label='UPFR',alpha=0.6,edgecolor='black')
axes.fill(lfsx,lfsy,label='LFS',alpha=0.6,edgecolor='black')
axes.fill(slfx,slfy,label='LFS-lower',alpha=0.6,edgecolor='black')
axes.fill(slfx,-slfy,label='LFS-upper',alpha=0.6,edgecolor='black')
axes.fill(divx,-divy,label='Lower div',alpha=0.6,edgecolor='black')
axes.fill(divx,divy,label='Upper div',alpha=0.6,edgecolor='black')
axes.fill(sdvx,-sdvy,alpha=0.6,label='Lower sub-div',edgecolor='black')
axes.fill(sdvx,sdvy,alpha=0.6,label='Upper sub-div',edgecolor='black')
axes.fill(ssdx,ssdy,alpha=0.6,label='Upper sub-sub-div',edgecolor='black')
axes.fill(ssdx,-ssdy,alpha=0.6,label='Lower sub-sub-div',edgecolor='black')

axes.fill([1.98,2.2,2.2,1.98,1.98],[-0.02,-0.02,0.02,0.02,-0.02],alpha=0.6,edgecolor='black',color='black')
axes.fill([1.98,2.2,2.2,1.98,1.98],[-1.9,-1.9,-1.86,-1.86,-1.9],alpha=0.6,edgecolor='black',color='black')
axes.fill([1.98,2.2,2.2,1.98,1.98],[1.9,1.9,1.86,1.86,1.9],alpha=0.6,edgecolor='black',color='black')
axes.legend(fontsize=fs,loc='upper right')
axes.set_xlabel('R / m',fontsize=fs)
axes.set_ylabel('Z / m',fontsize=fs)
axes.set_xlim([0,5.0])
axes.set_ylim([-2.2,2.2])
axes.set_aspect(aspect=1.0)

axes.text(2.02,0.05,'Main chamber\nFIG',fontsize=fs-1)
axes.text(2.02,-1.82,'Lower divertor\nFIG',fontsize=fs-1)
axes.text(2.02,1.6,'Upper divertor\nFIG',fontsize=fs-1)

plt.tight_layout()
plt.savefig("fig1.png", dpi=300,transparent=True)
plt.savefig("fig1.eps", dpi=300,transparent=True)
plt.savefig("fig1.pdf", dpi=300,transparent=True)
plt.show()
       
