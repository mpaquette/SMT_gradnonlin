import numpy as np
import nibabel as nib 
import sys


dpath = '/home/raid2/paquette/work/SMT_gradnonlin/data/'


dev_X = nib.load(dpath+'test_x.nii.gz').get_data()
dev_Y = nib.load(dpath+'test_y.nii.gz').get_data()
dev_Z = nib.load(dpath+'test_z.nii.gz').get_data()

mask_img = nib.load(dpath+'newmask.nii.gz')
mask = mask_img.get_data()
head = mask_img.header
aff = mask_img.affine

# convert percentage to fraction
dev_X *= 0.01
dev_Y *= 0.01
dev_Z *= 0.01


# make the grad non lin tensor
dev = np.concatenate((dev_X[...,None], dev_Y[...,None], dev_Z[...,None]), axis=4)

U,s,V = np.linalg.svd(dev)
out4 = np.linalg.norm(s-np.ones(3), axis=3)
out4[~mask.astype(np.bool)]=0







pl.figure()
pl.hist(allb, 50, density=True, color=(0.0,0.0,0.99,0.5))
sns.kdeplot(allb, bw=0.015, color='red')
frame1 = pl.gca()
frame1.axes.yaxis.set_ticklabels([])
# pl.title('Distribution of b-value modifier inside full brain')
pl.xlabel('b-value multiplier', size=16)
pl.show()







import pylab as pl
from mpl_toolkits.axes_grid1 import ImageGrid

pl.rcParams.update({'font.size': 12})

# fig = pl.figure(figsize=(9.75, 3))
fig = pl.figure()
grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(1,3),
                 axes_pad=0.15,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="7%",
                 cbar_pad=0.15,
                 )


xx,yy,zz = out4.shape
im = grid[0].imshow(out4[:,:,zz//2], cmap=pl.cm.jet, interpolation='nearest', vmin=out4.min(), vmax=out4.max())
im = grid[1].imshow(out4[:,yy//2,:], cmap=pl.cm.jet, interpolation='nearest', vmin=out4.min(), vmax=out4.max())
im = grid[2].imshow(out4[xx//2,:,:], cmap=pl.cm.jet, interpolation='nearest', vmin=out4.min(), vmax=out4.max())

# im = grid[0].imshow(out4[:,:,zz//2], cmap=pl.cm.jet, interpolation='bilinear', vmin=out4.min(), vmax=out4.max())
# im = grid[1].imshow(out4[:,yy//2,:], cmap=pl.cm.jet, interpolation='bilinear', vmin=out4.min(), vmax=out4.max())
# im = grid[2].imshow(out4[xx//2,:,:], cmap=pl.cm.jet, interpolation='bilinear', vmin=out4.min(), vmax=out4.max())

# im = grid[0].imshow(out4[:,:,zz//2], cmap=pl.cm.jet, interpolation='bicubic', vmin=out4.min(), vmax=out4.max())
# im = grid[1].imshow(out4[:,yy//2,:], cmap=pl.cm.jet, interpolation='bicubic', vmin=out4.min(), vmax=out4.max())
# im = grid[2].imshow(out4[xx//2,:,:], cmap=pl.cm.jet, interpolation='bicubic', vmin=out4.min(), vmax=out4.max())

# im = grid[0].imshow(out4[:,:,zz//2], cmap=pl.cm.jet, interpolation='spline16', vmin=out4.min(), vmax=out4.max())
# im = grid[1].imshow(out4[:,yy//2,:], cmap=pl.cm.jet, interpolation='spline16', vmin=out4.min(), vmax=out4.max())
# im = grid[2].imshow(out4[xx//2,:,:], cmap=pl.cm.jet, interpolation='spline16', vmin=out4.min(), vmax=out4.max())



for ax in grid:
	ax.xaxis.set_visible(False)
	ax.yaxis.set_visible(False)


# Colorbar
grid[2].cax.colorbar(im)
grid[2].cax.toggle_label(True)

pl.show()


