import numpy as np
import scipy.stats as st 
from dipy.data import get_sphere
from dipy.viz import fvtk


def plotODF(ODF, sphere):
	r = fvtk.ren()
	sfu = fvtk.sphere_funcs(ODF, sphere, scale=2.2, norm=True)
	# sfu.RotateX(90)
	# sfu.RotateY(180)
	fvtk.add(r, sfu)
	# outname = 'screenshot_signal_r.png'
	# fvtk.record(r, n_frames=1, out_path = outname, size=(3000, 1500), magnification = 2)
	fvtk.show(r)



def make2D(ODF_list):
	n = len(ODF_list)
	sn = int(np.ceil(np.sqrt(n)))
	grid = np.zeros((sn,sn,len(ODF_list[0])))
	for i in range(n):
		ix, iy = divmod(i, sn)
		grid[ix, iy] = ODF_list[i]
	return grid


def sphPDF(k, mu, direc):
	# Generate the PDF for a Von-Mises Fisher distribution p=3
	# at locations direc for concentration k and mean orientation mu
	C3 = k / (2*np.pi*(np.exp(k)-np.exp(-k)))
	tmp = np.exp(k*np.dot(direc,mu[:,None])).squeeze()
	return C3*tmp


# # sphere = get_sphere('repulsion724')
# sphere = get_sphere('repulsion724').subdivide()

# dists = []
# # for mu in [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]:
# for mu in [np.array([0,0,1])]:
# 	# for k in [1.,4.,10.]:
# 	for k in np.linspace(0.1,20,49):
# 		# make symmetric
# 		d1 = sphPDF(k, mu, sphere.vertices)
# 		d2 = sphPDF(k, mu, -sphere.vertices)
# 		dd = (d1+d2)/2.
# 		dists.append(dd)
# ODF = make2D(dists)
# plotODF(ODF, sphere)


# 2-mixture
# sphere = get_sphere('repulsion724')
sphere = get_sphere('repulsion724').subdivide()


dists = []
for mu1 in [np.array([0,0,1])]:
	for k1 in [1, 2, 4, 16]:
		# comp1 (symmetric)
		d1 = sphPDF(k1, mu1, sphere.vertices)
		d2 = sphPDF(k1, mu1, -sphere.vertices)
		dd1 = (d1+d2)/2.
		dd1 = dd1/dd1.sum()
		for mu2 in [np.array([1,0,0]), np.array([0,1,0])]:
			for k2 in [1, 2, 4, 16]:
				# comp2 (symmetric)
				d1 = sphPDF(k2, mu2, sphere.vertices)
				d2 = sphPDF(k2, mu2, -sphere.vertices)
				dd2 = (d1+d2)/2.
				dd2 = dd2/dd2.sum()
				for v1 in [0.25, 0.5, 0.75]:
					dists.append(v1*dd1+(1-v1)*dd2)
					# print(k1, k2, v1)
					# plotODF(v1*dd1+(1-v1)*dd2, sphere)
ODF = make2D(dists)
plotODF(ODF, sphere)








