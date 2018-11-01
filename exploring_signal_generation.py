import numpy as np
from dipy.data import get_sphere
# from utils import plotODF, make2D, sphPDF, sphPDF_sym, h_int, h_ext, h_all, h_gs, e_int, e_ext, e_all
from utils import plotODF, sphPDF_sym, h_gs, e_all
# from dipy.core.sphere import HemiSphere
from dipy.core.sphere import Sphere


dpath = '/home/raid2/paquette/work/SMT_gradnonlin/data/'
bvecs = np.genfromtxt(dpath+'bvecs_b10.txt')
if bvecs.shape[1] != 3:
	bvecs = bvecs.T

bvals = np.genfromtxt(dpath+'bvals_b10.txt')

# remove b0
bvecs = bvecs[bvals>10]
bvals = bvals[bvals>10]

# sym
bvecs = np.concatenate((bvecs,-bvecs), axis=0)
bvals = np.concatenate((bvals, bvals), axis=0)


# sample spherical distribution
sphere = get_sphere('repulsion724').subdivide()
mu1 = np.array([0,0,1])
k1 = 3
# d1 = sphPDF(k1, mu1, sphere.vertices)
# d2 = sphPDF(k1, mu1, -sphere.vertices)
# dd1 = (d1+d2)/2.
# dd1 = dd1/dd1.sum()
dd1 = sphPDF_sym(k1, mu1, sphere.vertices, True)
mu2 = np.array([0,1,0])
k2 = 4
# d1 = sphPDF(k2, mu2, sphere.vertices)
# d2 = sphPDF(k2, mu2, -sphere.vertices)
# dd2 = (d1+d2)/2.
# dd2 = dd2/dd2.sum()
dd2 = sphPDF_sym(k2, mu2, sphere.vertices, True)
v1 = 0.5
pdf = v1*dd1+(1-v1)*dd2
pdf = pdf/pdf.sum()



lam = 1.7
v_int = 0.6

signal = h_gs((bvals*1e-3), bvecs, sphere.vertices, pdf, lam, v_int)

exp_smt = signal.mean()
gt_smt = e_all((bvals*1e-3).mean(), lam, v_int)


print(exp_smt, gt_smt)

# plotODF(np.concatenate((signal,signal)), HemiSphere(xyz=bvecs).mirror())
plotODF(signal, Sphere(xyz=bvecs))


