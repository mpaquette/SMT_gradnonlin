import numpy as np
import scipy.stats as st 
from dipy.data import get_sphere
from dipy.viz import fvtk
from scipy.special import erf

# dists = []
# for mu1 in [np.array([0,0,1])]:
# 	for k1 in [1, 2, 4, 16]:
# 		# comp1 (symmetric)
# 		d1 = sphPDF(k1, mu1, sphere.vertices)
# 		d2 = sphPDF(k1, mu1, -sphere.vertices)
# 		dd1 = (d1+d2)/2.
# 		dd1 = dd1/dd1.sum()
# 		for mu2 in [np.array([1,0,0]), np.array([0,1,0])]:
# 			for k2 in [1, 2, 4, 16]:
# 				# comp2 (symmetric)
# 				d1 = sphPDF(k2, mu2, sphere.vertices)
# 				d2 = sphPDF(k2, mu2, -sphere.vertices)
# 				dd2 = (d1+d2)/2.
# 				dd2 = dd2/dd2.sum()
# 				for v1 in [0.25, 0.5, 0.75]:
# 					dists.append(v1*dd1+(1-v1)*dd2)

def sphPDF(k, mu, direc):
	# Generate the PDF for a Von-Mises Fisher distribution p=3
	# at locations direc for concentration k and mean orientation mu
	C3 = k / (2*np.pi*(np.exp(k)-np.exp(-k)))
	tmp = np.exp(k*np.dot(direc,mu[:,None])).squeeze()
	return C3*tmp



def h_int(b, g, w, lam):
	# single g
	return np.exp(-b*(np.dot(w,g[:,None])**2)*lam).squeeze()

def h_ext(b, g, w, lam_para, lam_perp):
	# single g
	para = np.exp(-b*(np.dot(w,g[:,None])**2)*lam_para).squeeze()
	perp = np.exp(-b*(1-np.dot(w,g[:,None])**2)*lam_perp).squeeze()
	return para*perp


def h_all(b, g, w, lam, v_int):
	# Kaden model with lam_int_para == lam_ext_para and lam_ext_perp tortuosity limit
	# single g
	lam_perp = (1-v_int)*lam
	hint = h_int(b, g, w, lam)
	hext = h_ext(b, g, w, lam, lam_perp)
	return v_int*hint + (1-v_int)*hext


def h_gs(bs, gs, w, ODF, lam, v_int):
	# loop over gradient directions
	sig = np.zeros(gs.shape[0])
	for i in range(gs.shape[0]):
		b = bs[i]
		g = gs[i]
		s = h_all(b, g, w, lam, v_int)
		sig[i] = np.sum(s*ODF)
	return sig


def e_int(b, lam):
	return np.sqrt(np.pi)*erf(np.sqrt(b*lam)) / (2*np.sqrt(b*lam))

def e_ext(b, lam_int, lam_ext):
	return np.exp(-b*lam_ext)*(np.sqrt(np.pi)*erf(np.sqrt(b*(lam_int-lam_ext))) / (2*np.sqrt(b*(lam_int-lam_ext))))
	
def e_all(b, lam, v_int):
	lam_perp = (1-v_int)*lam
	eint = e_int(b, lam)
	eext = e_ext(b, lam, lam_perp)
	return v_int*eint + (1-v_int)*eext



dpath = '/home/raid2/paquette/work/SMT_gradnonlin/data/'
bvecs = np.genfromtxt(dpath+'bvecs_b10.txt')
if bvecs.shape[1] != 3:
	bvecs = bvecs.T

bvals = np.genfromtxt(dpath+'bvals_b10.txt')

# remove b0
bvecs = bvecs[bvals>10]
bvals = bvals[bvals>10]


# sample spherical distribution
sphere = get_sphere('repulsion724').subdivide()
mu1 = np.array([0,0,1])
k1 = 3
d1 = sphPDF(k1, mu1, sphere.vertices)
d2 = sphPDF(k1, mu1, -sphere.vertices)
dd1 = (d1+d2)/2.
dd1 = dd1/dd1.sum()
mu2 = np.array([0,1,0])
k2 = 4
d1 = sphPDF(k2, mu2, sphere.vertices)
d2 = sphPDF(k2, mu2, -sphere.vertices)
dd2 = (d1+d2)/2.
dd2 = dd2/dd2.sum()
v1 = 0.5
pdf = v1*dd1+(1-v1)*dd2
pdf = pdf/pdf.sum()



lam = 1.7
v_int = 0.6

signal = h_gs((bvals*1e-3), bvecs, sphere.vertices, pdf, lam, v_int)

exp_smt = signal.mean()
gt_smt = e_all((bvals*1e-3).mean(), lam, v_int)







