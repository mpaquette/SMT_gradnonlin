import numpy as np
import scipy.stats as st 
from dipy.data import get_sphere
from dipy.viz import fvtk
from scipy.special import erf
import pylab as pl

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

def plotODF(ODF, sphere):
	r = fvtk.ren()
	sfu = fvtk.sphere_funcs(ODF, sphere, scale=2.2, norm=True)
	sfu.RotateY(90)
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


dpath = '/home/raid2/paquette/work/SMT_gradnonlin/data/'
bvecs = np.genfromtxt(dpath+'bvecs_b10.txt')
if bvecs.shape[1] != 3:
	bvecs = bvecs.T

bvals = np.genfromtxt(dpath+'bvals_b10.txt')

# remove b0
bvecs = bvecs[bvals>10]
bvals = bvals[bvals>10]


# sample spherical distribution
ODFS_low = []
sphere_low = get_sphere('repulsion724')
for mu1 in [np.array([0,0,1])]:
	for k1 in [1, 2, 4, 16]:
		# comp1 (symmetric)
		d1 = sphPDF(k1, mu1, sphere_low.vertices)
		d2 = sphPDF(k1, mu1, -sphere_low.vertices)
		dd1 = (d1+d2)/2.
		dd1 = dd1/dd1.sum()
		for mu2 in [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]:
			for k2 in [1, 2, 4, 16]:
				# comp2 (symmetric)
				d1 = sphPDF(k2, mu2, sphere_low.vertices)
				d2 = sphPDF(k2, mu2, -sphere_low.vertices)
				dd2 = (d1+d2)/2.
				dd2 = dd2/dd2.sum()
				for v1 in [0.25, 0.5, 0.75]:
					pdf = v1*dd1+(1-v1)*dd2
					pdf = pdf/pdf.sum()
					ODFS_low.append(pdf)


plotODF(make2D(ODFS_low), sphere_low)



lams = [0.5, 1, 1.5, 2, 2.5]
v_ints = [0.1, 0.3, 0.5, 0.7, 0.9]

gt_mean = []
exp_mean_low = []
exp_sig_low = []

ii=0
for lam in lams:
	for v_int in v_ints:
		print('{} out of {}'.format(ii,len(lams)*len(v_ints)))
		ii+=1
		gt_smt = e_all((bvals*1e-3).mean(), lam, v_int)
		gt_mean.append(gt_smt)
		tmp_mean = []
		tmp_sig = []
		for pdf in ODFS_low:
			signal = h_gs((bvals*1e-3), bvecs, sphere_low.vertices, pdf, lam, v_int)
			exp_smt = signal.mean()
			tmp_mean.append(exp_smt)
			tmp_sig.append(signal)
		exp_mean_low.append(tmp_mean)
		exp_sig_low.append(tmp_sig)


gt_means = np.array(gt_mean)
exp_means_low = np.array(exp_mean_low)

exp_means_low_mean = exp_means_low.mean(axis=1)
exp_means_low_std = exp_means_low.std(axis=1)

pl.figure()
pl.errorbar(range(len(exp_means_low_mean)), exp_means_low_mean, yerr=exp_means_low_std, fmt='.', label='Low order (+/- std)')
pl.plot(range(len(exp_means_low_mean)), gt_means, '.', label = 'GT')
pl.title('Low order sphere discret signal mean for many ODF vs GT')
pl.legend()
# pl.show()

# data.mean over GT +/- data.std over GT
# less than 0.1% std, less than 0.01% bias 
pl.figure()
pl.errorbar(range(len(exp_means_low_mean)), exp_means_low_mean/gt_means, xerr=0, yerr=exp_means_low_std/gt_means, label='Low order (+/- std)')
pl.title('data.mean over GT +/- data.std over GT')
pl.show()












# sample spherical distribution
ODFS_high = []
sphere_high = get_sphere('repulsion724').subdivide()
for mu1 in [np.array([0,0,1])]:
	for k1 in [1, 2, 4, 16]:
		# comp1 (symmetric)
		d1 = sphPDF(k1, mu1, sphere_high.vertices)
		d2 = sphPDF(k1, mu1, -sphere_high.vertices)
		dd1 = (d1+d2)/2.
		dd1 = dd1/dd1.sum()
		for mu2 in [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]:
			for k2 in [1, 2, 4, 16]:
				# comp2 (symmetric)
				d1 = sphPDF(k2, mu2, sphere_high.vertices)
				d2 = sphPDF(k2, mu2, -sphere_high.vertices)
				dd2 = (d1+d2)/2.
				dd2 = dd2/dd2.sum()
				for v1 in [0.25, 0.5, 0.75]:
					pdf = v1*dd1+(1-v1)*dd2
					pdf = pdf/pdf.sum()
					ODFS_high.append(pdf)


plotODF(make2D(ODFS_high), sphere_high)



lams = [0.5, 1, 1.5, 2, 2.5]
v_ints = [0.1, 0.3, 0.5, 0.7, 0.9]

# gt_mean = []
exp_mean_high = []
exp_sig_high = []

ii=0
for lam in lams:
	for v_int in v_ints:
		print('{} out of {}'.format(ii,len(lams)*len(v_ints)))
		ii+=1
		# gt_smt = e_all((bvals*1e-3).mean(), lam, v_int)
		# gt_mean.append(gt_smt)
		tmp_mean = []
		tmp_sig = []
		for pdf in ODFS_high:
			signal = h_gs((bvals*1e-3), bvecs, sphere_high.vertices, pdf, lam, v_int)
			exp_smt = signal.mean()
			tmp_mean.append(exp_smt)
			tmp_sig.append(signal)
		exp_mean_high.append(tmp_mean)
		exp_sig_high.append(tmp_sig)


# gt_means = np.array(gt_mean)
exp_means_high = np.array(exp_mean_high)

exp_means_high_mean = exp_means_high.mean(axis=1)
exp_means_high_std = exp_means_high.std(axis=1)

pl.figure()
pl.errorbar(range(len(exp_means_high_mean)), exp_means_high_mean,  yerr=exp_means_high_std, fmt='.', label='High order (+/- std)')
pl.plot(range(len(exp_means_high_mean)), gt_means, '.', label = 'GT')
pl.title('High order sphere discret signal mean for many ODF vs GT')
pl.legend()
# pl.show()

# data.mean over GT +/- data.std over GT
# less than 0.1% std, less than 0.01% bias 
pl.figure()
pl.errorbar(range(len(exp_means_high_mean)), exp_means_high_mean/gt_means, xerr=0, yerr=exp_means_high_std/gt_means, label='High order (+/- std)')
pl.title('Low order sphere discret signal mean for many ODF vs GT')
pl.show()



