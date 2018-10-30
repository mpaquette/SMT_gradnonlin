import numpy as np
from dipy.viz import fvtk
from scipy.special import erf

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

