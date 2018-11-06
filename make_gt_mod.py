import numpy as np
from dipy.data import get_sphere
import pylab as pl
# from utils import plotODF, make2D, sphPDF, sphPDF_sym, h_int, h_ext, h_all, h_gs, e_int, e_ext, e_all
from utils import plotODF, make2D, sphPDF_sym, h_gs, e_all
import seaborn as sns

res_folder = '/NOBACKUP2/paquette/SMT_gradnonlin_res/'

bbSS = [1, 2, 3, 5, 10, 11, 12, 12.5, 15, 20]

for bb in bbSS:

	dpath = '/home/raid2/paquette/work/SMT_gradnonlin/data/'
	bvecs = np.genfromtxt(dpath+'bvecs_b10.txt')
	if bvecs.shape[1] != 3:
		bvecs = bvecs.T

	bvals = np.genfromtxt(dpath+'bvals_b10.txt')

	# bb = 1

	# remove b0
	bvecs = bvecs[bvals>10]
	bvals = bvals[bvals>10]
	bvals = bvals*(bb/10.)

	# sym
	bvecs = np.concatenate((bvecs,-bvecs), axis=0)
	bvals = np.concatenate((bvals, bvals), axis=0)

	# load grad non lin tensor centroid
	centroid = np.load(dpath + 'dev_parc_centroid_n_100.npy')
	centroid = centroid.reshape(centroid.shape[0],3,3)





	# distorting btable
	new_bvecs = []
	new_bvals = []

	for ic in range(centroid.shape[0]):
		tensor = centroid[ic]
		new_bvec = np.dot(tensor, bvecs.T).T
		new_norm = np.linalg.norm(new_bvec, axis=1)
		new_bvec = new_bvec/new_norm[:, None]
		new_bval = bvals*(new_norm**2)
		new_bvecs.append(new_bvec)
		new_bvals.append(new_bval)


	bvals_mean = np.array([bv.mean() for bv in new_bvals])



	lams = [0.5, 1, 1.5, 2, 2.5]
	v_ints = [0.1, 0.3, 0.5, 0.7, 0.9]
	micro = [(l,v) for l in lams for v in v_ints]


	gt_mean = np.zeros((len(bvals_mean), len(micro)))



	for i_m, mic in enumerate(micro):
		lam, v_int = mic
		for i_b, bv in enumerate(bvals_mean):
			# ground truth spherical mean for this microstructure configuration
			gt_mean[i_b, i_m] = e_all((bv*1e-3), lam, v_int)


	# np.save(res_folder + 'gt_mean_mod_b{}_1.npy'.format(bb), gt_mean)




