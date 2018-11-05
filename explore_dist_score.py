import numpy as np
from dipy.data import get_sphere
import pylab as pl
# from utils import plotODF, make2D, sphPDF, sphPDF_sym, h_int, h_ext, h_all, h_gs, e_int, e_ext, e_all
from utils import plotODF, make2D, sphPDF_sym, h_gs, e_all
import seaborn as sns

res_folder = '/NOBACKUP2/paquette/SMT_gradnonlin_res/'

dpath = '/home/raid2/paquette/work/SMT_gradnonlin/data/'
bvecs = np.genfromtxt(dpath+'bvecs_b10.txt')
if bvecs.shape[1] != 3:
	bvecs = bvecs.T

bvals = np.genfromtxt(dpath+'bvals_b10.txt')

bb = 2

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

_,s,_ = np.linalg.svd(centroid)
dist_score = np.linalg.norm(s-np.ones(3), axis=1)


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

np.save(res_folder + 'bvecs_b{}_1.npy'.format(bb), np.array(new_bvecs))
np.save(res_folder + 'bvals_b{}_1.npy'.format(bb), np.array(new_bvals))




new_bvals_mean = np.array([b.mean() for b in new_bvals])
new_bvals_std = np.array([b.std() for b in new_bvals])



idxkey = np.argsort(dist_score)
AA = dist_score[idxkey]
BB = new_bvals_std[idxkey]

pl.figure()
# pl.plot(AA, BB, '.')
pl.scatter(AA, BB)
pl.title('Distorsion score vs bvals standard deviation, R^2 = {:.2f}'.format(np.corrcoef(AA, BB)[0,1]))
pl.show()

# scipy.stats.kendalltau(AA,BB).correlation
# scipy.stats.spearmanr(AA,BB).correlation







