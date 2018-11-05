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

bb = 1

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

# sort centroid by distortion score
_,s,_ = np.linalg.svd(centroid)
dist_score = np.linalg.norm(s-np.ones(3), axis=1)
centroid = centroid[np.argsort(dist_score)]



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



# mke uniform distribution
ODFS = []
sphere = get_sphere('repulsion724')
pdf = np.ones(sphere.vertices.shape[0])
pdf = pdf/pdf.sum()
ODFS.append(pdf)


# plotODF(make2D(ODFS), sphere)






lams = [2]
v_ints = [0.]
micro = [(l,v) for l in lams for v in v_ints]


# #grad_non_lin tensor X #microstructure config X #ODF X #bvec
signal = np.zeros((len(new_bvals), len(micro), len(ODFS), bvecs.shape[0]))

signal_nod = np.zeros((len(micro), len(ODFS), bvecs.shape[0]))

tot = np.prod(signal.shape[:3])

gt_mean = np.zeros(len(micro))




i_tot=0
for i_m, mic in enumerate(micro):
	lam, v_int = mic
	# ground truth spherical mean for this microstructure configuration
	gt_mean[i_m] = e_all((bvals*1e-3).mean(), lam, v_int)
	for i_o, pdf in enumerate(ODFS):
		# "ground truth" signal assuming no gradient non-linearity
		signal_nod[i_m, i_o, :] = h_gs((bvals*1e-3), bvecs, sphere.vertices, pdf, lam, v_int)
		for i_b in range(len(new_bvals)):
			vecs = new_bvecs[i_b]
			vals = new_bvals[i_b]
			signal[i_b, i_m, i_o, :] = h_gs((vals*1e-3), vecs, sphere.vertices, pdf, lam, v_int)
			i_tot += 1
			if not i_tot%1000:
				print('{} / {}   {:.2f}'.format(i_tot, tot, i_tot/float(tot)))

gt_mean = np.exp(-bb*lams[0])



# from dipy.core.sphere import Sphere
# plotODF(make2D(signal.squeeze()), Sphere(xyz=bvecs))



# pl.figure()
# pl.imshow(signal.squeeze()-gt_mean)
# pl.colorbar()
# pl.show()





import dipy.reconst.dti as dti
from dipy.core.gradients import gradient_table

gtab = gradient_table(bvals, bvecs)
tenmodel = dti.TensorModel(gtab)
tenfit = tenmodel.fit(signal)




pl.figure()
# pl.hist(tenfit.md.ravel(), 10)
# pl.title('MD')
# pl.figure()
pl.hist(tenfit.fa.ravel(), 10)
pl.title('FA')
# pl.figure()
# pl.hist(tenfit.sphericity.ravel(), 10)
# pl.title('Sphericity')
# pl.figure()
# pl.hist(tenfit.linearity.ravel(), 10)
# pl.title('Linearity')
# pl.figure()
# pl.hist(tenfit.planarity.ravel(), 10)
# pl.title('Planarity')
pl.show()












