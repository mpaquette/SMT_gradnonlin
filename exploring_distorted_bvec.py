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

# remove b0
bvecs = bvecs[bvals>10]
bvals = bvals[bvals>10]
bvals = bvals*(1/10.)

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

np.save(res_folder + 'bvecs_b1_1.npy', np.array(new_bvecs))
np.save(res_folder + 'bvals_b1_1.npy', np.array(new_bvals))



allb = np.array(new_bvals)
allb = allb.ravel()/bvals[0]

# pl.figure()
# pl.hist(allb, 50, density=True, color=(0.0,0.0,1.0,0.3))
# sns.kdeplot(allb, bw=0.015, color='red')
# frame1 = pl.gca()
# frame1.axes.yaxis.set_ticklabels([])
# pl.title('Distribution of b-value modifier inside full brain')
# pl.show()

allv = np.array(new_bvecs)
allvdiff = allv - bvecs
angle_diff = ((180/np.pi)*np.arccos((2-np.linalg.norm(allvdiff, axis=2)**2)/2)).ravel()

# pl.figure()
# pl.hist(angle_diff, 50, density=True, color=(0.0,0.0,1.0,0.3))
# sns.kdeplot(angle_diff, bw=0.3, color='red', kernel='cos')
# frame1 = pl.gca()
# frame1.axes.yaxis.set_ticklabels([])
# pl.title('Distribution of b-value modifier inside full brain')
# pl.show()







# sample spherical distribution
ODFS = []
sphere = get_sphere('repulsion724')
for mu1 in [np.array([0,0,1])]:
	for k1 in [1, 2, 4, 16]:
		# comp1 (symmetric)
		dd1 = sphPDF_sym(k1, mu1, sphere.vertices, True)
		for mu2 in [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]:
			for k2 in [1, 2, 4, 16]:
				# comp2 (symmetric)
				dd2 = sphPDF_sym(k2, mu2, sphere.vertices, True)
				for v1 in [0.25, 0.5, 0.75]:
					pdf = v1*dd1+(1-v1)*dd2
					pdf = pdf/pdf.sum()
					ODFS.append(pdf)


# plotODF(make2D(ODFS), sphere)






lams = [0.5, 1, 1.5, 2, 2.5]
v_ints = [0.1, 0.3, 0.5, 0.7, 0.9]
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




# np.save(res_folder + 'gt_mean_b10_1.npy', gt_mean)
# np.save(res_folder + 'signal_nod_b10_1.npy', signal_nod)
# np.save(res_folder + 'signal_b10_1.npy', signal)

np.save(res_folder + 'gt_mean_b1_1.npy', gt_mean)
np.save(res_folder + 'signal_nod_b1_1.npy', signal_nod)
np.save(res_folder + 'signal_b1_1.npy', signal)





# # fix microstructure configuration
# i_m = 22

# for i_m in range(25):
# 	pl.figure()
# 	pl.axhline(gt_mean[i_m], label='GT')
# 	pl.errorbar(np.arange(len(new_bvals))+0.0, signal[:,i_m,:,:].mean(axis=(1,2)), yerr=signal[:,i_m,:,:].mean(axis=2).std(axis=1), fmt='.', label='Distorted')
# 	pl.title('Effect of GNL on sph mean')
# 	pl.legend()
# pl.show()





# # fix microstructure configuration
# i_m = 22

# pl.figure()
# for i_m in range(25):
# 	# pl.axhline(gt_mean[i_m], label='GT')
# 	pl.errorbar(np.arange(len(new_bvals))+0.5*np.random.rand(len(new_bvals)), signal[:,i_m,:,:].mean(axis=(1,2))/gt_mean[i_m], fmt='.', label='Distorted', alpha=0.5)
# 	pl.title('Effect of GNL on sph mean')
# # pl.legend()
# pl.show()











