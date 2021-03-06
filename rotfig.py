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

bb = 5

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


scores = np.load(dpath+'scores_x.npy')
cdf_all = np.load(dpath+'scores_y.npy')

pos = np.searchsorted(scores, dist_score)
cdf_pos = cdf_all[pos]


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



# generate smmoth transition orientation on the sphere
n = 200
golden_angle = np.pi * (3 - np.sqrt(5))
theta = golden_angle * np.arange(n)
z = np.linspace(1 - 1.0 / n, 1.0 / n - 1, n)
radius = np.sqrt(1 - z * z)
 
points = np.zeros((n, 3))
points[:,0] = radius * np.cos(theta)
points[:,1] = radius * np.sin(theta)
points[:,2] = z



# generate ODF att all orientation
ODFS = []
sphere = get_sphere('repulsion724').subdivide(2)
for mu1 in points:
	for k1 in [2.0]:
		# comp1 (symmetric)
		dd1 = sphPDF_sym(k1, mu1, sphere.vertices, True)	
		pdf = dd1/dd1.sum()
		ODFS.append(pdf)


plotODF(make2D(ODFS[:10:]), sphere)

plotODF(np.array(ODFS)[::10], sphere)




lams = [1]
v_ints = [0.5]
micro = [(l,v) for l in lams for v in v_ints]


# #grad_non_lin tensor X #microstructure config X #ODF X #bvec
signal = np.zeros((len(new_bvals), len(micro), len(ODFS), bvecs.shape[0]))

signal_nod = np.zeros((len(micro), len(ODFS), bvecs.shape[0]))

tot = np.prod(signal.shape[:3])

gt_mean = np.zeros(len(micro))


res_folder = '/NOBACKUP2/paquette/SMT_gradnonlin_res/'


# i_tot=0
# for i_m, mic in enumerate(micro):
# 	lam, v_int = mic
# 	# ground truth spherical mean for this microstructure configuration
# 	gt_mean[i_m] = e_all((bvals*1e-3).mean(), lam, v_int)
# 	for i_o, pdf in enumerate(ODFS):
# 		# "ground truth" signal assuming no gradient non-linearity
# 		signal_nod[i_m, i_o, :] = h_gs((bvals*1e-3), bvecs, sphere.vertices, pdf, lam, v_int)
# 		for i_b in range(len(new_bvals)):
# 			vecs = new_bvecs[i_b]
# 			vals = new_bvals[i_b]
# 			signal[i_b, i_m, i_o, :] = h_gs((vals*1e-3), vecs, sphere.vertices, pdf, lam, v_int)
# 			i_tot += 1
# 			if not i_tot%1000:
# 				print('{} / {}   {:.2f}'.format(i_tot, tot, i_tot/float(tot)))



# np.save(res_folder + 'rotfig_signal_b{}_1.npy'.format(bb), signal)
# np.save(res_folder + 'rotfig_signal_nod_b{}_1.npy'.format(bb), signal_nod)
# np.save(res_folder + 'rotfig_gt_mean_b{}_1.npy'.format(bb), gt_mean)

signal = np.load(res_folder + 'rotfig_signal_b{}_1.npy'.format(bb))
signal_nod = np.load(res_folder + 'rotfig_signal_nod_b{}_1.npy'.format(bb))
gt_mean = np.load(res_folder + 'rotfig_gt_mean_b{}_1.npy'.format(bb))




# # fix microstructure configuration
# i_m = 0

# pl.figure()
# pl.axhline(gt_mean[i_m], label='GT')
# for i_b in range(0,100,10):
# 	pl.plot(signal[i_b,i_m,:,:].mean(axis=1), label='GNL {:.2e}'.format(sorted(dist_score)[i_b]))
# # pl.title('Effect of GNL on sph mean')
# pl.legend()
# pl.show()



# # fix microstructure configuration
# i_m = 0

# pl.figure()
# pl.axhline(gt_mean[i_m], label='GT')
# pl.errorbar(np.arange(len(new_bvals))+0.0, signal[:,i_m,:,:].mean(axis=(1,2)), yerr=signal[:,i_m,:,:].mean(axis=2).std(axis=1), fmt='.', label='GNL')
# # pl.title('Effect of GNL on sph mean')
# pl.legend()
# pl.show()


# # fix microstructure configuration
# i_m = 0

# pl.figure()
# pl.axhline(0, label='GT')
# pl.errorbar(np.arange(len(new_bvals))+0.0, np.abs(signal[:,i_m,:,:].mean(axis=(1,2))-gt_mean[i_m]), yerr=signal[:,i_m,:,:].mean(axis=2).std(axis=1), fmt='o', label='GNL')
# # pl.title('Effect of GNL on sph mean')
# pl.legend()
# pl.show()


# # fix microstructure configuration
# i_m = 0

# pl.figure()
# pl.axhline(0, label='GT')
# pl.errorbar(np.arange(len(new_bvals))+0.0, (signal[:,i_m,:,:].mean(axis=(1,2))-gt_mean[i_m])/gt_mean[i_m], yerr=signal[:,i_m,:,:].mean(axis=2).std(axis=1)/gt_mean[i_m], fmt='.', label='GNL')
# # pl.title('Effect of GNL on sph mean')
# pl.legend()
# pl.show()



		

gt_meanmod = np.load(res_folder + 'gt_mean_mod_b{}_1.npy'.format(bb))
gt_meanmod = gt_meanmod[np.argsort(dist_score)]


pl.rcParams.update({'font.size': 20})


# fix microstructure configuration
i_m = 0


# tmp = np.linspace(0,1,6)
# colors = [(0.2+0.4*i,i,1-i) for i in tmp]

# colors = [(0,1,0),
# (0,0,1),
# (0,1,1),
# (1,1,0),
# (0,0,0),
# (1,0,1)]


colors = [(0,1,0),
		  (0,0,1),
		  (0,1,1),
		  (1,0.8,0.3),
		  (0,0,0),
		  (1,0.2,1)]

np.random.seed(1234)
pl.figure()
# for i_b in range(0,100,5):
# for i_b in range(20,30):
ii = 0
for i_b in [19,7,5,14,9,27]:
	# cc = np.random.rand(3)
	cc = colors[ii]
	ii+=1
	pl.axhline(gt_meanmod[i_b, 7], c=cc)
	pl.plot(signal[i_b,i_m,:,:].mean(axis=1), c=cc, label='GNL score = {:.2f} (better than {:.1f}% of voxels)'.format(dist_score[i_b], 100-100*cdf_pos[i_b]))
pl.axhline(gt_mean[i_m], label='Undistorted ground truth', c=(1.0,0.0,0.0))
# frame1 = pl.gca()
# frame1.axes.xaxis.set_label('Orientation of structure')
# frame1.axes.yaxis.set_label('SM')
pl.xlabel('Orientation of structure', size=30)
pl.ylabel('Spherical Mean', size=30)

pl.gca().axes.get_xaxis().set_ticks([])

# pl.title('SM of (lam,v_int)={} at b~{} vs mean b for various GNLT'.format(micro[i_m],bb))
pl.title('Spherical Mean for a single microstructure configuration at all orientations')
pl.legend()
pl.show()






