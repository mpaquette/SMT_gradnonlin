import numpy as np
import pylab as pl
from utils import plotODF, make2D, sphPDF, sphPDF_sym, h_int, h_ext, h_all, h_gs, e_int, e_ext, e_all


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

# # load grad non lin tensor centroid
# centroid = np.load(dpath + 'dev_parc_centroid_n_100.npy')
# centroid = centroid.reshape(centroid.shape[0],3,3)


new_bvecs = np.load(res_folder + 'bvecs_b{}_1.npy'.format(bb))
new_bvals = np.load(res_folder + 'bvals_b{}_1.npy'.format(bb))

gt_mean = np.load(res_folder + 'gt_mean_b{}_1.npy'.format(bb))
signal_nod = np.load(res_folder + 'signal_nod_b{}_1.npy'.format(bb))
signal = np.load(res_folder + 'signal_b{}_1.npy'.format(bb))


# All directions are modeled independantly
# Simple exponential model

ADCs = -np.log(signal)/(new_bvals[:,None,None,:]*1e-3)
new_signal = np.exp(-ADCs*(bvals*1e-3))

gt_ADCs = -np.log(signal_nod)/(bvals*1e-3)



# fix microstructure configuration
i_m = 0

pl.figure()
pl.axhline(gt_mean[i_m], label='GT')
# pl.errorbar(np.arange(len(new_bvals))+0.0, signal[:,i_m,:,:].mean(axis=(1,2)), yerr=signal[:,i_m,:,:].mean(axis=2).std(axis=1), fmt='.', label='Distorted (+/-std)')
pl.errorbar(np.arange(len(new_bvals))+0.0, signal[:,i_m,:,:].mean(axis=(1,2)), yerr=10*signal[:,i_m,:,:].mean(axis=2).std(axis=1), fmt='.', label='Distorted (+/-10*std)')
# pl.errorbar(np.arange(len(new_bvals))+0.2, new_signal[:,i_m,:,:].mean(axis=(1,2)), yerr=new_signal[:,i_m,:,:].mean(axis=2).std(axis=1), fmt='.', label='ADC corr (+/-std)')
pl.errorbar(np.arange(len(new_bvals))+0.2, new_signal[:,i_m,:,:].mean(axis=(1,2)), yerr=10*new_signal[:,i_m,:,:].mean(axis=2).std(axis=1), fmt='.', label='ADC corr (+/-10*std)')
pl.title('Effect of GNL on sph mean')
pl.legend()
pl.show()




for i_m in range(25):
	pl.figure()
	pl.axhline(gt_mean[i_m], label='GT')
	pl.errorbar(np.arange(len(new_bvals))+0.0, signal[:,i_m,:,:].mean(axis=(1,2)), yerr=signal[:,i_m,:,:].mean(axis=2).std(axis=1), fmt='.', label='Distorted')
	pl.errorbar(np.arange(len(new_bvals))+0.2, new_signal[:,i_m,:,:].mean(axis=(1,2)), yerr=new_signal[:,i_m,:,:].mean(axis=2).std(axis=1), fmt='.', label='ADC corr')
	pl.title('Effect of GNL on sph mean')
	pl.legend()
pl.show()












