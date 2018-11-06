import numpy as np
import pylab as pl
from utils import plotODF, make2D, sphPDF, sphPDF_sym, h_int, h_ext, h_all, h_gs, e_int, e_ext, e_all


res_folder = '/NOBACKUP2/paquette/SMT_gradnonlin_res/'

dpath = '/home/raid2/paquette/work/SMT_gradnonlin/data/'
bvecs = np.genfromtxt(dpath+'bvecs_b10.txt')
if bvecs.shape[1] != 3:
	bvecs = bvecs.T

bvals = np.genfromtxt(dpath+'bvals_b10.txt')

bbSS = [1, 2, 3, 5, 10, 11, 12, 12.5, 15, 20]

# remove b0
bvecs = bvecs[bvals>10]
bvals = bvals[bvals>10]

# sym
bvecs = np.concatenate((bvecs,-bvecs), axis=0)
bvals = np.concatenate((bvals, bvals), axis=0)

# load grad non lin tensor centroid
centroid = np.load(dpath + 'dev_parc_centroid_n_100.npy')
centroid = centroid.reshape(centroid.shape[0],3,3)

_,s,_ = np.linalg.svd(centroid)
dist_score = np.linalg.norm(s-np.ones(3), axis=1)
idx = np.argsort(dist_score)



new_bvecsSS = []
new_bvalsSS = []
gt_meanSS = []
signal_nodSS = []
signalSS = []
new_signalSS = []

for bb in bbSS:

	bvalsS = bvals*(bb/10.)

	new_bvecs = np.load(res_folder + 'bvecs_b{}_1.npy'.format(bb))
	new_bvals = np.load(res_folder + 'bvals_b{}_1.npy'.format(bb))

	gt_mean = np.load(res_folder + 'gt_mean_b{}_1.npy'.format(bb))
	signal_nod = np.load(res_folder + 'signal_nod_b{}_1.npy'.format(bb))
	signal = np.load(res_folder + 'signal_b{}_1.npy'.format(bb))

	new_bvecs = new_bvecs[idx]
	new_bvals = new_bvals[idx]
	signal = signal[idx]

	new_bvecsSS.append(new_bvecs)
	new_bvalsSS.append(new_bvals)
	gt_meanSS.append(gt_mean)
	signal_nodSS.append(signal_nod)
	signalSS.append(signal)

	# All directions are modeled independantly
	# Simple exponential model

	ADCs = -np.log(signal)/(new_bvals[:,None,None,:]*1e-3)
	new_signal = np.exp(-ADCs*(bvalsS*1e-3))
	new_signalSS.append(new_signal)

	gt_ADCs = -np.log(signal_nod)/(bvalsS*1e-3)



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
	# pl.show()

	# fix microstructure configuration
	i_m = 24

	pl.figure()
	pl.axhline(gt_mean[i_m], label='GT')
	# pl.errorbar(np.arange(len(new_bvals))+0.0, signal[:,i_m,:,:].mean(axis=(1,2)), yerr=signal[:,i_m,:,:].mean(axis=2).std(axis=1), fmt='.', label='Distorted (+/-std)')
	pl.errorbar(np.arange(len(new_bvals))+0.0, signal[:,i_m,:,:].mean(axis=(1,2)), yerr=signal[:,i_m,:,:].mean(axis=2).std(axis=1), fmt='.', label='Distorted (+/-std)')
	# pl.errorbar(np.arange(len(new_bvals))+0.2, new_signal[:,i_m,:,:].mean(axis=(1,2)), yerr=new_signal[:,i_m,:,:].mean(axis=2).std(axis=1), fmt='.', label='ADC corr (+/-std)')
	pl.errorbar(np.arange(len(new_bvals))+0.2, new_signal[:,i_m,:,:].mean(axis=(1,2)), yerr=new_signal[:,i_m,:,:].mean(axis=2).std(axis=1), fmt='.', label='ADC corr (+/-std)')
	pl.title('Effect of GNL on sph mean')
	pl.legend()
	pl.show()




# for i_m in range(25):
# 	pl.figure()
# 	pl.axhline(gt_mean[i_m], label='GT')
# 	pl.errorbar(np.arange(len(new_bvals))+0.0, signal[:,i_m,:,:].mean(axis=(1,2)), yerr=signal[:,i_m,:,:].mean(axis=2).std(axis=1), fmt='.', label='Distorted')
# 	pl.errorbar(np.arange(len(new_bvals))+0.2, new_signal[:,i_m,:,:].mean(axis=(1,2)), yerr=new_signal[:,i_m,:,:].mean(axis=2).std(axis=1), fmt='.', label='ADC corr')
# 	pl.title('Effect of GNL on sph mean')
# 	pl.legend()
# pl.show()











