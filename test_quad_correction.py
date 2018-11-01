import numpy as np
import pylab as pl
from utils import plotODF, make2D, sphPDF, sphPDF_sym, h_int, h_ext, h_all, h_gs, e_int, e_ext, e_all


res_folder = '/NOBACKUP2/paquette/SMT_gradnonlin_res/'

dpath = '/home/raid2/paquette/work/SMT_gradnonlin/data/'
bvecs = np.genfromtxt(dpath+'bvecs_b10.txt')
if bvecs.shape[1] != 3:
	bvecs = bvecs.T

bvals = np.genfromtxt(dpath+'bvals_b10.txt')

# remove b0
bvecs = bvecs[bvals>10]
bvals = bvals[bvals>10]
bvals_low = bvals*(2/10.)
bvals_high = bvals*(3/10.)


# sym
bvecs = np.concatenate((bvecs,-bvecs), axis=0)
bvals_low = np.concatenate((bvals_low, bvals_low), axis=0)
bvals_high = np.concatenate((bvals_high, bvals_high), axis=0)

# # load grad non lin tensor centroid
# centroid = np.load(dpath + 'dev_parc_centroid_n_100.npy')
# centroid = centroid.reshape(centroid.shape[0],3,3)


new_bvecs_low = np.load(res_folder + 'bvecs_b2_1.npy')
new_bvals_low = np.load(res_folder + 'bvals_b2_1.npy')

new_bvecs_high = np.load(res_folder + 'bvecs_b3_1.npy')
new_bvals_high = np.load(res_folder + 'bvals_b3_1.npy')

gt_mean_low = np.load(res_folder + 'gt_mean_b2_1.npy')
signal_nod_low = np.load(res_folder + 'signal_nod_b2_1.npy')
signal_low = np.load(res_folder + 'signal_b2_1.npy')

gt_mean_high = np.load(res_folder + 'gt_mean_b3_1.npy')
signal_nod_high = np.load(res_folder + 'signal_nod_b3_1.npy')
signal_high = np.load(res_folder + 'signal_b3_1.npy')



# All directions are modeled independantly
# D and K model

# De = (b1**2*np.log(S2) - b2**2*np.log(S1)) / (b2**2*b1 - b1**2*b2)
Ds = ((new_bvals_low[:,None,None,:]*1e-3)**2*np.log(signal_high) - (new_bvals_high[:,None,None,:]*1e-3)**2*np.log(signal_low)) / ((new_bvals_high[:,None,None,:]*1e-3)**2*(new_bvals_low[:,None,None,:]*1e-3) - (new_bvals_low[:,None,None,:]*1e-3)**2*(new_bvals_high[:,None,None,:]*1e-3))

# Ke = 6*(np.log(S1)+b1*De) / (b1**2*De**2)
Ks = 6*(np.log(signal_low)+(new_bvals_low[:,None,None,:]*1e-3)*Ds) / ((new_bvals_low[:,None,None,:]*1e-3)**2*Ds**2)

# S1 = np.exp(-b1*D+((b1*D)**2*K)/6.)
# S2 = np.exp(-b2*D+((b2*D)**2*K)/6.)
new_signal_low = np.exp(-(bvals_low*1e-3)*Ds+(((bvals_low*1e-3)*Ds)**2*Ks)/6.)
new_signal_high = np.exp(-(bvals_high*1e-3)*Ds+(((bvals_high*1e-3)*Ds)**2*Ks)/6.)




# fix microstructure configuration
i_m = 0

pl.figure()
pl.axhline(gt_mean_low[i_m], label='GT')
pl.errorbar(np.arange(len(new_bvals_low))+0.0, signal_low[:,i_m,:,:].mean(axis=(1,2)), yerr=signal_low[:,i_m,:,:].mean(axis=2).std(axis=1), fmt='.', label='Distorted (+/-std)')
pl.errorbar(np.arange(len(new_bvals_low))+0.2, new_signal_low[:,i_m,:,:].mean(axis=(1,2)), yerr=new_signal_low[:,i_m,:,:].mean(axis=2).std(axis=1), fmt='.', label='ADC corr (+/-std)')
pl.title('Effect of GNL on sph mean')
pl.legend()

pl.figure()
pl.axhline(gt_mean_high[i_m], label='GT')
pl.errorbar(np.arange(len(new_bvals_high))+0.0, signal_high[:,i_m,:,:].mean(axis=(1,2)), yerr=signal_high[:,i_m,:,:].mean(axis=2).std(axis=1), fmt='.', label='Distorted (+/-std)')
pl.errorbar(np.arange(len(new_bvals_high))+0.2, new_signal_high[:,i_m,:,:].mean(axis=(1,2)), yerr=new_signal_high[:,i_m,:,:].mean(axis=2).std(axis=1), fmt='.', label='ADC corr (+/-std)')
pl.title('Effect of GNL on sph mean')
pl.legend()

pl.show()



for i_m in range(25):
	pl.figure()
	pl.subplot(1,2,1)
	pl.axhline(gt_mean_low[i_m], label='GT')
	pl.errorbar(np.arange(len(new_bvals_low))+0.0, signal_low[:,i_m,:,:].mean(axis=(1,2)), yerr=signal_low[:,i_m,:,:].mean(axis=2).std(axis=1), fmt='.', label='Distorted (+/-std)')
	pl.errorbar(np.arange(len(new_bvals_low))+0.2, new_signal_low[:,i_m,:,:].mean(axis=(1,2)), yerr=new_signal_low[:,i_m,:,:].mean(axis=2).std(axis=1), fmt='.', label='ADC corr (+/-std)')
	pl.title('Effect of GNL on sph mean')
	pl.legend()

	pl.subplot(1,2,2)
	pl.axhline(gt_mean_high[i_m], label='GT')
	pl.errorbar(np.arange(len(new_bvals_high))+0.0, signal_high[:,i_m,:,:].mean(axis=(1,2)), yerr=signal_high[:,i_m,:,:].mean(axis=2).std(axis=1), fmt='.', label='Distorted (+/-std)')
	pl.errorbar(np.arange(len(new_bvals_high))+0.2, new_signal_high[:,i_m,:,:].mean(axis=(1,2)), yerr=new_signal_high[:,i_m,:,:].mean(axis=2).std(axis=1), fmt='.', label='ADC corr (+/-std)')
	pl.title('Effect of GNL on sph mean')
	pl.legend()

pl.show()











