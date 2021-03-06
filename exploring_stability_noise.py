import numpy as np
from dipy.data import get_sphere
import pylab as pl
# from utils import plotODF, make2D, sphPDF, sphPDF_sym, h_int, h_ext, h_all, h_gs, e_int, e_ext, e_all
from utils import plotODF, make2D, sphPDF_sym, h_gs, e_all



dpath = '/home/raid2/paquette/work/SMT_gradnonlin/data/'
bvecs = np.genfromtxt(dpath+'bvecs_b10.txt')
if bvecs.shape[1] != 3:
	bvecs = bvecs.T

bvals = np.genfromtxt(dpath+'bvals_b10.txt')

# remove b0
bvecs = bvecs[bvals>10]
bvals = bvals[bvals>10]

# sym
bvecs = np.concatenate((bvecs,-bvecs), axis=0)
bvals = np.concatenate((bvals, bvals), axis=0)

# sample spherical distribution
ODFS_low = []
sphere_low = get_sphere('repulsion724')
for mu1 in [np.array([0,0,1])]:
	for k1 in [1, 2, 4, 16]:
		# comp1 (symmetric)
		# d1 = sphPDF(k1, mu1, sphere_low.vertices)
		# d2 = sphPDF(k1, mu1, -sphere_low.vertices)
		# dd1 = (d1+d2)/2.
		# dd1 = dd1/dd1.sum()
		dd1 = sphPDF_sym(k1, mu1, sphere_low.vertices, True)
		for mu2 in [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]:
			for k2 in [1, 2, 4, 16]:
				# comp2 (symmetric)
				# d1 = sphPDF(k2, mu2, sphere_low.vertices)
				# d2 = sphPDF(k2, mu2, -sphere_low.vertices)
				# dd2 = (d1+d2)/2.
				# dd2 = dd2/dd2.sum()
				dd2 = sphPDF_sym(k2, mu2, sphere_low.vertices, True)
				for v1 in [0.25, 0.5, 0.75]:
					pdf = v1*dd1+(1-v1)*dd2
					pdf = pdf/pdf.sum()
					ODFS_low.append(pdf)


# plotODF(make2D(ODFS_low), sphere_low)



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




extrema_low = np.zeros((len(exp_sig_low),len(exp_sig_low[0]),2))
for i1 in range(len(exp_sig_low)):
    for i2 in range(len(exp_sig_low[0])):
        extrema_low[i1,i2,0] = exp_sig_low[i1][i2].min()
        extrema_low[i1,i2,1] = exp_sig_low[i1][i2].max()


# pl.figure()
# pl.imshow(extrema_low[...,0], interpolation='nearest')
# pl.colorbar()
# pl.title('Signal mins  [{:.2e} {:.2e}]'.format(extrema_low[...,0].min(), extrema_low[...,0].max()))

# pl.figure()
# pl.imshow(extrema_low[...,1], interpolation='nearest')
# pl.colorbar()
# pl.title('Signal maxs  [{:.2e} {:.2e}]'.format(extrema_low[...,1].min(), extrema_low[...,1].max()))

# pl.show()


np.random.seed(0)

cliping = False


SNR = 100.
N_trial = 100

# N_micro X N_ODF X bvec X N_trial
noisy_data_100 = np.zeros((len(exp_sig_low),len(exp_sig_low[0]),len(bvals),N_trial))
for i1 in range(len(exp_sig_low)):
	for i2 in range(len(exp_sig_low[0])):
		for i4 in range(N_trial):
			# gaussian
			noise = (1/SNR)*np.random.randn(len(bvals))
			newsig = exp_sig_low[i1][i2] + noise
			# clip inside [0,1]
			if cliping:
				noisy_data_100[i1,i2,:,i4] = np.clip(newsig, 0, 1)
			else:
				noisy_data_100[i1,i2,:,i4] = newsig

noisy_data_smt_100 = noisy_data_100.mean(axis=2)
individual_odf_mean_100 = noisy_data_smt_100.mean(axis=2)
individual_odf_std_100 = noisy_data_smt_100.std(axis=2)
group_odf_mean_100 = noisy_data_smt_100.mean(axis=(1,2))
group_odf_std_100 = noisy_data_smt_100.std(axis=(1,2))


SNR = 50.
N_trial = 100

# N_micro X N_ODF X bvec X N_trial
noisy_data_50 = np.zeros((len(exp_sig_low),len(exp_sig_low[0]),len(bvals),N_trial))
for i1 in range(len(exp_sig_low)):
	for i2 in range(len(exp_sig_low[0])):
		for i4 in range(N_trial):
			# gaussian
			noise = (1/SNR)*np.random.randn(len(bvals))
			newsig = exp_sig_low[i1][i2] + noise
			# clip inside [0,1]
			if cliping:
				noisy_data_50[i1,i2,:,i4] = np.clip(newsig, 0, 1)
			else:
				noisy_data_50[i1,i2,:,i4] = newsig

noisy_data_smt_50 = noisy_data_50.mean(axis=2)
individual_odf_mean_50 = noisy_data_smt_50.mean(axis=2)
individual_odf_std_50 = noisy_data_smt_50.std(axis=2)
group_odf_mean_50 = noisy_data_smt_50.mean(axis=(1,2))
group_odf_std_50 = noisy_data_smt_50.std(axis=(1,2))


SNR = 20.
N_trial = 100

# N_micro X N_ODF X bvec X N_trial
noisy_data_20 = np.zeros((len(exp_sig_low),len(exp_sig_low[0]),len(bvals),N_trial))
for i1 in range(len(exp_sig_low)):
	for i2 in range(len(exp_sig_low[0])):
		for i4 in range(N_trial):
			# gaussian
			noise = (1/SNR)*np.random.randn(len(bvals))
			newsig = exp_sig_low[i1][i2] + noise
			# clip inside [0,1]
			if cliping:
				noisy_data_20[i1,i2,:,i4] = np.clip(newsig, 0, 1)
			else:
				noisy_data_20[i1,i2,:,i4] = newsig

noisy_data_smt_20 = noisy_data_20.mean(axis=2)
individual_odf_mean_20 = noisy_data_smt_20.mean(axis=2)
individual_odf_std_20 = noisy_data_smt_20.std(axis=2)
group_odf_mean_20 = noisy_data_smt_20.mean(axis=(1,2))
group_odf_std_20 = noisy_data_smt_20.std(axis=(1,2))







gt_means = np.array(gt_mean)
exp_means_low = np.array(exp_mean_low)

exp_means_low_mean = exp_means_low.mean(axis=1)
exp_means_low_std = exp_means_low.std(axis=1)




pl.figure()
pl.plot(np.arange(len(exp_means_low_mean))+0.0, gt_means, '.', label = 'GT')
pl.errorbar(np.arange(len(exp_means_low_mean))+0.15, exp_means_low_mean, yerr=exp_means_low_std, fmt='.', label='Noiseless')
pl.errorbar(np.arange(len(exp_means_low_mean))+0.3, group_odf_mean_100, yerr=group_odf_std_100, fmt='.', label='SNR = {}'.format(100))
pl.errorbar(np.arange(len(exp_means_low_mean))+0.45, group_odf_mean_50, yerr=group_odf_std_50, fmt='.', label='SNR = {}'.format(50))
pl.errorbar(np.arange(len(exp_means_low_mean))+0.6, group_odf_mean_20, yerr=group_odf_std_20, fmt='.', label='SNR = {}'.format(20))
pl.title('Sph Mean :: GT vs Noiseless vs Noisy')
pl.legend()
pl.show()





# rician and order-1 corrected rician

cliping = True

sigma = 100.
N_trial = 100

# N_micro X N_ODF X bvec X N_trial
noisy2_data_100 = np.zeros((len(exp_sig_low),len(exp_sig_low[0]),len(bvals),N_trial)) # rician
noisy3_data_100 = np.zeros((len(exp_sig_low),len(exp_sig_low[0]),len(bvals),N_trial)) # rician with correction
for i1 in range(len(exp_sig_low)):
	for i2 in range(len(exp_sig_low[0])):
		for i4 in range(N_trial):
			# rician
			noise_r = (1/sigma)*np.random.randn(len(bvals))
			noise_i = (1/sigma)*np.random.randn(len(bvals))
			newsig1 = np.sqrt((exp_sig_low[i1][i2] + noise_r)**2 + noise_i**2)
			if cliping:
				newsig2 = np.sqrt(np.clip(newsig1**2 - 2*(1/sigma)**2, 0, 1))
			else:
				newsig2 = np.sqrt(newsig1**2 - 2*(1/sigma)**2)

			noisy2_data_100[i1,i2,:,i4] = newsig1
			noisy3_data_100[i1,i2,:,i4] = newsig2

noisy2_data_smt_100 = noisy2_data_100.mean(axis=2)
individual2_odf_mean_100 = noisy2_data_smt_100.mean(axis=2)
individual2_odf_std_100 = noisy2_data_smt_100.std(axis=2)
group2_odf_mean_100 = noisy2_data_smt_100.mean(axis=(1,2))
group2_odf_std_100 = noisy2_data_smt_100.std(axis=(1,2))

noisy3_data_smt_100 = noisy3_data_100.mean(axis=2)
individual3_odf_mean_100 = noisy3_data_smt_100.mean(axis=2)
individual3_odf_std_100 = noisy3_data_smt_100.std(axis=2)
group3_odf_mean_100 = noisy3_data_smt_100.mean(axis=(1,2))
group3_odf_std_100 = noisy3_data_smt_100.std(axis=(1,2))



sigma = 50.
N_trial = 100

# N_micro X N_ODF X bvec X N_trial
noisy2_data_50 = np.zeros((len(exp_sig_low),len(exp_sig_low[0]),len(bvals),N_trial)) # rician
noisy3_data_50 = np.zeros((len(exp_sig_low),len(exp_sig_low[0]),len(bvals),N_trial)) # rician with correction
for i1 in range(len(exp_sig_low)):
	for i2 in range(len(exp_sig_low[0])):
		for i4 in range(N_trial):
			# rician
			noise_r = (1/sigma)*np.random.randn(len(bvals))
			noise_i = (1/sigma)*np.random.randn(len(bvals))
			newsig1 = np.sqrt((exp_sig_low[i1][i2] + noise_r)**2 + noise_i**2)
			if cliping:
				newsig2 = np.sqrt(np.clip(newsig1**2 - 2*(1/sigma)**2, 0, 1))
			else:
				newsig2 = np.sqrt(newsig1**2 - 2*(1/sigma)**2)

			noisy2_data_50[i1,i2,:,i4] = newsig1
			noisy3_data_50[i1,i2,:,i4] = newsig2

noisy2_data_smt_50 = noisy2_data_50.mean(axis=2)
individual2_odf_mean_50 = noisy2_data_smt_50.mean(axis=2)
individual2_odf_std_50 = noisy2_data_smt_50.std(axis=2)
group2_odf_mean_50 = noisy2_data_smt_50.mean(axis=(1,2))
group2_odf_std_50 = noisy2_data_smt_50.std(axis=(1,2))

noisy3_data_smt_50 = noisy3_data_50.mean(axis=2)
individual3_odf_mean_50 = noisy3_data_smt_50.mean(axis=2)
individual3_odf_std_50 = noisy3_data_smt_50.std(axis=2)
group3_odf_mean_50 = noisy3_data_smt_50.mean(axis=(1,2))
group3_odf_std_50 = noisy3_data_smt_50.std(axis=(1,2))



sigma = 20.
N_trial = 100

# N_micro X N_ODF X bvec X N_trial
noisy2_data_20 = np.zeros((len(exp_sig_low),len(exp_sig_low[0]),len(bvals),N_trial)) # rician
noisy3_data_20 = np.zeros((len(exp_sig_low),len(exp_sig_low[0]),len(bvals),N_trial)) # rician with correction
for i1 in range(len(exp_sig_low)):
	for i2 in range(len(exp_sig_low[0])):
		for i4 in range(N_trial):
			# rician
			noise_r = (1/sigma)*np.random.randn(len(bvals))
			noise_i = (1/sigma)*np.random.randn(len(bvals))
			newsig1 = np.sqrt((exp_sig_low[i1][i2] + noise_r)**2 + noise_i**2)
			if cliping:
				newsig2 = np.sqrt(np.clip(newsig1**2 - 2*(1/sigma)**2, 0, 1))
			else:
				newsig2 = np.sqrt(newsig1**2 - 2*(1/sigma)**2)

			noisy2_data_20[i1,i2,:,i4] = newsig1
			noisy3_data_20[i1,i2,:,i4] = newsig2

noisy2_data_smt_20 = noisy2_data_20.mean(axis=2)
individual2_odf_mean_20 = noisy2_data_smt_20.mean(axis=2)
individual2_odf_std_20 = noisy2_data_smt_20.std(axis=2)
group2_odf_mean_20 = noisy2_data_smt_20.mean(axis=(1,2))
group2_odf_std_20 = noisy2_data_smt_20.std(axis=(1,2))

noisy3_data_smt_20 = noisy3_data_20.mean(axis=2)
individual3_odf_mean_20 = noisy3_data_smt_20.mean(axis=2)
individual3_odf_std_20 = noisy3_data_smt_20.std(axis=2)
group3_odf_mean_20 = noisy3_data_smt_20.mean(axis=(1,2))
group3_odf_std_20 = noisy3_data_smt_20.std(axis=(1,2))






# gt_means = np.array(gt_mean)
# exp_means_low = np.array(exp_mean_low)

# exp_means_low_mean = exp_means_low.mean(axis=1)
# exp_means_low_std = exp_means_low.std(axis=1)



pl.figure()
pl.plot(np.arange(len(exp_means_low_mean))+0.0, gt_means, '.', label = 'GT')
pl.errorbar(np.arange(len(exp_means_low_mean))+0.1, exp_means_low_mean, yerr=exp_means_low_std, fmt='.', label='Noiseless')
pl.errorbar(np.arange(len(exp_means_low_mean))+0.2, group2_odf_mean_100, yerr=group2_odf_std_100, fmt='.', label='sigma = {}'.format(100))
pl.errorbar(np.arange(len(exp_means_low_mean))+0.3, group3_odf_mean_100, yerr=group3_odf_std_100, fmt='.', label='sigma = {} corr'.format(100))
pl.errorbar(np.arange(len(exp_means_low_mean))+0.4, group2_odf_mean_50, yerr=group2_odf_std_50, fmt='.', label='sigma = {}'.format(50))
pl.errorbar(np.arange(len(exp_means_low_mean))+0.5, group3_odf_mean_50, yerr=group3_odf_std_50, fmt='.', label='sigma = {} corr'.format(50))
pl.errorbar(np.arange(len(exp_means_low_mean))+0.6, group2_odf_mean_20, yerr=group2_odf_std_20, fmt='.', label='sigma = {}'.format(20))
pl.errorbar(np.arange(len(exp_means_low_mean))+0.7, group3_odf_mean_20, yerr=group3_odf_std_20, fmt='.', label='sigma = {} corr'.format(20))
pl.title('Sph Mean :: GT vs Noiseless vs Noisy')
pl.legend()
pl.show()



















