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


bvalsSS = []
new_bvecsSS = []
new_bvalsSS = []
gt_meanSS = []
signal_nodSS = []
signalSS = []


for bb in bbSS:

	bvalsS = bvals*(bb/10.)
	bvalsSS.append(bvalsS)

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



b_idx = []
new_signalSS = []


Nb = len(bbSS)
for i1 in range(Nb-1):
	bvals_low = bvalsSS[i1]
	new_bvecs_low = new_bvecsSS[i1]
	new_bvals_low = new_bvalsSS[i1]
	gt_mean_low = gt_meanSS[i1]
	signal_nod_low = signal_nodSS[i1]
	signal_low = signalSS[i1]

	for i2 in range(i1+1,Nb):
		bvals_high = bvalsSS[i2]
		new_bvecs_high = new_bvecsSS[i2]
		new_bvals_high = new_bvalsSS[i2]
		gt_mean_high = gt_meanSS[i2]
		signal_nod_high = signal_nodSS[i2]
		signal_high = signalSS[i2]



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


		b_idx.append((i1,i2))
		# new_signalSS.append((new_signal_low,new_signal_high))
		new_signalSS.append((new_signal_low.mean(3),new_signal_high.mean(3)))




for ii in range(len(b_idx)):
	i1,i2 = b_idx[ii]
	new_signal_low, new_signal_high = new_signalSS[ii]

	gt_mean_low = gt_meanSS[i1]
	gt_mean_high = gt_meanSS[i2]

	signal_low = signalSS[i1]
	signal_high = signalSS[i2]


	# fix microstructure configuration
	i_m = 8

	pl.figure()
	pl.subplot(1,2,1)
	pl.axhline(gt_mean_low[i_m], label='GT')
	pl.errorbar(np.arange(len(new_bvals_low))+0.0, signal_low[:,i_m,:,:].mean(axis=(1,2)), yerr=signal_low[:,i_m,:,:].mean(axis=2).std(axis=1), fmt='.', label='Distorted (+/-std)')
	# pl.errorbar(np.arange(len(new_bvals_low))+0.0, signal_low[:,i_m,:].mean(axis=(1,)), yerr=signal_low[:,i_m,:].std(axis=1), fmt='.', label='Distorted (+/-std)')
	# pl.errorbar(np.arange(len(new_bvals_low))+0.2, new_signal_low[:,i_m,:,:].mean(axis=(1,2)), yerr=new_signal_low[:,i_m,:,:].mean(axis=2).std(axis=1), fmt='.', label='ADC corr (+/-std)')
	pl.errorbar(np.arange(len(new_bvals_low))+0.2, new_signal_low[:,i_m,:,].mean(axis=(1)), yerr=new_signal_low[:,i_m,:].std(axis=1), fmt='.', label='ADC corr (+/-std)')
	pl.title(bbSS[i1])
	pl.legend()

	# pl.figure()
	pl.subplot(1,2,2)
	pl.axhline(gt_mean_high[i_m], label='GT')
	pl.errorbar(np.arange(len(new_bvals_high))+0.0, signal_high[:,i_m,:,:].mean(axis=(1,2)), yerr=signal_high[:,i_m,:,:].mean(axis=2).std(axis=1), fmt='.', label='Distorted (+/-std)')
	# pl.errorbar(np.arange(len(new_bvals_high))+0.0, signal_high[:,i_m,:].mean(axis=(1,)), yerr=signal_high[:,i_m,:].std(axis=1), fmt='.', label='Distorted (+/-std)')
	# pl.errorbar(np.arange(len(new_bvals_high))+0.2, new_signal_high[:,i_m,:,:].mean(axis=(1,2)), yerr=new_signal_high[:,i_m,:,:].mean(axis=2).std(axis=1), fmt='.', label='ADC corr (+/-std)')
	pl.errorbar(np.arange(len(new_bvals_high))+0.2, new_signal_high[:,i_m,:].mean(axis=(1,)), yerr=new_signal_high[:,i_m,:].std(axis=1), fmt='.', label='ADC corr (+/-std)')
	pl.title(bbSS[i2])
	pl.legend()


pl.show()





from prettytable import PrettyTable


errors_low = np.zeros((len(b_idx),5))
errors_high = np.zeros((len(b_idx),5))
bcouple = np.zeros(len(b_idx), dtype=tuple)

for ii in range(len(b_idx)):
	i1,i2 = b_idx[ii]
	new_signal_low, new_signal_high = new_signalSS[ii]

	gt_mean_low = gt_meanSS[i1]
	gt_mean_high = gt_meanSS[i2]

	# signal_low = signalSS[i1]
	# signal_high = signalSS[i2]

	bcouple[ii] = (bbSS[i1],bbSS[i2])
	# tmp = (new_signal_low - gt_mean_low[:,None])**2 / (gt_mean_low[:,None])**2
	tmp = np.abs(new_signal_low - gt_mean_low[:,None])
	errors_low[ii] = [tmp[:,iD*5:(iD+1)*5,:].mean() for iD in range(5)]
	tmp = np.abs(new_signal_high - gt_mean_high[:,None])
	# errors_high[ii] = [tmp[:,iD*5:(iD+1)*5,:].mean() for iD in range(5)]
	errors_high[ii] = [tmp[:,iD*5:(iD+1)*5,:].mean() for iD in range(5)]


errors_unc = np.zeros((len(bbSS),5))
for ii in range(len(bbSS)):
	print(ii)
	gt_mean = gt_meanSS[ii]
	signal = signalSS[ii]
	# signal = signalSS[i1] # !!!@%^#&$*%&@#$
	# tmp = (signal.mean(3) - gt_mean[:,None])**2 / (gt_mean[:,None])**2
	tmp = np.abs(signal.mean(3) - gt_mean[:,None])
	errors_unc[ii] = [tmp[:,iD*5:(iD+1)*5,:].mean() for iD in range(5)]




# np.save(res_folder + 'dkifig_errtable_low.npy', errors_low)
# np.save(res_folder + 'dkifig_errtable_high.npy', errors_high)
# np.save(res_folder + 'dkifig_errtable_unc.npy', errors_unc)



    
x_low = PrettyTable()
x_low.field_names = ["bvals\Diff", "0.5", "1.0", "1.5", "2.0", "2.5"]
for ii in range(len(b_idx)):
	# x_low.add_row([bcouple[ii]] + errors_low[ii].tolist())
	x_low.add_row([bcouple[ii]] + ['{:.2e}'.format(i) for i in errors_low[ii]])
print(x_low)

    
x_high = PrettyTable()
x_high.field_names = ["bvals\Diff", "0.5", "1.0", "1.5", "2.0", "2.5"]
for ii in range(len(b_idx)):
	# x_high.add_row([bcouple[ii]] + errors_low[ii].tolist())
	x_high.add_row([bcouple[ii]] + ['{:.2e}'.format(i) for i in errors_high[ii]])
print(x_high)


x_low_rat = PrettyTable()
x_low_rat.field_names = ["bvals\Diff", "0.5", "1.0", "1.5", "2.0", "2.5"]
for ii in range(len(b_idx)):
	# x_low_rat.add_row([bcouple[ii]] + errors_low[ii].tolist())
	x_low_rat.add_row([bcouple[ii]] + ['{:.2e}'.format(i) for i in errors_unc[bbSS.index(bcouple[ii][0])]])
print(x_low_rat)


x_high_rat = PrettyTable()
x_high_rat.field_names = ["bvals\Diff", "0.5", "1.0", "1.5", "2.0", "2.5"]
for ii in range(len(b_idx)):
	# x_low_rat.add_row([bcouple[ii]] + errors_low[ii].tolist())
	x_high_rat.add_row([bcouple[ii]] + ['{:.2e}'.format(i) for i in errors_unc[bbSS.index(bcouple[ii][1])]])
print(x_high_rat)



    
x_low = PrettyTable()
x_low.field_names = ["bvals\Diff", "0.5", "1.0", "1.5", "2.0", "2.5"]
for ii in range(len(b_idx)):
	# x_low.add_row([bcouple[ii]] + errors_low[ii].tolist())
	x_low.add_row([bcouple[ii]] + ['{:.2e}'.format(i) for i in errors_low[ii]/errors_unc[bbSS.index(bcouple[ii][0])]])
print(x_low)

    
x_high = PrettyTable()
x_high.field_names = ["bvals\Diff", "0.5", "1.0", "1.5", "2.0", "2.5"]
for ii in range(len(b_idx)):
	# x_high.add_row([bcouple[ii]] + errors_low[ii].tolist())
	x_high.add_row([bcouple[ii]] + ['{:.2e}'.format(i) for i in errors_high[ii]/errors_unc[bbSS.index(bcouple[ii][1])]])
print(x_high)




























# x_low_rat = PrettyTable()
# x_low_rat.field_names = ["bvals\Diff", "0.5", "1.0", "1.5", "2.0", "2.5"]
# for ii in range(len(b_idx)):
# 	# x_low_rat.add_row([bcouple[ii]] + errors_low[ii].tolist())
# 	x_low_rat.add_row([bcouple[ii]] + ['{:.2f}'.format(i) for i in (errors_low[ii]/errors_unc[bbSS.index(bcouple[ii][0])])])
# print(x_low_rat)

    
# x_high_rat = PrettyTable()
# x_high_rat.field_names = ["bvals\Diff", "0.5", "1.0", "1.5", "2.0", "2.5"]
# for ii in range(len(b_idx)):
# 	# x_high_rat.add_row([bcouple[ii]] + errors_low[ii].tolist())
# 	x_high_rat.add_row([bcouple[ii]] + ['{:.2f}'.format(i) for i in (errors_high[ii]/errors_unc[bbSS.index(bcouple[ii][1])])])
# print(x_high_rat)









# for ii in range(len(b_idx)):
# 	i1,i2 = b_idx[ii]
# 	new_signal_low, new_signal_high = new_signalSS[ii]

# 	gt_mean_low = gt_meanSS[i1]
# 	gt_mean_high = gt_meanSS[i2]

# 	signal_low = signalSS[i1]
# 	signal_high = signalSS[i2]


# 	# fix microstructure configuration
# 	i_m = 8

# 	pl.figure()
# 	pl.subplot(1,2,1)
# 	# pl.axhline(gt_mean_low[i_m], label='GT')
# 	# pl.plot(np.arange(len(new_bvals_low))+0.0, (signal_low[:,i_m,:,:].mean(axis=(1,2))-gt_mean_low[i_m])/gt_mean_low[i_m], '.', label='Distorted (+/-std)')
# 	pl.plot(np.arange(len(new_bvals_low))+0.0, (signal_low[:,i_m,:].mean(axis=(1,))-gt_mean_low[i_m])/gt_mean_low[i_m], '.', label='Distorted (+/-std)')
# 	# pl.plot(np.arange(len(new_bvals_low))+0.2, (new_signal_low[:,i_m,:,:].mean(axis=(1,2))-gt_mean_low[i_m])/gt_mean_low[i_m], '.', label='ADC corr (+/-std)')
# 	pl.plot(np.arange(len(new_bvals_low))+0.2, (new_signal_low[:,i_m,:].mean(axis=(1,))-gt_mean_low[i_m])/gt_mean_low[i_m], '.', label='ADC corr (+/-std)')
# 	pl.title(bbSS[i1])
# 	pl.legend()

# 	# pl.figure()
# 	pl.subplot(1,2,2)
# 	# pl.axhline(gt_mean_high[i_m], label='GT')
# 	# pl.plot(np.arange(len(new_bvals_high))+0.0, (signal_high[:,i_m,:,:].mean(axis=(1,2))-gt_mean_high[i_m])/gt_mean_high[i_m], '.', label='Distorted (+/-std)')
# 	pl.plot(np.arange(len(new_bvals_high))+0.0, (signal_high[:,i_m,:].mean(axis=(1,))-gt_mean_high[i_m])/gt_mean_high[i_m], '.', label='Distorted (+/-std)')
# 	# pl.plot(np.arange(len(new_bvals_high))+0.2, (new_signal_high[:,i_m,:,:].mean(axis=(1,2))-gt_mean_high[i_m])/gt_mean_high[i_m], '.', label='ADC corr (+/-std)')
# 	pl.plot(np.arange(len(new_bvals_high))+0.2, (new_signal_high[:,i_m,:].mean(axis=(1,))-gt_mean_high[i_m])/gt_mean_high[i_m], '.', label='ADC corr (+/-std)')
# 	pl.title(bbSS[i2])
# 	pl.legend()


# pl.show()

