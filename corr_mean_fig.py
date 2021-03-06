import numpy as np
import pylab as pl
from utils import plotODF, make2D, sphPDF, sphPDF_sym, h_int, h_ext, h_all, h_gs, e_int, e_ext, e_all


res_folder = '/NOBACKUP2/paquette/SMT_gradnonlin_res/'

dpath = '/home/raid2/paquette/work/SMT_gradnonlin/data/'
bvecs = np.genfromtxt(dpath+'bvecs_b10.txt')
if bvecs.shape[1] != 3:
	bvecs = bvecs.T

bvals = np.genfromtxt(dpath+'bvals_b10.txt')

# bbSS = [1, 2, 3, 5, 10, 11, 12, 12.5, 15, 20]
bbSS = [1, 2, 3, 5, 10, 11, 12, 15, 20]

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





gt_meanSS = []
gt_meanmodSS = []
signalSS = []

for bb in bbSS:

	gt_mean = np.load(res_folder + 'gt_mean_b{}_1.npy'.format(bb))
	gt_meanmod = np.load(res_folder + 'gt_mean_mod_b{}_1.npy'.format(bb))
	signal = np.load(res_folder + 'signal_b{}_1.npy'.format(bb))

	signal = signal[idx]
	gt_meanmod = gt_meanmod[idx]

	gt_meanSS.append(gt_mean)
	gt_meanmodSS.append(gt_meanmod)
	signalSS.append(signal)




# for i_b in range(len(bbSS)):
# 	pl.figure()
# 	pl.errorbar(np.arange(len(gt_meanSS[0]))+0.0, np.array(gt_meanmodSS[i_b]).mean(axis=(0,)), yerr=np.array(gt_meanmodSS[i_b]).std(axis=(0,)), fmt='x', label='MOD')
# 	pl.plot(gt_meanSS[i_b], '.', label='ORIG')
# 	pl.legend()
# 	pl.title(bbSS[i_b])
# pl.show()


# for i_b in range(len(bbSS)):
# 	pl.figure()
# 	pl.imshow(np.abs(gt_meanmodSS[i_b] - gt_meanSS[i_b][None,:]).T/gt_meanSS[i_b][:,None], interpolation='nearest')
# 	pl.colorbar()
# 	pl.title(bbSS[i_b])
# pl.show()



from prettytable import PrettyTable



errors = np.zeros((len(bbSS),5))
bcouple = np.zeros(len(bbSS), dtype=tuple)

for ii in range(len(bbSS)):
	bcouple[ii] = (bbSS[ii],)

	gt_meanmod = gt_meanmodSS[ii]
	signal = signalSS[ii]
	# tmp = (signal.mean(3) - gt_meanmod[:,:,None])**2 / (gt_meanmod[:,:,None])**2
	# tmp = (signal.mean(3) - gt_meanmod[:,:,None])**2
	tmp = np.abs(signal.mean(3) - gt_meanmod[:,:,None])
	errors[ii] = [tmp[:,iD*5:(iD+1)*5,:].mean() for iD in range(5)]


errors_unc = np.zeros((len(bbSS),5))
for ii in range(len(bbSS)):
	print(ii)
	gt_mean = gt_meanSS[ii]
	signal = signalSS[ii]
	# tmp = (signal.mean(3) - gt_mean[:,None])**2 / (gt_mean[:,None])**2
	# tmp = (signal.mean(3) - gt_mean[:,None])**2
	tmp = np.abs(signal.mean(3) - gt_mean[:,None])
	errors_unc[ii] = [tmp[:,iD*5:(iD+1)*5,:].mean() for iD in range(5)]





x = PrettyTable()
x.field_names = ["bvals\Diff", "0.5", "1.0", "1.5", "2.0", "2.5"]
for ii in range(len(bbSS)):
	# x_low.add_row([bcouple[ii]] + errors_low[ii].tolist())
	x.add_row([bcouple[ii]] + ['{:.2e}'.format(i) for i in errors[ii]])
print(x)

x = PrettyTable()
x.field_names = ["bvals\Diff", "0.5", "1.0", "1.5", "2.0", "2.5"]
for ii in range(len(bbSS)):
	# x_low.add_row([bcouple[ii]] + errors_low[ii].tolist())
	x.add_row([bcouple[ii]] + ['{:.2e}'.format(i) for i in errors_unc[ii]])
print(x)






x_rat = PrettyTable()
x_rat.field_names = ["bvals\Diff", "0.5", "1.0", "1.5", "2.0", "2.5"]
for ii in range(len(bbSS)):
	# x_low_rat.add_row([bcouple[ii]] + errors_low[ii].tolist())
	x_rat.add_row([bcouple[ii]] + ['{:.2e}'.format(i) for i in (errors[ii]/errors_unc[bbSS.index(bcouple[ii][0])])])
print(x_rat)







import matplotlib.colors as colors

# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))



# bvalues
xaxistick = bbSS
# diffusivity
yaxistick = [0.5, 1, 1.5, 2, 2.5]


# fig, ax = pl.subplots()

# # We want to show all ticks...
# ax.set_xticks(np.arange(len(xaxistick)))
# ax.set_yticks(np.arange(len(yaxistick)))
# # ... and label them with the respective list entries
# ax.set_xticklabels(xaxistick)
# ax.set_yticklabels(yaxistick)

# # Rotate the tick labels and set their alignment.
# pl.setp(ax.get_xticklabels(), rotation=45, ha="right",
#          rotation_mode="anchor")


from matplotlib import rc
pl.rcParams.update({'font.size': 20})
rc('text', usetex=True)




AA = errors/errors_unc

vmin = 0
vmax = AA.max()
midpoint = 1


fig, ax = pl.subplots()


# pl.figure()
pl.imshow(AA.T, cmap=pl.cm.viridis, interpolation='nearest')
# pl.imshow(AA.T, cmap=pl.cm.seismic, interpolation='nearest', norm=MidpointNormalize(midpoint=midpoint,vmin=vmin,vmax=vmax))
# pl.imshow(AA.T, cmap=pl.cm.RdBu_r, interpolation='nearest', norm=MidpointNormalize(midpoint=midpoint,vmin=vmin,vmax=vmax))
pl.colorbar()
# pl.show()

# We want to show all ticks...
ax.set_xticks(np.arange(len(xaxistick)))
ax.set_yticks(np.arange(len(yaxistick)))
# ... and label them with the respective list entries
ax.set_xticklabels(xaxistick)
ax.set_yticklabels(yaxistick)

# # Rotate the tick labels and set their alignment.
# pl.setp(ax.get_xticklabels(), rotation=45, ha="right",
#          rotation_mode="anchor")

pl.xlabel(r'b-value (ms/$\mu$m$^2$)', size=30)
pl.ylabel(r'diffusivity ($\mu$m$^2$/ms)', size=30)

pl.show()



