import numpy as np
import pylab as pl
import nibabel as nib


dpath = '/home/raid2/paquette/work/SMT_gradnonlin/data/'


# load grad non lin tensor centroid
centroid = np.load(dpath + 'dev_parc_centroid_n_100.npy')
centroid = centroid.reshape(centroid.shape[0],3,3)

_,s,_ = np.linalg.svd(centroid)
dist_score = np.linalg.norm(s-np.ones(3), axis=1)
idx = np.argsort(dist_score)



# load parcel count
p_count = np.load(dpath + 'dev_parc_centroid_n_100_counts.npy')

tot = p_count[1:].sum()
scores = np.linspace(0, dist_score.max(), 1000)


cdf = np.array([p_count[1:][dist_score<=s].sum()/float(tot) for s in scores])


# pl.figure()
# pl.plot(scores, cdf)
# pl.title('Cummulative distribution GNL severity score')
# pl.xlabel('GNL severity score')
# pl.ylabel('Proportion of voxels')
# pl.show()




########################

dev_X = nib.load(dpath+'test_x.nii.gz').get_data()
dev_Y = nib.load(dpath+'test_y.nii.gz').get_data()
dev_Z = nib.load(dpath+'test_z.nii.gz').get_data()

mask_img = nib.load(dpath+'newmask.nii.gz')
mask = mask_img.get_data()
head = mask_img.header
aff = mask_img.affine

# convert percentage to fraction
dev_X *= 0.01
dev_Y *= 0.01
dev_Z *= 0.01


# make the grad non lin tensor
dev = np.concatenate((dev_X[...,None], dev_Y[...,None], dev_Z[...,None]), axis=4)

U,s,V = np.linalg.svd(dev)
out4 = np.linalg.norm(s-np.ones(3), axis=3)
out4[~mask.astype(np.bool)]=0

all_scores = out4[mask.astype(np.bool)]

cdf_all = np.array([(all_scores<=s).sum()/float(tot) for s in scores])




# from matplotlib import rc
pl.rcParams.update({'font.size': 20})
# rc('text', usetex=True)


pl.figure()
pl.plot(scores, cdf_all)
# pl.title('Cummulative distribution GNL severity score')
pl.xlabel('GNL score', size=30)
pl.ylabel('Proportion of voxels', size=30)
pl.show()



np.save(dpath+'scores_x.npy', scores)
np.save(dpath+'scores_y.npy', cdf_all)

# pl.figure()
# pl.plot(scores, cdf_all)
# pl.plot(scores, cdf)
# pl.title('Cummulative distribution GNL severity score')
# pl.xlabel('GNL severity score')
# pl.ylabel('Proportion of voxels')
# pl.show()




