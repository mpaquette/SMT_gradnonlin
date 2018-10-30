import numpy as np
import nibabel as nib 
import sklearn.cluster as clu
import pylab as pl




dpath = '/home/raid2/paquette/work/SMT_gradnonlin/data/'

# bvecs = np.genfromtxt(dpath+'bvecs_b10.txt')
# if bvecs.shape[1] != 3:
# 	bvecs = bvecs.T

# bvals = np.genfromtxt(dpath+'bvals_b10.txt')

dev_X = nib.load(dpath+'test_x.nii.gz').get_data()
dev_Y = nib.load(dpath+'test_y.nii.gz').get_data()
dev_Z = nib.load(dpath+'test_z.nii.gz').get_data()

mask_img = nib.load(dpath+'newmask.nii.gz')
mask = mask_img.get_data()



# convert percentage to fraction
dev_X *= 0.01
dev_Y *= 0.01
dev_Z *= 0.01

# make the grad non lin tensor
dev = np.concatenate((dev_X, dev_Y, dev_Z), axis=3)

dev_flat = dev[mask.astype(np.bool)]


n_parcel = 3
# for n_parcel in range(2,21):
for n_parcel in range(100,101):
	print(n_parcel)
	n_init = 3
	clusterer = clu.MiniBatchKMeans(n_clusters=n_parcel, n_init=n_init, init="k-means++", compute_labels=True)
	fitted_cluster = clusterer.fit(dev_flat)

	# ll = np.array([(fitted_cluster.labels_==i).sum() for i in range(n_parcel)])
	# # print(ll.sum(), pts.shape[0])
	# print('parcel_id   #voxel')
	# for i in range(n_parcel):
	#     print(i, ll[i])

	parcelation = np.zeros_like(mask)
	parcelation[mask.astype(np.bool)] = fitted_cluster.labels_ + 1

	output = nib.Nifti1Image(parcelation.astype(np.float32), affine=mask_img.affine, header=mask_img.header)
	nib.save(output, dpath + 'dev_parc_n_{}.nii.gz'.format(n_parcel))

	np.save(dpath+'dev_parc_centroid_n_{}.npy'.format(n_parcel), fitted_cluster.cluster_centers_)


# pl.figure()
# pl.imshow(parcelation[:,:,20])
# pl.title('XY plane, center Z')
# pl.figure()
# pl.imshow(parcelation[:,35,:])
# pl.title('XZ plane, center Y')
# pl.figure()
# pl.imshow(parcelation[35,:,:])
# pl.title('YZ plane, center X')
# pl.show()







