import numpy as np
import nibabel as nib 
import sys

# dpath = '/home/raid2/paquette/work/SMT_gradnonlin/data/'

# bvecs = np.genfromtxt(dpath+'bvecs_b10.txt')
# if bvecs.shape[1] != 3:
# 	bvecs = bvecs.T

# bvals = np.genfromtxt(dpath+'bvals_b10.txt')

# dev_X = nib.load(dpath+'test_x.nii.gz').get_data()
# dev_Y = nib.load(dpath+'test_y.nii.gz').get_data()
# dev_Z = nib.load(dpath+'test_z.nii.gz').get_data()

# mask_img = nib.load(dpath+'bet_mask.nii.gz')
# mask = mask_img.get_data()



if __name__ == "__main__":

	print("inputs:  bvecs, bvals, grad_perc_dev_X, grad_perc_dev_Y, grad_perc_dev_Z, mask, newgrad_X.nii, newgrad_Y.nii, newgrad_Z.nii, newB.nii")

	# load everything
	bvecs = np.genfromtxt(sys.argv[1])
	if bvecs.shape[1] != 3:
		bvecs = bvecs.T

	bvals = np.genfromtxt(sys.argv[2])

	dev_X = nib.load(sys.argv[3]).get_data()
	dev_Y = nib.load(sys.argv[4]).get_data()
	dev_Z = nib.load(sys.argv[5]).get_data()

	mask_img = nib.load(sys.argv[6])
	mask = mask_img.get_data()

	# convert percentage to fraction
	dev_X *= 0.01
	dev_Y *= 0.01
	dev_Z *= 0.01

	# make the grad non lin tensor
	dev = np.concatenate((dev_X[...,None], dev_Y[...,None], dev_Z[...,None]), axis=4)


	# q space gradient
	bscaled_grad = (bvecs*np.sqrt(bvals)[:,None])

	new_bscaled_grad = np.zeros(mask.shape+bvecs.shape)
	for idx in np.ndindex(mask.shape):
		if mask[idx]:
			# distort bvecs
			new_bscaled_grad[idx] = bscaled_grad.dot(dev[idx])

	# renormalize gradient direction
	new_b = np.linalg.norm(new_bscaled_grad, axis=4)
	new_grad = new_bscaled_grad / new_b[...,None]
	# nan removal (from the b division at b0)
	new_grad[...,bvals<10,:] = 0

	# new b is the norm squared of the distorded gradient
	new_b = new_b**2


	head = mask_img.header
	aff = mask_img.affine

	# output grads and b
	outputimg = nib.nifti1.Nifti1Image(new_grad[...,0], affine=aff, header=head)
	nib.save(outputimg, sys.argv[7])
	outputimg = nib.nifti1.Nifti1Image(new_grad[...,1], affine=aff, header=head)
	nib.save(outputimg, sys.argv[8])
	outputimg = nib.nifti1.Nifti1Image(new_grad[...,2], affine=aff, header=head)
	nib.save(outputimg, sys.argv[9])
	outputimg = nib.nifti1.Nifti1Image(new_b, affine=aff, header=head)
	nib.save(outputimg, sys.argv[10])

