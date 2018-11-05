import numpy as np
import nibabel as nib 
import sys


# dpath = '/home/raid2/paquette/work/SMT_gradnonlin/data/'


# dev_X = nib.load(dpath+'test_x.nii.gz').get_data()
# dev_Y = nib.load(dpath+'test_y.nii.gz').get_data()
# dev_Z = nib.load(dpath+'test_z.nii.gz').get_data()

# mask_img = nib.load(dpath+'newmask.nii.gz')
# mask = mask_img.get_data()


if __name__ == "__main__":

print("inputs: grad_perc_dev_X, grad_perc_dev_Y, grad_perc_dev_Z, mask, froout1.nii, diagout2.nii, eigout3.nii, eigout4.nii")

# load everything
dev_X = nib.load(sys.argv[1]).get_data()
dev_Y = nib.load(sys.argv[2]).get_data()
dev_Z = nib.load(sys.argv[3]).get_data()

mask_img = nib.load(sys.argv[4])
mask = mask_img.get_data()
head = mask_img.header
aff = mask_img.affine

# convert percentage to fraction
dev_X *= 0.01
dev_Y *= 0.01
dev_Z *= 0.01

# make the grad non lin tensor
dev = np.concatenate((dev_X[...,None], dev_Y[...,None], dev_Z[...,None]), axis=4)


## OUT1
## Frob norm of the whole tensor

out1 = np.linalg.norm(dev, axis=(3,4))
out1[~mask.astype(np.bool)]=0

outputimg1 = nib.nifti1.Nifti1Image(out1, affine=aff, header=head)
nib.save(outputimg1, sys.argv[5])



## OUT2
## l2 norm of diag(tensor-I)

out2 = dev - np.eye(3)
out2 = (out2[...,0,0]**2 + out2[...,1,1]**2 + out2[...,2,2]**2)**0.5
out2[~mask.astype(np.bool)]=0

outputimg2 = nib.nifti1.Nifti1Image(out2, affine=aff, header=head)
nib.save(outputimg2, sys.argv[6])


## OUT3
## sum of eigenvalue

# w,v = np.linalg.eig(dev)
U,s,V = np.linalg.svd(dev)
out3 = np.linalg.norm(s, axis=3)
out3[~mask.astype(np.bool)]=0

outputimg3 = nib.nifti1.Nifti1Image(out3, affine=aff, header=head)
nib.save(outputimg3, sys.argv[7])


## OUT4
## sum of eigenvalue-1

# U,s,V = np.linalg.svd(dev)
out4 = np.linalg.norm(s-np.ones(3), axis=3)
out4[~mask.astype(np.bool)]=0

outputimg4 = nib.nifti1.Nifti1Image(out4, affine=aff, header=head)
nib.save(outputimg4, sys.argv[8])


