import numpy as np
from dipy.data import get_sphere
# from utils import plotODF, make2D, sphPDF, sphPDF_sym, h_int, h_ext, h_all, h_gs, e_int, e_ext, e_all
from utils import plotODF, make2D, sphPDF_sym





# # sphere = get_sphere('repulsion724')
# sphere = get_sphere('repulsion724').subdivide()

# dists = []
# # for mu in [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]:
# for mu in [np.array([0,0,1])]:
# 	# for k in [1.,4.,10.]:
# 	for k in np.linspace(0.1,20,49):
# 		# make symmetric
# 		d1 = sphPDF(k, mu, sphere.vertices)
# 		d2 = sphPDF(k, mu, -sphere.vertices)
# 		dd = (d1+d2)/2.
# 		dists.append(dd)
# ODF = make2D(dists)
# plotODF(ODF, sphere)


# 2-mixture
sphere = get_sphere('repulsion724')
# sphere = get_sphere('repulsion724').subdivide()


dists = []
for mu1 in [np.array([0,0,1])]:
	for k1 in [1, 2, 4, 16]:
		# comp1 (symmetric)
		# d1 = sphPDF(k1, mu1, sphere.vertices)
		# d2 = sphPDF(k1, mu1, -sphere.vertices)
		# dd1 = (d1+d2)/2.
		# dd1 = dd1/dd1.sum()
		dd1 = sphPDF_sym(k1, mu1, sphere.vertices, True)
		for mu2 in [np.array([1,0,0]), np.array([0,1,0])]:
			for k2 in [1, 2, 4, 16]:
				# comp2 (symmetric)
				# d1 = sphPDF(k2, mu2, sphere.vertices)
				# d2 = sphPDF(k2, mu2, -sphere.vertices)
				# dd2 = (d1+d2)/2.
				# dd2 = dd2/dd2.sum()
				dd2 = sphPDF_sym(k2, mu2, sphere.vertices, True)
				for v1 in [0.25, 0.5, 0.75]:
					dists.append(v1*dd1+(1-v1)*dd2)
					# print(k1, k2, v1)
					# plotODF(v1*dd1+(1-v1)*dd2, sphere)
ODF = make2D(dists)
plotODF(ODF, sphere)








