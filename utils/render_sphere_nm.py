import numpy as np

def render_sphere_nm(radius, num):
	# nm is a batch of normal maps
	nm = []

	for i in range(num):
		### hemisphere
		height = 2*radius
		width = 2*radius
		centre = radius
		x_grid, y_grid = np.meshgrid(np.arange(1.,2*radius+1), np.arange(1.,2*radius+1))
		# grids are (-radius, radius)
		x_grid -= centre
		# y_grid -= centre
		y_grid = centre - y_grid
		# scale range of h and w grid in (-1,1)
		x_grid /= radius
		y_grid /= radius
		dist = 1 - (x_grid**2+y_grid**2)
		mask = dist > 0
		z_grid = np.ones_like(mask) * np.nan
		z_grid[mask] = np.sqrt(dist[mask])

		# remove xs and ys by masking out nans in zs
		x_grid[~(mask)] = np.nan
		y_grid[~(mask)] = np.nan

		# concatenate normal map
		nm.append(np.stack([x_grid,y_grid,z_grid],axis=2))



		### sphere 
		# span the regular grid for computing azimuth and zenith angular map
		# height = 2*radius
		# width = 2*radius
		# centre = radius
		# h_grid, v_grid = np.meshgrid(np.arange(1.,2*radius+1), np.arange(1.,2*radius+1))
		# # grids are (-radius, radius)
		# h_grid -= centre
		# # v_grid -= centre
		# v_grid = centre - v_grid
		# # scale range of h and v grid in (-1,1)
		# h_grid /= radius
		# v_grid /= radius

		# # z_grid is linearly spread along theta/zenith in range (0,pi)
		# dist_grid = np.sqrt(h_grid**2+v_grid**2)
		# dist_grid[dist_grid>1] = np.nan
		# theta_grid = dist_grid * np.pi
		# z_grid = np.cos(theta_grid)

		# rho_grid = np.arctan2(v_grid,h_grid)
		# x_grid = np.sin(theta_grid)*np.cos(rho_grid)
		# y_grid = np.sin(theta_grid)*np.sin(rho_grid)

		# # concatenate normal map
		# nm.append(np.stack([x_grid,y_grid,z_grid],axis=2))


	# construct batch
	nm = np.stack(nm,axis=0)



	return nm

