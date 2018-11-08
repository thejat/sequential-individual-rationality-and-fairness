import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Settings for Plotting
plt.style.use('seaborn-white')
plt.rcParams['font.size'] 		= 30
plt.rcParams['axes.labelsize'] 	= 30
plt.rcParams['axes.titlesize'] 	= 30
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 30
plt.rcParams['figure.titlesize']= 30
# plt.rcParams['text.usetex'] 	= True


#Constants for Model
params = {}
params['c_op'] 	=  .1
params['delta_same'] =  .3
params['support_v'] = (0,np.inf) #(0,1)
params['degradation_multiplier'] = 2
params['EEPP_coeff'] = 1
params['p_s_1_per_mile'] = 0.15
params['k_bar'] = 1

#Constants for Optimization
params['solver_type'] = 'closed_form' #  'gridsearch' #
params['gridsearch_num'] = 21
if np.isinf(params['support_v'][1]):
	params['p_x_max_per_mile'] = 1e3 #params['support_v'][1] #TBD Bad Hardcode
else:
	params['p_x_max_per_mile'] = params['support_v'][1]

#Profit vs EEPP_coeff parameters

params['scenario'] =  'ssd' #'all' #'sdsdsd' #'sdsd' #'ssssd' #'sssd' # 'ssd' #here 'sdsd' and 'sdsdsd' mean different destinations, 'ssd' means a common one, two customers
params['sdsdsd_scale'] = 'small'


#Location grid and other choices
params['xy_grid_resolution_num'] = 20 #NOTE: Anything above 10 is large!!!
if params['scenario'] in ['sdsd','sdsdsd']:
	if params['sdsdsd_scale']=='large':
		params['x_min'] = -5
		params['x_max'] = 15
		params['y_min'] = -15
		params['y_max'] = 5
		params['xy_grid_resolution_num'] = 20 #20 #100
	else:	
		params['x_min'] = -1
		params['x_max'] = 3
		params['y_min'] = -2
		params['y_max'] = 2
elif params['scenario'] in ['ssd','sssd','ssssd']:
	params['x_min'] = -3
	params['x_max'] = 3
	params['y_min'] = -3
	params['y_max'] = 3
else:
	params['x_min'] = -3
	params['x_max'] = 3
	params['y_min'] = -3
	params['y_max'] = 3
params['EEPP_coeff_array'] = [1,20]
params['xvals'] = np.array(list(range(params['x_min']*params['xy_grid_resolution_num'],params['x_max']*params['xy_grid_resolution_num'],1)))/params['xy_grid_resolution_num']
params['yvals'] = np.array(list(range(params['y_min']*params['xy_grid_resolution_num'],params['y_max']*params['xy_grid_resolution_num'],1)))/params['xy_grid_resolution_num']

params['multiprocessing'] = False
params['nprocesses'] = 8

params['all_data_keys'] = [
	'profitval',
	'expost_penalty',
	'ps',
	'px',
	'prob_pool',
	'prob_exclusive',
	'prob_nothing',
	't_j',
	'circle_s1d',
	'profitval_and_prob_pool',
	'circle_delta_1_bar',
	'circle_delta_2_bar',
	'circle_delta_3_bar',
	'profitval_and_prob_pool_and_deltabars',
	'circle_delta_1_bar_region',
	'circle_delta_2_bar_region',
	'circle_delta_3_bar_region',
	'foc_condition',
	'foc_condition_boundary',
	'foc_condition_boundary_overlay_prob_pool',
	'deltabars_intersection',
	's2d2']

params['plot_keys'] = [
	# 'profitval',
	# 'expost_penalty',
	# 'ps',
	# 'px',
	'prob_pool',
	# 'prob_exclusive',
	# 'prob_nothing',
	# 't_j',
	# 'circle_s1d',
	# 'profitval_and_prob_pool',
	# 'circle_delta_1_bar',
	# 'circle_delta_2_bar',
	# 'circle_delta_3_bar',
	'profitval_and_prob_pool_and_deltabars',
	'circle_delta_1_bar_region',
	'circle_delta_2_bar_region',
	'circle_delta_3_bar_region',
	# 'foc_condition',
	# 'foc_condition_boundary',
	'foc_condition_boundary_overlay_prob_pool',
	'deltabars_intersection']

params['plot_probabilities'] = [
	'prob_pool',
	'prob_exclusive',
	'prob_nothing']