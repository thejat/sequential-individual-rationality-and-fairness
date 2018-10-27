import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Settings for Plotting
plt.style.use('fivethirtyeight')
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
params['c_op'] 	=  .1 #0.32  #operational cost per unit time
params['delta_same'] =  .3#0.54  #fixed promise by the firm
params['support_v'] = (0,1)
params['degradation_multiplier'] = 2
params['EEPP_coeff'] = .25 #3
params['p_s_1_per_mile'] = 0.15
params['k_bar'] = 1
params['scenario'] = 'ssssd' #'sssd' # 'ssd' #'sdsd' means two different destinations, 'ssd' means a common one, two customers

#Constants for Optimization
params['solver_type'] = 'closed_form' #'gridsearch' # 
params['gridsearch_num'] = 21
params['p_x_max_per_mile'] = params['support_v'][1]

#Location grid
params['x_min'] = -1
params['x_max'] = 4
params['y_min'] = -1
params['y_max'] = 2
params['xy_grid_resolution_num'] = 20
params['xvals'] = np.array(list(range(params['x_min']*params['xy_grid_resolution_num'],params['x_max']*params['xy_grid_resolution_num'],1)))/params['xy_grid_resolution_num']
params['yvals'] = np.array(list(range(params['y_min']*params['xy_grid_resolution_num'],params['y_max']*params['xy_grid_resolution_num'],1)))/params['xy_grid_resolution_num']



#Profit vs EEPP_coeff
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
	'profitval_and_prob_pool_and_delta1bar_delta2bar_delta3bar',
	'circle_delta_bars_intersection',
	'circle_delta_1_bar_region',
	'circle_delta_2_bar_region',
	'circle_delta_3_bar_region',
	'foc_condition',
	'foc_condition_boundary',
	'foc_condition_boundary_overlay_prob_pool']

params['plot_keys'] = [
	'profitval',
	'expost_penalty',
	'ps',
	'px',
	'prob_pool',
	'prob_exclusive',
	'prob_nothing',
	't_j',
	# 'circle_s1d',
	'profitval_and_prob_pool',
	# 'circle_delta_1_bar',
	# 'circle_delta_2_bar',
	# 'circle_delta_3_bar',
	'profitval_and_prob_pool_and_delta1bar_delta2bar_delta3bar',
	# 'circle_delta_bars_intersection',
	# 'circle_delta_1_bar_region',
	# 'circle_delta_2_bar_region',
	# 'circle_delta_3_bar_region',
	# 'foc_condition',
	# 'foc_condition_boundary',
	'foc_condition_boundary_overlay_prob_pool']
params['plot_probabilities'] = [
	'prob_pool',
	'prob_exclusive',
	'prob_nothing']