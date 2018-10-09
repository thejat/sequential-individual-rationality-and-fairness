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
params['c_op'] 	=  0.3  #operational cost per unit time
params['delta_same'] =  0.3  #fixed promise by the firm
params['support_v'] = (0,1)
params['p_x_max'] = params['support_v'][1]	#maximum price that the firm charges
params['degradation_multiplier'] = 3
params['EEPP_coeff'] = 1
params['p_s_1'] = 0.35

#Constants for Optimization
params['solver_type'] = 'gridsearch'
params['gridsearch_resolution'] = .05






#Constants for Experiment
#####Customer 2's location grid
params['x_min'] = -28
params['x_max'] = 52
params['y_max'] = 40
params['xvals'] = np.array(list(range(params['x_min'],params['x_max'],1)))/10
params['yvals'] = np.array(list(range(-params['y_max'],params['y_max'],1)))/10
#####for profit vs EEPP_coeff
params['multiprocessing'] = False
params['scenario'] = 'sdsd' # 'ssd' #'sdsd' means two different destinations, 'ssd' means a common one, two customers
params['nprocesses'] = 8
params['pb0_xlim'] = [-2.7,4.3]
params['pb0_ylim'] = [-3.1,3.1]
params['all_data_keys'] = [
	'profitval',
	'expost_penalty',
	'ps',
	'px',
	'profitvals_choose_pool',
	'profitvals_choose_exclu',
	'profitvals_choose_exclu_vs_pool',
	'profitvals_choose_exclu_vs_pool_sign',
	'prob_pool',
	'prob_exclusive',
	'prob_nothing',
	'profitval_and_prob_pool',
	'circle_delta_1_bar',
	'circle_delta_2_bar',
	'circle_s1d',
	'circle_test1',
	'circle_test3',
	't_j',
	'profitval_and_prob_pool_and_delta1bar',
	'circle_delta_bars_intersection',
	'circle_delta_1_bar_region',
	'circle_delta_2_bar_region']

params['plot_keys00'] = [
	'profitval',
	'expost_penalty',
	'profitval_and_prob_pool',
	'profitval_and_prob_pool_and_delta1bar',
	't_j',
	'circle_delta_1_bar',
	'circle_delta_2_bar',
	'circle_delta_bars_intersection',
	'circle_delta_1_bar_region',
	'circle_delta_2_bar_region'
	]
params['plot_keys01'] = [
	'prob_pool',
	'prob_exclusive',
	'prob_nothing']
params['plot_keys02'] = ['ps','px']