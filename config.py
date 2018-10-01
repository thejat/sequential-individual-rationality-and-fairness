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
params['delta_small'] =  0.3  #fixed promise by the firm
params['support_v'] = (0,1)
params['p_max'] = params['support_v'][1]	#maximum price that the firm charges
params['degradation_multiplier'] = 4
params['EEPP_coeff'] = 1
params['p_p_1'] = .6


#Constants for Optimization
params['gridsearch_resolution'] = .01
params['solver_type'] = 'gridsearch'

#Constants for Experiment
#####Customer 2's location grid
params['x_min'] = -28
params['x_max'] = 52
params['y_max'] = 40
params['xvals'] = np.array(list(range(params['x_min'],params['x_max'],1)))/10
params['yvals'] = np.array(list(range(-params['y_max'],params['y_max'],1)))/10
#####for profit vs EEPP_coeff
params['multiprocessing'] = False
params['nprocesses'] = 8
params['pb0_xlim'] = [-2.7,4.3]
params['pb0_ylim'] = [-3.1,3.1]
params['all_data_keys'] = [
	'profitval',
	'expost_penalty',
	'pp',
	'px',
	'profitvals_choose_pool_vs_nothing',
	'profitvals_choose_exclu_vs_nothing',
	'profitvals_choose_exclu_vs_pool',
	'profitvals_choose_exclu_vs_pool_sign',
	'prob_pool',
	'prob_exclusive',
	'prob_nothing',
	'profitval_and_prob_pool',
	'circle_delta_1_max',
	'circle_s1d',
	'circle_test1',
	'circle_test3']
params['plot_keys00'] = [
	'profitval',
	'expost_penalty',
	'profitval_and_prob_pool',
	'profitval_and_prob_pool_and_delta1max']
params['plot_keys01'] = [
	'prob_pool',
	'prob_exclusive',
	'prob_nothing']
params['plot_keys02'] = ['pp','px']