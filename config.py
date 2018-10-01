import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] 		= 30
plt.rcParams['axes.labelsize'] 	= 30
plt.rcParams['axes.titlesize'] 	= 30
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 30
plt.rcParams['figure.titlesize']= 30
# plt.rcParams['text.usetex'] 	= True


#Constants
params = {}
params['c_op'] 	=  0.3  #operational cost per unit time
params['delta_small'] =  0.4  #fixed promise by the firm
params['support_v'] = (0,1)
params['p_max'] = params['support_v'][1]	#maximum price that the firm charges
params['s1'] 	= np.array([0,0])
params['d'] 	= np.array([2.5,0])
params['gridsearch_resolution'] = .05
# params['solver_type'] = 'gridsearch'
# params['solver_type'] = 'slsqp'
params['solver_type'] = 'closed_form'
