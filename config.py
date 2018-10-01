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
params['delta_small'] =  0.4  #fixed promise by the firm
params['support_v'] = (0,1)
params['p_max'] = params['support_v'][1]	#maximum price that the firm charges
params['degradation_multiplier'] = 4
params['EEPPcoeff'] = 1


#Constants for Optimization
params['gridsearch_resolution'] = .01
params['solver_type'] = 'gridsearch'

