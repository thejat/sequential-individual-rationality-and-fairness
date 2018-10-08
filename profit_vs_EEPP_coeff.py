'''
Analysis of maximum profit as a function of EEPP_coeff for 2 customers, single source setting
'''

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import time
import copy
import pickle
from multiprocessing import Pool
from config import *
from profit_optimization_core import *


def opt_profits_given_multiplier(params):

	data = {}

	for key in params['all_data_keys']:
		data[key] = np.zeros((params['xvals'].shape[0],params['yvals'].shape[0]))



	customers = OrderedDict()
	customers[1] = {}
	customers[2] = {}
	customers[1]['s'] = np.array([0,0])
	customers[1]['d'] = np.array([2.5,0])
	customers[1]['p_s'] = params['p_s_1']
	customers[1]['p_x'] = params['support_v'][1]

	for idx in [1]:
		customers[idx]['sd']  = distance(customers[idx]['s'],customers[idx]['d'])
		customers[idx]['delta_bar'] = params['delta_same']
		customers[idx]['k_delta_bar'] = degradation(customers[idx]['delta_bar'],params['degradation_multiplier'])
		customers[idx]['actual_detour_wo_j'] = 0
		customers[idx]['is_bootstrapped'] = True

	assert_p_s_1_greater_than_c_op(customers[1]['p_s'],params['c_op'])
	assert_ex_ante_customer1_IR(params['support_v'],customers[1]['p_s'],customers[1]['delta_bar'],customers[1]['k_delta_bar'],customers[1]['sd'])


	for idxi,i in enumerate(params['xvals']):
		
		print('Time elapsed:','%.3f'%(time.time()-params['start_time']),'EEPP_coeff',params['EEPP_coeff'],': Cust2 xloc is ', i,' of ',params['xvals'][-1])

		for idxj,j in enumerate(params['yvals']):
			

			customers[2]['s'] = np.array([i,j])
			customers[2]['d'] = customers[1]['d']

			for idx in [2]:
				customers[idx]['sd']  = distance(customers[idx]['s'],customers[idx]['d'])
				customers[idx]['delta_bar'] = params['delta_same']
				customers[idx]['k_delta_bar'] = degradation(customers[idx]['delta_bar'],params['degradation_multiplier'])
				customers[idx]['actual_detour_wo_j'] = 0
				customers[idx]['is_bootstrapped'] = False


			if customers[idx]['k_delta_bar'] < 0:
				data['px'][idxi,idxj] = phi_v_inv(params['c_op']) #HARDCODED
				data['ps'][idxi,idxj] = phi_v_inv(params['c_op'])
				data['profitval'][idxi,idxj] = get_incremental_profit_adding_j([data['px'][idxi,idxj],data['ps'][idxi,idxj]],customers,params['c_op'],params['support_v'],params['degradation_multiplier'],params['EEPP_coeff'],t_j)
			else:

				[profit,prices,profit_surface] = maximize_incremental_profit_j(params,customers)
				data['profitval'][idxi,idxj] = profit
				data['ps'][idxi,idxj] = prices.get('p_s')
				data['px'][idxi,idxj] = prices.get('p_x')


			t_j = opt_customer_to_drop_after_j(customers)
			(prob_exclusive_val,prob_pool_val,incr_profit_exclusive_val,incr_profit_pool_val,expost_penalty_sum) = get_incremental_profit_adding_j_components([prices['p_x'],prices['p_s']],customers,params['c_op'],params['support_v'],params['degradation_multiplier'],params['EEPP_coeff'],t_j)

			data['expost_penalty'][idxi,idxj] = expost_penalty_sum
			data['profitvals_choose_pool'][idxi,idxj] = incr_profit_pool_val
			data['profitvals_choose_exclu'][idxi,idxj] = incr_profit_exclusive_val
			data['profitvals_choose_exclu_vs_pool'][idxi,idxj] = incr_profit_exclusive_val - incr_profit_pool_val
			data['profitvals_choose_exclu_vs_pool_sign'][idxi,idxj] = np.sign(incr_profit_exclusive_val - incr_profit_pool_val)           
			data['prob_pool'][idxi,idxj] = prob_pool_val
			data['prob_exclusive'][idxi,idxj] = prob_exclusive_val
			data['prob_nothing'][idxi,idxj] = 1 - prob_pool_val - prob_exclusive_val


			data['circle_delta_1_bar'][idxi,idxj] = indicator_of(abs(distance(customers[1]['s'],customers[2]['s']) + distance(customers[2]['s'],customers[1]['d']) - (1 +customers[1]['delta_bar'])*customers[1]['sd']) < 5e-2) #HARDCODE

			data['circle_s1d'][idxi,idxj] = indicator_of(abs(distance(np.array([i,j]),customers[1]['s']) -customers[1]['sd']) < 1e-2)

	data['profitval_and_prob_pool'] = data['profitval']*np.sign(data['prob_pool'])#-np.min(data['prob_pool'])
	temp1 = .1 + data['profitval_and_prob_pool']
	temp2 = 1 - data['circle_delta_1_bar']
	data['profitval_and_prob_pool_and_delta1max'] = temp1*temp2

	return {'data':data,'params':params}
	# pickle.dump(all_data,open('./output/all_data.pkl','wb'))


def plot_data(data_params,EEPP_coeff):
	data = data_params['data']
	params = data_params['params']

	for key in params['plot_keys00']:
		temp = pd.DataFrame(data=data[key],index=params['xvals'],columns=params['yvals'])
		ax = sns.heatmap(temp, cmap="YlGnBu")
		fig = ax.get_figure()
		fig.savefig('./output/'+key+'_multiplier'+str(EEPP_coeff)+'.png', bbox_inches='tight', pad_inches=0)
		fig.clf()

	for key in params['plot_keys01']:
		assert np.max(data[key]) <= 1
		assert np.min(data[key]) >= 0
		temp = pd.DataFrame(data=data[key],index=params['xvals'],columns=params['yvals'])
		ax = sns.heatmap(temp, cmap="YlGnBu",vmin=0, vmax=1)
		fig = ax.get_figure()
		fig.savefig('./output/'+key+'_multiplier'+str(EEPP_coeff)+'.png', bbox_inches='tight', pad_inches=0)
		fig.clf()

	for key in params['plot_keys02']:
		assert np.min(data[key]) >= 0
		temp = pd.DataFrame(data=data[key],index=params['xvals'],columns=params['yvals'])
		if key=='ps':
			temp_vmax = params['p_x_max']
		elif key=='px':
			temp_vmax = params['p_x_max']
		ax = sns.heatmap(temp, cmap="YlGnBu",vmin=0, vmax=temp_vmax)
		fig = ax.get_figure()
		fig.savefig('./output/'+key+'_multiplier'+str(EEPP_coeff)+'.png', bbox_inches='tight', pad_inches=0)
		fig.clf()

if __name__=='__main__':
	params['start_time'] = time.time()

	EEPP_coeff_array = [0.25,.5,1,2]


	if params['multiprocessing'] is True:
		plist = []
		for idx,EEPP_coeff in enumerate(EEPP_coeff_array):
			temp = copy.deepcopy(params)
			temp['EEPP_coeff'] = EEPP_coeff
			plist.append(temp)
		with Pool(params['nprocesses']) as p:
			all_data = p.map(opt_profits_given_multiplier,plist)
	else:
		all_data  = []
		for EEPP_coeff in EEPP_coeff_array:
			params['EEPP_coeff'] = EEPP_coeff
			all_data.append(opt_profits_given_multiplier(params))
			pickle.dump(all_data,open('./output/all_data.pkl','wb'))


	for idx,EEPP_coeff in enumerate(EEPP_coeff_array):
		plot_data(all_data[idx],EEPP_coeff)
	
	pickle.dump(all_data,open('./output/all_data.pkl','wb'))
	print('Experiment Profit vs EEPP_coeff finished. Time elapsed', time.time()-params['start_time'])


