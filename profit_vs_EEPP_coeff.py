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


	params['s1'] 	= np.array([0,0])
	params['d'] 	= np.array([2.5,0])

	customers = {1:{},2:{}} #for 2 customers
	customers[1]['s'] = params['s1']
	customers[1]['d'] = params['d']
	customers[1]['p_p'] = params['p_p_1']
	assert_p_p_1_greater_than_c_op(customers[1]['p_p'],params['c_op'])
	for idx in [1]:
		customers[idx]['sd']  = distance(customers[idx]['s'],customers[idx]['d']) #the drive by distance between customer idx and their destination
		customers[idx]['delta_max'] = params['delta_small']*customers[idx]['sd']
		customers[idx]['k_delta_max'] = degradation(customers[idx]['delta_max'],customers[idx]['sd'],params['degradation_multiplier'])
	assert_ex_ante_customer1_IR(params['support_v'][1],customers[1]['p_p'],params['delta_small'],customers[1]['k_delta_max'],customers[1]['delta_max'],customers[1]['sd'])


	customer_j = 2 # i.e., j = 2


	for idxi,i in enumerate(params['xvals']):
		
		print('Time elapsed:','%.3f'%(time.time()-params['start_time']),'EEPP_coeff',params['EEPP_coeff'],': Cust2 xcoord is ', i,' of ',params['xvals'][-1])

		for idxj,j in enumerate(params['yvals']):
			

			customers[2]['s'] = np.array([i,j])
			customers[2]['d'] = params['d']
			for idx in customers:
				customers[idx]['sd']  = distance(customers[idx]['s'],customers[idx]['d']) #the drive by distance between customer idx and their destination
				customers[idx]['delta_max'] = params['delta_small']*customers[idx]['sd']
				customers[idx]['k_delta_max'] = degradation(customers[idx]['delta_max'],customers[idx]['sd'],params['degradation_multiplier'])


			[profit,prices,profit_surface] = maximize_incremental_profit_j(params,customer_j,customers)

			data['profitval'][idxi,idxj] = profit
			data['pp'][idxi,idxj] = prices.get('p_p')
			data['px'][idxi,idxj] = prices.get('p_x')



			prob_exclusive_val,prob_pool_val,profit_exclusive_val,profit_pool_val,expost_penalty = incremental_profit_j_single_destination_components([prices['p_x'],prices['p_p']],params['delta_small'],params['c_op'],params['support_v'],params['EEPP_coeff'],params['degradation_multiplier'],customer_j,customers)
			data['expost_penalty'][idxi,idxj] = expost_penalty
			data['profitvals_choose_pool'][idxi,idxj] = profit_pool_val
			data['profitvals_choose_exclu'][idxi,idxj] = profit_exclusive_val
			data['profitvals_choose_exclu_vs_pool'][idxi,idxj] = profit_exclusive_val - profit_pool_val
			data['profitvals_choose_exclu_vs_pool_sign'][idxi,idxj] = np.sign(profit_exclusive_val - profit_pool_val)           
			data['prob_pool'][idxi,idxj] = prob_pool_val
			data['prob_exclusive'][idxi,idxj] = prob_exclusive_val
			data['prob_nothing'][idxi,idxj] = 1 - prob_pool_val - prob_exclusive_val


			data['circle_delta_1_max'][idxi,idxj] = indicator_of(abs(get_detour_two_customers_common_destination(customers[1],customers[2]) -customers[1]['delta_max']) < 5e-2) #HARDCODE

			data['circle_s1d'][idxi,idxj] = indicator_of(abs(distance(np.array([i,j]),params['s1']) -customers[1]['sd']) < 1e-2)

	data['profitval_and_prob_pool'] = data['profitval']*np.sign(data['prob_pool'])#-np.min(data['prob_pool'])
	temp1 = .1 + data['profitval_and_prob_pool']
	temp2 = 1 - data['circle_delta_1_max']
	data['profitval_and_prob_pool_and_delta1max'] = temp1*temp2

	return {'data':data,'params':params}
	# pickle.dump(all_data,open('./output/all_data.pkl','wb'))


def plot_data(data_params):
	data = data_params['data']
	params = data_params['params']

	for key in params['plot_keys00']:
		temp = pd.DataFrame(data=data[key],index=params['xvals'],columns=params['yvals'])
		ax = sns.heatmap(temp, cmap="YlGnBu")
		fig = ax.get_figure()
		fig.savefig('./output/'+key+'_multiplier'+str(EEPP_coeff)+'.png', bbox_inches='tight', pad_inches=0)
		fig.clf()

	for key in params['plot_keys01']:
		temp = pd.DataFrame(data=data[key],index=params['xvals'],columns=params['yvals'])
		ax = sns.heatmap(temp, cmap="YlGnBu",vmin=0, vmax=1)
		fig = ax.get_figure()
		fig.savefig('./output/'+key+'_multiplier'+str(EEPP_coeff)+'.png', bbox_inches='tight', pad_inches=0)
		fig.clf()

	for key in params['plot_keys02']:
		temp = pd.DataFrame(data=data[key],index=params['xvals'],columns=params['yvals'])
		if key=='pp':
			temp_vmax = params['p_max']
		elif key=='px':
			temp_vmax = params['p_max']
		ax = sns.heatmap(temp, cmap="YlGnBu",vmin=0, vmax=temp_vmax)
		fig = ax.get_figure()
		fig.savefig('./output/'+key+'_multiplier'+str(EEPP_coeff)+'.png', bbox_inches='tight', pad_inches=0)
		fig.clf()

if __name__=='__main__':
	params['start_time'] = time.time()

	EEPP_coeff_array = [5]


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
		plot_data(all_data[idx])
	
	pickle.dump(all_data,open('./output/all_data.pkl','wb'))
	print('Experiment finished. Time elapsed', time.time()-params['start_time'])


