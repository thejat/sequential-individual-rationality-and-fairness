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
	assert_ex_ante_customer1_IR(params['support_v'][1],customers[1]['p_p'],params['delta_small'],customers[1]['k_delta_max'],customers[1]['delta_max'],customers[1]['sd'])


	customer_j = 2 # i.e., j = 2


	for idxi,i in enumerate(params['xvals']):
		
		print('Time elapsed:','%.3f'%(time.time()-params['start_time']),'gamma',params['gamma'],'EEPP_coeff',params['EEPP_coeff'],': Cust2 xcoord is ', i,' of ',params['xvals'][-1])

		for idxj,j in enumerate(params['yvals']):
			

			customers[2]['s'] = np.array([i,j])
			customers[2]['d'] = params['d']
			for idx in customers:
				customers[idx]['sd']  = distance(customers[idx]['s'],customers[idx]['d']) #the drive by distance between customer idx and their destination
				customers[idx]['delta_max'] = params['delta_small']*customers[idx]['sd']
				customers[idx]['k_delta_max'] = degradation(customers[idx]['delta_max'],customers[idx]['sd'],params['degradation_multiplier'])


			[profit,prices,profit_surface] = maximize_incremental_profit_j(params,customer_j,customers)

			data['profitval'][idxi,idxj] = profit

			data['expost_penalty'][idxi,idxj] =  get_ex_post_IR_expected_penalty(s1s2,s2d,s1d,f_v_1,params['support_v_1'],params['p_1'],params['alpha_1'],params['delta_1_max'],params['k'])

			if len(options)==1:
				temp = 	evaluate_profit_px(options['p_x'], F_x, params['support_v_x'], params['k'], params['p_1'], params['c_op'], s1d, s2d, s1s2, params['EEPP_coeff'], f_v_1, params['support_v_1'], params['alpha_1'], params['delta_1_max'],params['gamma'])

				pp_temp = params['k']*options.get('p_x')
				print('NEED TO DOUBLE CHECK')

			else:
				temp = 	evaluate_profit_px_pp([options['p_x'],options['p_p']], F_x, params['support_v_x'], params['k'], params['p_1'], params['c_op'], s1d, s2d, s1s2, params['EEPP_coeff'], f_v_1, params['support_v_1'], params['alpha_1'], params['delta_1_max'],params['gamma'])

				pp_temp = options['p_p']

			data['pp'][idxi,idxj] = pp_temp
			data['px'][idxi,idxj] = options.get('p_x')
			data['profitvals_choose_pool_vs_nothing'][idxi,idxj] = temp['profitB'] - temp['profitC']
			data['profitvals_choose_exclu_vs_nothing'][idxi,idxj] = temp['profitA'] - temp['profitC']
			data['profitvals_choose_exclu_vs_pool'][idxi,idxj] = temp['profitA'] - temp['profitB']
			data['profitvals_choose_exclu_vs_pool_sign'][idxi,idxj] = np.sign(temp['profitA'] - temp['profitB'])           
			data['prob_pool'][idxi,idxj] = temp['probB']
			data['prob_exclusive'][idxi,idxj] = temp['probA']
			data['prob_nothing'][idxi,idxj] = temp['probC']


			data['circle_delta_1_max'][idxi,idxj] = indicator_of(abs(s1s2+s2d-s1d -params['delta_1_max']) < 5e-2) #HARDCODE

			data['circle_s1d'][idxi,idxj] = indicator_of(abs(distance(np.array([i,j]),params['s1']) -s1d) < 1e-2)

			data['circle_test1'][idxi,idxj] = indicator_of((temp['probB']==1 and distance(np.array([i,j]),params['s1']) > s1d) or (temp['probB']<1 and distance(np.array([i,j]),params['s1']) < s1d))

			# data['circle_test2'][idxi,idxj] = indicator_of(get_detour_pooled(s1s2,s2d,s1d) -1.5*s2d < 0)

			data['circle_test3'][idxi,idxj] = data['prob_pool'][idxi,idxj]+data['circle_s1d'][idxi,idxj]


	data['profitval_and_prob_pool'] = data['profitval']*np.sign(data['prob_pool']-np.min(data['prob_pool']))
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
			temp_vmax = params['k']*params['p_max']
		elif key=='px':
			temp_vmax = params['p_max']
		ax = sns.heatmap(temp, cmap="YlGnBu",vmin=0, vmax=temp_vmax)
		fig = ax.get_figure()
		fig.savefig('./output/'+key+'_multiplier'+str(EEPP_coeff)+'.png', bbox_inches='tight', pad_inches=0)
		fig.clf()

if __name__=='__main__':
	params['start_time'] = time.time()

	EEPP_coeff_array = [1]


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


