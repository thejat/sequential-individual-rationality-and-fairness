'''
This generalizes the destination of the second customer

	scenario 1
		Analysis of maximum profit as a function of EEPP_coeff for 2 customers, single source setting

	scenario 2
		for each s2, 

		s2 needs to be inside

		more flexibility of d2

		delta max ellipse

		2 cases
		where can d2 be conditional on dropping off first
			detour by 1st guy= s2d2+ d2d1 - s2d1 should be less than second
			triangle detour  + source detour < delta 1 max

		where can d2 be conditional on dropping off second
			detour by 1st guy = 0
			d1d2

		all the other parameters the same


	change scenarios using params flag

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

def idx_of_point(xvals,yvals,point):
	temp_resx = (np.max(xvals) - np.min(xvals))/xvals.shape[0]
	temp_resy = (np.max(yvals) - np.min(yvals))/yvals.shape[0]
	idx_valx = 0
	idx_valy = 0
	for idxi,i in enumerate(xvals):
		if abs(i - point[0]) <= temp_resx:
			idx_valx = idxi
			break
	for idxj,j in enumerate(yvals):
		if abs(j - point[1]) <= temp_resy:
			idx_valy = idxj
			break
	return idx_valx,idx_valy

def opt_profits_given_multiplier(params):

	#Logging data initialize
	data = {}
	for key in params['all_data_keys']:
		data[key] = np.zeros((params['xvals'].shape[0],params['yvals'].shape[0]))

	#Customer 1 initialize
	customers = OrderedDict()
	customers[1] = {}
	customers[1]['s'] = np.array([0,0]) #HARDCODE
	customers[1]['d'] = np.array([2.5,0])  #HARDCODE
	customers[1]['sd']  = distance(customers[1]['s'],customers[1]['d'])
	customers[1]['delta_bar'] = params['delta_same']
	customers[1]['k_delta_bar'] = degradation(customers[1]['delta_bar'],params['degradation_multiplier'],params['k_bar'])
	customers[1]['actual_detour_wo_j'] = 0
	customers[1]['is_bootstrapped'] = True

	#Pricing for Customer 1
	customers[1]['p_s'] = params['p_s_1_per_mile']*customers[1]['sd']
	customers[1]['p_x'] = params['support_v'][1]*customers[1]['sd']
	assert_p_s_1_greater_than_c_op(customers[1]['p_s'],params['c_op'],customers[1]['sd'])
	assert_ex_ante_customer1_IR(params['support_v'],customers[1]['p_s'],customers[1]['delta_bar'],customers[1]['k_delta_bar'],customers[1]['sd'])

	# Customer 2 initialize, these will be overwritten in the for loops below
	customers[2] = {}
	customers[2]['s'] = np.array([.5,.5])  #HARDCODE
	customers[2]['d'] = customers[1]['d']
	customers[2]['sd']  = distance(customers[2]['s'],customers[2]['d'])
	customers[2]['delta_bar'] = params['delta_same']
	customers[2]['k_delta_bar'] = degradation(customers[2]['delta_bar'],params['degradation_multiplier'],	params['k_bar'])
	customers[2]['actual_detour_wo_j'] = 0
	customers[2]['is_bootstrapped'] = False

	if params['scenario'] in ['sssd','ssssd']:

		#Pricing for Customer 2
		prices_j = solve_for_customer_j_wrapper(customers,params)
		customers = update_customer_information(customers,prices_j)

		#Initialize customer 3		
		customers[3] = {}
		customers[3]['s'] = np.array([1.7,.3]) #HARDCODE
		customers[3]['d'] = customers[1]['d']
		customers[3]['sd']  = distance(customers[3]['s'],customers[3]['d'])
		customers[3]['delta_bar'] = params['delta_same']
		customers[3]['k_delta_bar'] = degradation(customers[3]['delta_bar'],params['degradation_multiplier'],params['k_bar'])
		customers[3]['actual_detour_wo_j'] = 0
		customers[3]['is_bootstrapped'] = False

	if params['scenario'] == 'ssssd':

		#Pricing for Customer 3
		prices_j = solve_for_customer_j_wrapper(customers,params)
		customers = update_customer_information(customers,prices_j)

		#Initialize customer 4		
		customers[4] = {}
		customers[4]['s'] = np.array([2,0.1])
		customers[4]['d'] = customers[1]['d']
		customers[4]['sd']  = distance(customers[4]['s'],customers[4]['d'])
		customers[4]['delta_bar'] = params['delta_same']
		customers[4]['k_delta_bar'] = degradation(customers[4]['delta_bar'],params['degradation_multiplier'],params['k_bar'])
		customers[4]['actual_detour_wo_j'] = 0
		customers[4]['is_bootstrapped'] = False


	for idxi,i in enumerate(params['xvals']):
		
		print('Time elapsed:','%.3f'%(time.time()-params['start_time']),'EEPP_coeff',params['EEPP_coeff'],': CustJ xloc is ', i,' of ',params['xvals'][-1])

		for idxj,j in enumerate(params['yvals']):
			
			if params['scenario']=='ssd':
				customers[2]['s'] = np.array([i,j])
				customers[2]['sd']  = distance(customers[2]['s'],customers[2]['d'])
			elif params['scenario']=='sdsd':
				customers[2]['d'] = np.array([i,j])
				customers[2]['sd']  = distance(customers[2]['s'],customers[2]['d'])
			elif params['scenario']=='sssd':
				customers[3]['s'] = np.array([i,j])
				customers[3]['sd']  = distance(customers[3]['s'],customers[3]['d'])
			elif params['scenario']=='ssssd':
				customers[4]['s'] = np.array([i,j])
				customers[4]['sd']  = distance(customers[4]['s'],customers[4]['d'])
			else:
				print('ERROR: scenario is incorrectly specified.')
				break

			customer_j = len(customers)
			t_j,temp_route = opt_customer_to_drop_after_j(customers)
			[profit,prices,profit_surface] = maximize_incremental_profit_j(params,customers)
			(prob_exclusive_val,prob_pool_val,incr_profit_exclusive_val,incr_profit_pool_val,expost_penalty_sum) = get_incremental_profit_adding_j_components([prices['p_x'],prices['p_s']],customers,params['c_op'],params['support_v'],params['degradation_multiplier'],params['EEPP_coeff'],t_j,params['k_bar'])


			data['profitval'][idxi,idxj] = profit
			data['expost_penalty'][idxi,idxj] = expost_penalty_sum
			data['ps'][idxi,idxj] = prices.get('p_s')
			data['px'][idxi,idxj] = prices.get('p_x')

			threshold_min_prob = 1e-3           
			data['prob_pool'][idxi,idxj] = prob_pool_val*indicator_of(prob_pool_val > threshold_min_prob)
			data['prob_exclusive'][idxi,idxj] = prob_exclusive_val
			data['prob_nothing'][idxi,idxj] = 1 - data['prob_pool'][idxi,idxj] - prob_exclusive_val
			data['t_j'][idxi,idxj] = t_j

			threshold_circle = 5e-2
			data['circle_s1d'][idxi,idxj] = indicator_of(abs(distance(np.array([i,j]),customers[1]['s']) -customers[1]['sd']) < threshold_circle)


			if params['scenario']=='sdsd':
				if t_j == 2:
					temp_circle_val = source_detour_for_j(customers) - customers[1]['delta_bar']*customers[1]['sd']

					data['circle_delta_1_bar'][idxi,idxj] = indicator_of(abs(temp_circle_val) < threshold_circle)
					data['circle_delta_1_bar_region'][idxi,idxj] = indicator_of(temp_circle_val < 0)


					temp_circle_val = distance(customers[2]['s'],customers[1]['d']) + distance(customers[1]['d'],customers[2]['d']) - (1 +customers[2]['delta_bar'])*customers[2]['sd']

					data['circle_delta_2_bar'][idxi,idxj] = indicator_of(abs(temp_circle_val) < threshold_circle)
					data['circle_delta_2_bar_region'][idxi,idxj] = indicator_of(temp_circle_val < 0)
				elif t_j == 1:
					temp_circle_val = source_detour_for_j(customers) + destination_detour_for_j(customers,t_j) - customers[1]['delta_bar']*customers[1]['sd']

					data['circle_delta_1_bar'][idxi,idxj] = indicator_of(abs(temp_circle_val) < threshold_circle)
					data['circle_delta_1_bar_region'][idxi,idxj] = indicator_of(temp_circle_val < 0)

					data['circle_delta_2_bar'][idxi,idxj] = 0			
					data['circle_delta_2_bar_region'][idxi,idxj] = 1

			if params['scenario']=='sssd':

					temp_circle_val = distance(customers[1]['s'],customers[2]['s']) + distance(customers[2]['s'],customers[3]['s']) + distance(customers[3]['s'],customers[1]['d']) - (1 +customers[1]['delta_bar'])*customers[1]['sd']

					data['circle_delta_1_bar'][idxi,idxj] = indicator_of(abs(temp_circle_val) < threshold_circle)
					data['circle_delta_1_bar_region'][idxi,idxj] = indicator_of(temp_circle_val < 0)

					temp_circle_val = distance(customers[2]['s'],customers[3]['s']) + distance(customers[3]['s'],customers[1]['d']) - (1 +customers[2]['delta_bar'])*customers[2]['sd']					

					data['circle_delta_2_bar'][idxi,idxj] = indicator_of(abs(temp_circle_val) < threshold_circle)
					data['circle_delta_2_bar_region'][idxi,idxj] = indicator_of(temp_circle_val < 0)

			if params['scenario']=='ssssd':

					temp_circle_val = distance(customers[1]['s'],customers[2]['s']) + distance(customers[2]['s'],customers[3]['s']) + distance(customers[3]['s'],customers[4]['s']) + distance(customers[4]['s'],customers[1]['d']) - (1 +customers[1]['delta_bar'])*customers[1]['sd']

					data['circle_delta_1_bar'][idxi,idxj] = indicator_of(abs(temp_circle_val) < threshold_circle)
					data['circle_delta_1_bar_region'][idxi,idxj] = indicator_of(temp_circle_val < 0)

					temp_circle_val = distance(customers[2]['s'],customers[3]['s']) + distance(customers[3]['s'],customers[4]['s']) + distance(customers[4]['s'],customers[1]['d']) - (1 +customers[2]['delta_bar'])*customers[2]['sd']					

					data['circle_delta_2_bar'][idxi,idxj] = indicator_of(abs(temp_circle_val) < threshold_circle)
					data['circle_delta_2_bar_region'][idxi,idxj] = indicator_of(temp_circle_val < 0)

					temp_circle_val = distance(customers[3]['s'],customers[4]['s']) + distance(customers[4]['s'],customers[1]['d']) - (1 +customers[3]['delta_bar'])*customers[3]['sd']					

					data['circle_delta_3_bar'][idxi,idxj] = indicator_of(abs(temp_circle_val) < threshold_circle)
					data['circle_delta_3_bar_region'][idxi,idxj] = indicator_of(temp_circle_val < 0)

			if params['scenario'] in ['ssd','sssd','ssssd']:
				threshold_foc = 1e-2

				data['foc_condition'][idxi,idxj] = params['c_op']*(customers[customer_j]['sd']*customers[customer_j]['k_delta_bar'] - (source_detour_for_j(customers) + destination_detour_for_j(customers,t_j))) - EEPP_coeff*expost_penalty_sum

				data['foc_condition_boundary'][idxi,idxj] = indicator_of(abs(data['foc_condition'][idxi,idxj]) < threshold_foc)
				data['foc_condition_boundary_overlay_prob_pool'][idxi,idxj] = (.1 + data['prob_pool'][idxi,idxj])*(1-data['foc_condition_boundary'][idxi,idxj])



	data['profitval_and_prob_pool'] = data['profitval']*np.sign(data['prob_pool'])#-np.min(data['prob_pool'])
	temp1 = 1e-1 + data['profitval_and_prob_pool']
	temp2 = 1 - data['circle_delta_1_bar']
	temp3 = 1 - data['circle_delta_2_bar']
	temp4 = 1 - data['circle_delta_3_bar']	
	data['profitval_and_prob_pool_and_deltabars'] = temp1*temp2*temp3*temp4

	return {'data':data,'params':params,'customers':customers}
	# pickle.dump(all_data,open('./output/all_data.pkl','wb'))

def plot_data(data_params_customers,EEPP_coeff):
	data = data_params_customers['data']
	params = data_params_customers['params']
	customers = data_params_customers['customers']

	s1x,s1y = idx_of_point(params['xvals'],params['yvals'],customers[1]['s'])
	d1x,d1y = idx_of_point(params['xvals'],params['yvals'],customers[1]['d'])
	s2x,s2y = idx_of_point(params['xvals'],params['yvals'],customers[2]['s'])
	if params['scenario'] in ['sssd','ssssd']:
		s3x,s3y = idx_of_point(params['xvals'],params['yvals'],customers[3]['s'])

	for key in params['plot_keys']:
		temp = pd.DataFrame(data=data[key],index=params['xvals'],columns=params['yvals'])
		if key in params['plot_probabilities']:
			assert np.max(data[key]) <= 1
			assert np.min(data[key]) >= 0
			ax = sns.heatmap(temp, cmap="YlGnBu",vmin=0, vmax=1)
		else:
			ax = sns.heatmap(temp, cmap="YlGnBu")
	
		ax.scatter(s1y,s1x, marker='*', s=100, color='red') 
		ax.scatter(d1y,d1x, marker='*', s=100, color='red')
		if params['scenario'] in ['sdsd','sssd','ssssd']:
			ax.scatter(s2y,s2x, marker='*', s=100, color='red') 
		if params['scenario'] in ['ssssd']:
			ax.scatter(s3y,s3x, marker='*', s=100, color='red') 
		fig = ax.get_figure()
		fig.savefig('./output/'+key+'_multiplier'+str(EEPP_coeff)+'.png', bbox_inches='tight', pad_inches=0)
		fig.clf()


#Global constants
# EEPP_coeff_array = [params['EEPP_coeff']] 
EEPP_coeff_array = [0.25,1,10,50,100,1000,100000]

if __name__=='__main__':
	params['start_time'] = time.time()
	print('Run scenario: ',params['scenario'])
	print('EEPP_coeff_array is',EEPP_coeff_array)

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
	print('Experiment finished. Time elapsed', time.time()-params['start_time'])


