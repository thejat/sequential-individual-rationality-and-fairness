'''
SIR region plotter

'''

import numpy as np
import time
import pickle
from collections import OrderedDict
from profit_optimization_core import distance, active_customers_j

def boundary_given_alpha_op(params):

	#Logging data initialize
	data = {}
	for key in ['sir_condition']:
		data[key] = np.zeros((params['xvals'].shape[0],params['yvals'].shape[0]))

	#Customer 1 initialize
	customers = OrderedDict()
	customers[1] = {}
	customers[1]['s'] = np.array([0,0]) #HARDCODE
	customers[1]['d'] = np.array([2.5,0])  #HARDCODE
	customers[1]['sd']  = distance(customers[1]['s'],customers[1]['d'])
	customers[1]['alpha_k'] = params['alpha_same']

	# Customer 2 initialize, these will be overwritten in the for loops below
	customers[2] = {}
	customers[2]['s'] = np.array([1,.8])
	customers[2]['d'] = customers[1]['d']
	customers[2]['sd']  = distance(customers[2]['s'],customers[2]['d'])
	customers[2]['alpha_k'] = params['alpha_same']

	if params['scenario'] in ['sssd','ssssd']:

		#Initialize customer 3		
		customers[3] = {}
		customers[3]['s'] = np.array([1.7,.75]) #HARDCODE
		customers[3]['d'] = customers[1]['d']
		customers[3]['sd']  = distance(customers[3]['s'],customers[3]['d'])
		customers[3]['alpha_k'] = params['alpha_same']

		if params['scenario'] == 'ssssd':

			#Initialize customer 4		
			customers[4] = {}
			customers[4]['s'] = np.array([2,0.1])
			customers[4]['d'] = customers[1]['d']
			customers[4]['sd']  = distance(customers[4]['s'],customers[4]['d'])
			customers[4]['alpha_k'] = params['alpha_same']


	for idxi,i in enumerate(params['xvals']):
		
		print('Time elapsed:','%.3f'%(time.time()-params['start_time']),'alpha_op',params['alpha_op'],': CustJ xloc is ', i,' of ',params['xvals'][-1])

		for idxj,j in enumerate(params['yvals']):
			
			if params['scenario']=='ssd':
				customers[2]['s'] = np.array([i,j])
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
			active_customer_idxes = active_customers_j(customers)
			customer_jm1 = active_customer_idxes[-1]

			threshold_sir = 5e-2


			data['sir_condition'][idxi,idxj] = \
					distance(customers[customer_jm1]['s'],customers[customer_j]['s']) + customers[customer_j]['sd'] - customers[customer_jm1]['sd'] \
						- customers[customer_j]['sd']/(1 + (sum([customers[acidx]['alpha_k'] for acidx in active_customer_idxes]))/params['alpha_op'])

	return {'data':data,'params':params,'customers':customers}
	# pickle.dump(all_data,open('./output/all_data.pkl','wb'))

if __name__=='__main__':

	#Constants for Model
	params = {}
	params['scenario']= 'ssd' # 'ssd' #'sssd' # 'ssssd'
	params['start_time'] = time.time()
	params['alpha_same'] =  1
	params['alpha_op_array'] = [.1,1]

	#Location grid and other choices
	params['xy_grid_resolution_num'] = 100 #NOTE: Anything above 10 is large!!!
	params['x_min'] = -3
	params['x_max'] = 3
	params['y_min'] = -3
	params['y_max'] = 3
	params['xvals'] = np.array(list(range(params['x_min']*params['xy_grid_resolution_num'],params['x_max']*params['xy_grid_resolution_num'],1)))/params['xy_grid_resolution_num']
	params['yvals'] = np.array(list(range(params['y_min']*params['xy_grid_resolution_num'],params['y_max']*params['xy_grid_resolution_num'],1)))/params['xy_grid_resolution_num']

	print('Run scenario: ',params['scenario'])
	print('alpha_op_array is',params['alpha_op_array'])

	all_data  = []
	for alpha_op in params['alpha_op_array']:
		params['alpha_op'] = alpha_op
		all_data.append(boundary_given_alpha_op(params))

	pickle.dump(all_data,open('./output/sir_data_'+params['scenario']+'.pkl','wb'))
	print('Experiment finished. Time elapsed', time.time()-params['start_time'])