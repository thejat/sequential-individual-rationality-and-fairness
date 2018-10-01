# Accounting for Expected Ex Post IR
'''
This file has the core function that maximizes profit of the firm in a two rider ride-sharing setting.

Author: Theja Tulabandhula
Year: 2018

'''
import numpy as np 
import math, time
from config import *

def phi_v_inv(y):
	'''
	Inverse function of virtual valuation for the uniform distribution. 
	See https://en.wikipedia.org/wiki/Regular_distribution_(economics)
	'''
	return 0.5*(1+y)

def f_v(v,support_v):
	'''
	PDF of Uniform[0,1]
	'''
	if v >= support_v[0] and v<= support_v[1]:
		return 1.0/(support_v[1] - support_v[0])
	else:
		return 0

def F_v(v,support_v):
	'''
	CDF of Uniform[0,1]
	'''
	return min(max(0,(v- support_v[0])*1.0/(support_v[1] - support_v[0])),1)

def indicator_of(boo):
	'''
	returns 1 if argument is true, else returns false
	'''
	if boo:
		return 1
	else:
		return 0

def distance(a,b):
	dist = np.linalg.norm(a-b)
	return dist

def degradation(delta_i,sidi,degradation_multiplier):
	'''
	delta_i is essentially a variable and this function needs to be reevaluated everytime
	'''
	degradation_coeff = 1 - delta_i/(degradation_multiplier*sidi)
	return degradation_coeff

def assert_p_p_1_greater_than_c_op(p_p_1,c_op):
	assert p_p_1 >= c_op

def expost_customer1_integration_lowerbound(p_p_1,delta_small,k_delta_1):
	lb = p_p_1*(1 + delta_small)/k_delta_1
	return lb

def get_customer1_IR(v_1,p_p_1,delta_small,k_delta_1,delta_1,s1d1):
	temp = s1d1*k_delta_1*(v_1 - expost_customer1_integration_lowerbound(p_p_1,delta_small,k_delta_1))
	return temp

def assert_ex_ante_customer1_IR(v_1,p_p_1,delta_small,k_1,delta_1_max,s1d1):
	#customer 1's ex ante IR constraint
	ir_1_ante = get_customer1_IR(v_1,p_p_1,delta_small,k_1)
	print('Ex ante customer 1 IR should be nonneg for support_v[1]:',ir_1_ante)
	assert ir_1_ante > 0

def get_detour_two_customers_common_destination(cust1,cust2):
	'''
	actual detour of customer i (who immediately preceeds j)  if customer j is offered AND decides to pool
	applicability: multi-source single destination
	'''

	delta_j = distance(cust1['s'],cust2['s']) + cust2['sd'] - cust1['sd']
	return delta_j

def get_cumulative_expected_ex_post_penalty_customer1(s1d,support_v,p_p_1,delta_1,delta_small,k_delta_1):
	'''
	only called when customer 2 chooses pool
	customer 1's ex post IR constraint if customer 2 decides to pool
	different from other customer's expected ex post penalties
	NEED TO CHECK: TBD INCREMENTAL if more than {1,2,destination}
	'''
	# expectation conditioned on ex ante IR being satisfied
	lb = expost_customer1_integration_lowerbound(p_p_1,delta_small,k_delta_1)

	# k_delta_1 is a function of delta_1 and can change

	expected_ex_post_penalty_cust1 = integrate.quad(lambda v1var: f_v(v1var,support_v)*max(0,-( k_delta_1*v1var*s1d - p_p_1*(s1d + delta_1))),
										min(max(support_v[0],lb),support_v[1]),support_v[1]) 
	# print('expected ex post customer 1 IR penalty:',expected_ir_1_post_penalty[0])

	return expected_ex_post_penalty_cust1[0]

def previous_customers_j(customer_j):
	return [x for x in range(1,customer_j)] #NEED TO CHECK

def sum_EEPPs(customer_j,customers,support_v,delta_small):
	previous_customer_idxes = previous_customers_j(customer_j)
	summed_val = 0
	for idx in previous_customer_idxes:
		if idx ==1: #our first customer is indexed from 1 and NOT 0
			delta_1 = get_detour_two_customers_common_destination(customers[1],customers[2])
			summed_val += get_cumulative_expected_ex_post_penalty_customer1(customers[1]['s1d1'],support_v,customers[1]['p_p_1'],delta_1,delta_small,customers[1]['k_delta_1'])
		else:
			summed_val += 0 #TBD

	return summed_val

def sum_previous_customer_prices(customer_j,customers):
	previous_customer_idxes = previous_customers_j(customer_j)
	summed_p_p = 0
	for idx in previous_customer_idxes:
		summed_p_p += customers[idx]['p_p']
	return summed_p_p

def prob_exclusive_j(p_x_j,p_p_j,delta_small,support_v,k_delta_j_max):
	'''
	applicability: multi-source and multi-destination
	'''
	argument = (p_x_j - p_p_j*(1+delta_small))/(1-k_delta_j_max)
	prob_exclusive_val = 1 - F_v(argument,support_v)
	return prob_exclusive_val

def prob_pool_j(p_x_j,p_p_j,delta_small,support_v,k_delta_j_max,flag_print_arguments=False):
	'''
	applicability: multi-source and multi-destination
	'''
	argument1 = (p_x_j - p_p_j*(1+delta_small))/(1-k_delta_j_max)
	argument2 = p_p_j*(1+delta_small)/k_delta_j_max
	prob_pool_val = F_v(argument1,support_v) - F_v(argument2,support_v)
	if flag_print_arguments is True:
		print('Need args (probability of pooling computation) between 0 and 1 with the second larger than the firs: ',argument1,argument2)
	return prob_pool_val

def incremental_profit_j_single_destination_components(x,delta_small,c_op,support_v,EEPP_coeff,customer_j,customers):
	'''
	applicability: multi-source and single-destination
	'''
	#Because of single destination
	delta_op_j = get_detour_two_customers_common_destination(customers[1],customers[2])
	delta_j = 0
	EEPPjj = 0

	p_x_var,p_p_var = x[0],x[1]
	prob_exclusive_val = prob_exclusive_j(p_x_var,p_p_var,delta_small,support_v,customers[customer_j]['k_delta_j_max'])
	prob_pool_val = prob_pool_j(p_x_var,p_p_var,delta_small,support_v,customers[customer_j]['k_delta_j_max'])
	profit_exclusive_val = (p_x_var - c_op)*customers[customer_j]['sd']
	profit_pool_val = p_p_var*(customers[customer_j]['sd'] + delta_j) + (sum_previous_customer_prices(customer_j,customers)-c_op)*delta_op_j - EEPP_coeff*sum_EEPPs(customer_j,customers) - EEPP_coeff*EEPPjj

	return (prob_exclusive_val,prob_pool_val,profit_exclusive_val,profit_pool_val)

def incremental_profit_j_single_destination(x,delta_small,c_op,support_v,EEPP_coeff,customer_j,customers):

	prob_exclusive_val,prob_pool_val,profit_exclusive_val,profit_pool_val = incremental_profit_j_single_destination_components(x,delta_small,c_op,support_v,EEPP_coeff,customer_j,customers)
	return prob_exclusive_val*profit_exclusive_val + prob_pool_val*profit_pool_val

def pricing_feasibility_constraint(x,delta_small,k_delta_j_max):
	p_x_var,p_p_var = x[0],x[1]
	return k_delta_j_max*p_x_var - p_p_var*(1 + delta_small) 


def maximize_incremental_profit_j(params,customer_j,customers):

	solver_type = params['solver_type']
	c_op =	params['c_op']
	p_max =	params['p_max']
	EEPP_coeff = params['EEPP_coeff']
	gridsearch_resolution = params['gridsearch_resolution']
	delta_small = params['delta_small']
	support_v = params['support_v']

	sjdj = customers[customer_j]['sd']
	k_delta_j_max = customers[customer_j]['k_delta_max']

	px_lb = c_op
	px_ub = p_max
	pp_lb = 0
	pp_ub = k_delta_j_max*p_max/(1 + delta_small)
	initial_guess = [(1+c_op)/2,k_1*(1+c_op)/2]
	assert px_lb <= initial_guess[0] <= px_ub
	assert pp_lb <= initial_guess[1] <= pp_ub
	print('initial_guess',initial_guess)
	profit = incremental_profit_j_single_destination(initial_guess,delta_small,c_op,support_v,EEPP_coeff,customer_j,customers)
	profit_surface = None

	if solver_type == 'gridsearch':
		print('\nUsing Gridsearch:')
		px_gridsearch_num = int((p_max-c_op)/gridsearch_resolution)
		pp_gridsearch_num = int((pp_ub - pp_lb)/gridsearch_resolution)
		px_gridvals = np.linspace(px_lb,px_ub,num=px_gridsearch_num)
		pp_gridvals = np.linspace(pp_lb,pp_ub,num=pp_gridsearch_num)
		profit_surface = np.zeros((px_gridsearch_num,pp_gridsearch_num))

		for idxx,p_x_var in enumerate(px_gridvals):
			for idxp,p_p_var in enumerate(pp_gridvals):
				if pricing_feasibility_constraint([p_x_var,p_p_var],delta_small,k_delta_j_max) >= 0:

					profit_var = incremental_profit_j_single_destination([p_x_var,p_p_var],delta_small,c_op,support_v,EEPP_coeff,customer_j,customers)
					profit_surface[idxx,idxp] = profit_var
					if profit_var > profit:
						profit = profit_var
						p_x_opt = p_x_var
						p_p_opt = p_p_var
	else:
		print('NO SOLVER!')

	return (profit,{'p_x':p_x_opt,'p_p':p_p_opt},profit_surface)


if __name__=='__main__':

	params['s1'] 	= np.array([0,0])
	params['d'] 	= np.array([2.5,0])
	params['s2'] 	= np.array([1,1])

	customers = {1:{},2:{}} #for 2 customers
	customers[1]['s'] = params['s1']
	customers[1]['d'] = params['d']
	customers[2]['s'] = params['s2']
	customers[2]['d'] = params['d']


	for idx in customers:
		customers[idx]['sd']  = distance(customers[idx]['s'],customers[idx]['d']) #the drive by distance between customer idx and their destination
		customers[idx]['delta_max'] = params['delta_small']*customers[idx]['sd']
		customers[idx]['k_delta_max'] = degradation(customers[idx]['delta_max'],customers[idx]['sd'],params['degradation_multiplier'])
	
		print('idx',idx,'sd',customers[idx]['sd'],'delta_max',customers[idx]['delta_max'],'k_delta_max',customers[idx]['k_delta_max'])


	customer_j = 2 # i.e., j = 2
	[profit,prices,profit_surface] = maximize_incremental_profit_j(params,customer_j,customers)
	print('profit',profit,'prices',prices)
	print('pricing constraint should be positive: ',pricing_feasibility_constraint([prices['p_x'],prices['p_p']],params['delta_small'],customers[customer_j]['k_delta_max']))	


	# 	prob_exclusive_val,prob_pool_val,profit_exclusive_val,profit_pool_val = profit_1_components([prices['p_x'],prices['p_p']],params['delta_small'],s1d1,params['c_op'],params['support_v'])
	# 	print('prob exclusive: ',prob_exclusive_val,'prob pool: ',prob_pool_val,'profit exclusive: ',profit_exclusive_val,'profit pool: ',profit_pool_val)

	# 	temp = prob_pool_1(prices['p_x'],prices['p_p'],params['delta_small'],s1d1,params['support_v'],flag_print_arguments=True)
