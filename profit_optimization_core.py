# Accounting for Expected Ex Post IR
'''
This file has the core function that maximizes profit of the firm in a two rider ride-sharing setting.

Author: Theja Tulabandhula
Year: 2018

'''
import numpy as np 
import math, time
import scipy.integrate as integrate
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

def degradation(delta_i,degradation_multiplier):
	'''
	delta_i is essentially a variable and this function needs to be reevaluated everytime
	'''
	assert 0 <= delta_i <= 1 #it should be normalized by sidi beforehand
	k_bar = 1
	degradation_coeff = k_bar - delta_i/degradation_multiplier
	return degradation_coeff

def assert_p_p_1_greater_than_c_op(p_p_1,c_op):
	assert p_p_1 >= c_op

def assert_ex_ante_customer1_IR(support_v,p_s_1,delta_bar,k_delta_bar,s1d1):
	#customer 1's ex ante IR constraint
	ir_1_ante = s1d1*k_delta_bar*support_v[1] - s1d1*p_s_1*(1+delta_bar)
	print('Ex ante customer 1 IR should be nonneg for support_v[1]:',ir_1_ante)
	assert ir_1_ante > 0

def active_customers_j(customer_j):
	'''
	assume all customers before j are active
	'''
	return [x for x in range(1,customer_j)]

def pricing_feasibility_constraint(x,delta_bar,k_delta_bar):
	'''
	we need the computed value to be positive for the customer to even consider sharing. when its zero, p_s does not matter, so covers the exclusive vs declined case.
	'''
	p_x_var,p_s_var = x[0],x[1]
	return (k_delta_bar/(1 + delta_bar))*p_x_var - p_s_var

def prob_exclusive_j(p_x,p_s,delta_bar,support_v,k_delta_bar):
	'''
	applicability: multi-source and multi-destination
	'''
	v_ubar = (p_x - p_s*(1+delta_bar))/(1-k_delta_bar)
	prob_exclusive_val = 1 - F_v(v_ubar,support_v)
	return prob_exclusive_val

def prob_pool_j(p_x,p_s,delta_bar,support_v,k_delta_bar,flag_print_arguments=False):
	'''
	applicability: multi-source and multi-destination
	'''
	v_ubar = (p_x - p_s*(1+delta_bar))/(1-k_delta_bar)
	v_lbar = p_s*(1+delta_bar)/(k_delta_bar)
	if v_ubar-v_lbar < -1e-4: #HARDCODED	
		print('WARNING: Prob(Shared) computation issue. Returning 0')
		if flag_print_arguments is True:
			print('Need args (probability of choosing shared) between 0 and 1 with the second smaller than the first: ',v_ubar,'>',v_lbar)
		return 0
	prob_pool_val = F_v(v_ubar,support_v) - F_v(v_lbar,support_v)

	return min(max(0,prob_pool_val),1)

def planned_customer_drop_before_j(active_customer_idxes):
	'''
	assume customers will be dropped in the order they are indexed
	'''
	return active_customer_idxes[0]

def last_customer_picked_up(active_customer_idxes):
	'''
	for simplicity assume this is the last index
	'''
	return active_customer_idxes[-1]

def source_detour_for_j(customers,customer_j,active_customer_idxes):
	location_from_which_detour_starts = customers[active_customer_idxes[-1]]['s']
	location_next_customer_drop = customers[planned_customer_drop_before_j(active_customer_idxes)]['d']
	source_detour_val = distance(location_from_which_detour_starts,customers[customer_j]['s']) + distance(customers[customer_j]['s'],location_next_customer_drop) - distance(customers[customer_j]['s'],location_next_customer_drop)
	return source_detour_val

def set_actual_detours_wo_j():
	'''
	these fields in the customers dict need to be updated for another pickup
	'''
	raise NotImplemetedError

#=========================================


def get_detour_customers(customers,active_customer_idxes,customer_j,current_route='s1s2d'):
	'''
	applicability: multi-source multi-destination
	current status: partially implemented
	'''
	if customer_j==2 and len(active_customer_idxes)==1 and active_customer_idxes[0]==1:
		if current_route=='s1s2d1d2':
			pass
		elif current_route=='s1s2d2d1':
			pass
		elif current_route=='s1s2d':
			delta_j = distance(cust1['s'],cust2['s']) + cust2['sd'] - cust1['sd']
			return delta_j
	else:
		return NotImplemetedError

def get_cumulative_expected_ex_post_penalty_customer1(s1d,support_v,p_p_1,delta_1,delta_small,k_delta_1,k_delta_1_max):
	'''
	only called when customer 2 chooses pool
	customer 1's ex post IR constraint if customer 2 decides to pool
	different from other customer's expected ex post penalties
	NEED TO CHECK: TBD INCREMENTAL if more than {1,2,destination}
	'''
	# expectation conditioned on ex ante IR being satisfied

	def expost_customer1_integration_lowerbound(p_p_1,delta_small,k_delta_1):
		lb = p_p_1*(1 + delta_small)/k_delta_1
		return lb


	lb = expost_customer1_integration_lowerbound(p_p_1,delta_small,k_delta_1_max)

	# k_delta_1 is a function of delta_1 and can change

	expected_ex_post_penalty_cust1 = integrate.quad(lambda v1var: f_v(v1var,support_v)*max(0,-( k_delta_1*v1var*s1d - p_p_1*(s1d + delta_1))),min(max(support_v[0],lb),support_v[1]),support_v[1]) 
	# print('expected ex post customer 1 IR penalty:',expected_ir_1_post_penalty[0])

	return expected_ex_post_penalty_cust1[0]

def sum_EEPPs(customer_j,customers,support_v,delta_small,degradation_multiplier):
	active_customer_idxes = active_customers_j(customer_j)
	summed_val = 0
	for idx in active_customer_idxes:
		if idx ==1: #our first customer is indexed from 1 and NOT 0
			delta_1 = get_detour_two_customers_common_destination(customers[1],customers[2]) #based on actual detour, not cust[1]['delta_max']
			k_delta_1 = degradation(delta_1,customers[1]['sd'],degradation_multiplier)
			summed_val += get_cumulative_expected_ex_post_penalty_customer1(customers[1]['sd'],support_v,customers[1]['p_p'],delta_1,delta_small,k_delta_1,customers[1]['k_delta_max']) #HARDCODED BUT NEEDS TO CHANGE
		else:
			summed_val += 0 #TBD

	return summed_val

def sum_previous_customer_prices(customer_j,customers):
	active_customer_idxes = active_customers_j(customer_j)
	summed_p_p = 0
	for idx in active_customer_idxes:
		summed_p_p += customers[idx]['p_p']
	return summed_p_p

def incremental_profit_j_single_destination_components(x,delta_small,c_op,support_v,EEPP_coeff,degradation_multiplier,customer_j,customers):
	'''
	applicability: multi-source and single-destination
	'''
	#Because of single destination
	delta_op_j = get_detour_two_customers_common_destination(customers[1],customers[2])
	delta_j = 0
	EEPPjj = 0

	p_x_var,p_p_var = x[0],x[1]
	prob_exclusive_val = prob_exclusive_j(p_x_var,p_p_var,delta_small,support_v,customers[customer_j]['k_delta_max'])
	prob_pool_val = prob_pool_j(p_x_var,p_p_var,delta_small,support_v,customers[customer_j]['k_delta_max'])
	profit_exclusive_val = (p_x_var - c_op)*customers[customer_j]['sd']
	expost_penalty = sum_EEPPs(customer_j,customers,support_v,delta_small,degradation_multiplier) + EEPPjj
	profit_pool_val = p_p_var*(customers[customer_j]['sd'] + delta_j) + (sum_previous_customer_prices(customer_j,customers)-c_op)*delta_op_j - EEPP_coeff*expost_penalty

	return (prob_exclusive_val,prob_pool_val,profit_exclusive_val,profit_pool_val,expost_penalty)

def incremental_profit_j_single_destination(x,delta_small,c_op,support_v,EEPP_coeff,degradation_multiplier,customer_j,customers):

	prob_exclusive_val,prob_pool_val,profit_exclusive_val,profit_pool_val,expost_penalty = incremental_profit_j_single_destination_components(x,delta_small,c_op,support_v,EEPP_coeff,degradation_multiplier,customer_j,customers)
	return prob_exclusive_val*profit_exclusive_val + prob_pool_val*profit_pool_val

def maximize_incremental_profit_j(params,customer_j,customers):

	solver_type = params['solver_type']
	c_op =	params['c_op']
	p_max =	params['p_max']
	EEPP_coeff = params['EEPP_coeff']
	gridsearch_resolution = params['gridsearch_resolution']
	delta_small = params['delta_small']
	support_v = params['support_v']
	degradation_multiplier = params['degradation_multiplier']

	sjdj = customers[customer_j]['sd']
	k_delta_j_max = customers[customer_j]['k_delta_max']

	px_lb = c_op
	px_ub = p_max
	pp_lb = 0
	pp_ub = k_delta_j_max*p_max/(1 + delta_small)
	initial_guess = [min(px_ub,max(px_lb,(1+c_op)/2)),min(pp_ub,max(pp_lb,k_delta_j_max*(1+c_op)/(2*(1+delta_small))))]
	# print('initial_guess',initial_guess,'px_lb',px_lb,'px_ub',px_ub,'pp_lb',pp_lb,'pp_ub',pp_ub)
	assert px_lb <= initial_guess[0] <= px_ub
	assert pp_lb <= initial_guess[1] <= pp_ub
	# print('initial_guess',initial_guess)
	profit = incremental_profit_j_single_destination(initial_guess,delta_small,c_op,support_v,EEPP_coeff,degradation_multiplier, customer_j,customers)
	profit_surface = None
	p_x_opt = initial_guess[0]
	p_p_opt = initial_guess[1]

	if solver_type == 'gridsearch':
		# print('\nUsing Gridsearch:')
		px_gridsearch_num = int((p_max-c_op)/gridsearch_resolution)
		pp_gridsearch_num = int((pp_ub - pp_lb)/gridsearch_resolution)
		px_gridvals = np.linspace(px_lb,px_ub,num=px_gridsearch_num)
		pp_gridvals = np.linspace(pp_lb,pp_ub,num=pp_gridsearch_num)
		profit_surface = np.zeros((px_gridsearch_num,pp_gridsearch_num))

		for idxx,p_x_var in enumerate(px_gridvals):
			for idxp,p_p_var in enumerate(pp_gridvals):
				if pricing_feasibility_constraint([p_x_var,p_p_var],delta_small,k_delta_j_max) >= 0:

					profit_var = incremental_profit_j_single_destination([p_x_var,p_p_var],delta_small,c_op,support_v,EEPP_coeff,degradation_multiplier,customer_j,customers)
					profit_surface[idxx,idxp] = profit_var
					if profit_var > profit:
						profit = profit_var
						p_x_opt = p_x_var
						p_p_opt = p_p_var
	else:
		print('NO SOLVER!')

	return (profit,{'p_x':p_x_opt,'p_p':p_p_opt},profit_surface)


def new_customer_drop_after_j(active_customer_idxes,customer_j,customers):
	'''
	notation t_j used in the paper
	'''
	return pass

def destination_detour_for_j(customers,customer_j,active_customer_idxes):
	'''
	assume active_customer_idxes are ordered and have values 1,...,j-1
	'''
	


	if t_j == planned_customer_drop_before_j(active_customer_idxes):
		location_next_customer_drop = customers[planned_customer_drop_before_j(active_customer_idxes)]['d']
		destination_detour_val = customers[customer_j]['sd'] + distance(customers[customer_j]['d'],location_next_customer_drop) + distance(customers[customer_j]['s'],location_next_customer_drop)

	elif t_j > 1 and t_j in active_customer_idxes:

	else:
		destination_detour_val = 0

	return destination_detour_val

def set_actual_detours_w_j(customers,active_customer_idxes,customer_j):
	for idx in active_customer_idxes:
		customers[idx]['actual_detours_w_j'] = customers[idx]['actual_detours_wo_j'] + (source_detour_for_j(customers,customer_j,active_customer_idxes) + destination_detour_for_j())/customers[idx]['sd']

	return customers



#=========================================

if __name__=='__main__':

	# exp_s1s2d = True
	# if exp_s1s2d == True:

	customers = {1:{},2:{}} #for 2 customers
	customers[1]['s'] = np.array([0,0])
	customers[1]['d'] = np.array([2.5,0])
	customers[2]['s'] = np.array([1,3])
	customers[2]['d'] = customers[1]['d']
	customers[1]['p_s'] = params['p_s_1']

	for idx in customers:
		customers[idx]['sd']  = distance(customers[idx]['s'],customers[idx]['d'])
		customers[idx]['delta_bar'] = params['delta_same']
		customers[idx]['k_delta_bar'] = degradation(customers[idx]['delta_bar'],params['degradation_multiplier'])
		customers[idx]['actual_detour_wo_j'] = 0

		print('customer ',idx,': sd',customers[idx]['sd'],'delta_bar',customers[idx]['delta_bar'],'k_delta_bar',customers[idx]['k_delta_bar'])


	assert_p_p_1_greater_than_c_op(customers[1]['p_s'],params['c_op'])
	assert_ex_ante_customer1_IR(params['support_v'],customers[1]['p_s'],customers[1]['delta_bar'],customers[1]['k_delta_bar'],customers[1]['sd'])

	customer_j = 2 # i.e., j = 2
	active_customer_idxes = active_customers_j(customer_j)

	# [incremental_profit_j,prices_j,incremental_profit_j_surface] = maximize_incremental_profit_j(params,customer_j,customers)
	# print('Incremental profit for j:',incremental_profit_j,'prices',prices_j)