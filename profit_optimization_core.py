# Accounting for Expected Ex Post IR
'''
This file has the core function that maximizes profit of the firm in a two rider ride-sharing setting.

Author: Theja Tulabandhula
Year: 2018

'''
import numpy as np 
import math, time
import scipy.integrate as integrate
from collections import OrderedDict
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
	k_bar = .8
	degradation_coeff = k_bar - delta_i/degradation_multiplier
	return degradation_coeff

def assert_p_s_1_greater_than_c_op(p_s_1,c_op):
	'''
	assuming they are bootstrapped. otherwise we don't need shared price to be greater than c_op
	'''
	assert p_s_1 >= c_op

def assert_ex_ante_customer1_IR(support_v,p_s_1,delta_bar,k_delta_bar,s1d1):
	#customer 1's ex ante IR constraint
	ir_1_ante = s1d1*k_delta_bar*support_v[1] - s1d1*p_s_1*(1+delta_bar)
	print('Ex ante customer 1 IR should be nonneg for support_v[1]:',ir_1_ante)
	assert ir_1_ante > 0

def active_customers_j(customers):
	'''
	assume all customers before j are active. i.e., 1.2,...,j-1
	'''
	return sorted(customers.keys())[:-1]

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

	return min(max(0,prob_pool_val),1),v_ubar,v_lbar

def last_customer_picked_up_or_dropped_off(active_customer_idxes):
	'''
	for simplicity assume this is the last index
	TBD
	'''
	return active_customer_idxes[-1]

def source_detour_for_j(customers):

	customer_j = len(customers)
	active_customer_idxes = active_customers_j(customers)

	location_from_which_detour_starts = customers[last_customer_picked_up_or_dropped_off(active_customer_idxes)]['s']
	location_next_customer_drop = customers[1]['d']

	# print('loc detour starts',location_from_which_detour_starts)
	# print('loc next cust drop', location_next_customer_drop)

	source_detour_val = distance(location_from_which_detour_starts,customers[customer_j]['s']) + distance(customers[customer_j]['s'],location_next_customer_drop) - distance(location_from_which_detour_starts,location_next_customer_drop)
	return source_detour_val

def set_actual_detours_wo_j(customers):
	'''
	these fields in the customers dict need to be updated for another pickup
	'''
	raise NotImplementedError

def destination_detour_for_j(customers,t_j):
	'''
	assume active_customer_idxes are ordered and have values 1,...,j-1
	'''

	customer_j = len(customers)
	active_customer_idxes = active_customers_j(customers)

	if t_j == 1:
		location_next_customer_drop = customers[t_j]['d']
		destination_detour_val = customers[customer_j]['sd'] \
							+ distance(customers[customer_j]['d'],location_next_customer_drop) \
							- distance(customers[customer_j]['s'],location_next_customer_drop)

	elif t_j > 1 and t_j < active_customer_idxes[-1]:
		destination_detour_val = distance(customers[t_j-1]['d'],customers[customer_j]['d']) \
								+ distance(customers[customer_j]['d'],customers[t_j]['d']) \
								- distance(customers[t_j]['d'],customers[t_j+1]['d'])

	elif t_j == active_customer_idxes[-1]:
		destination_detour_val = distance(customers[t_j-1]['d'],customers[customer_j]['d']) \
								+ distance(customers[customer_j]['d'],customers[t_j]['d']) \
								- distance(customers[t_j]['d'],customers[customer_j]['d'])
	else:
		destination_detour_val = distance(customers[active_customer_idxes[-1]]['d'],customers[customer_j]['d'])

	return destination_detour_val

def set_actual_detours_w_j(customers,t_j):

	customer_j = len(customers)
	active_customer_idxes = active_customers_j(customers)

	for idx in active_customer_idxes:
		# print('source detour',source_detour_for_j(customers))
		# print('dest detour',destination_detour_for_j(customers,t_j))
		customers[idx]['actual_detour_w_j'] = customers[idx]['actual_detour_wo_j'] + (source_detour_for_j(customers) + indicator_of(idx >= t_j)*destination_detour_for_j(customers,t_j))/customers[idx]['sd']

	
	#new customer
	if t_j == customer_j or t_j==1:
		customers[customer_j]['actual_detour_w_j'] = 0
	else:
		delta_j_j = distance(customers[customer_j]['s'],customers[1]['d'])
		for idx in range(1,t_j-1):
			delta_j_j += distance(customers[idx]['d'],customers[idx+1]['d'])
		delta_j_j += distance(customers[t_j-1]['d'],customers[customer_j]['d']) - customers[customer_j]['sd']

		delta_j_j /= customers[customer_j]['sd']

		customers[customer_j]['actual_detour_w_j'] = delta_j_j

	return customers

def opt_customer_to_drop_after_j(customers):

	'''
	notation t_j used in the paper
	this is the customer to be dropped off right after j
	is a linear search operation as shown below
	WARNING: we are not checking if the previous dropoff sequence is good.
	'''

	customer_j = len(customers)
	active_customer_idxes = active_customers_j(customers)

	cost_base = 0 #before finding a drop position for j
	for idx in active_customer_idxes:
		if idx==1:
			cost_base += distance(customers[customer_j]['s'],	customers[idx]['d'])
		else:
			cost_base += distance(customers[idx-1]['d'],customers[idx]['d'])

	# opt_route = [customer_j] + active_customer_idxes
	opt_route_cost = cost_base + customers[customer_j]['sd'] \
		+ distance(customers[customer_j]['d'],customers[1]['d']) \
		- distance(customers[customer_j]['s'],	customers[1]['d'])
	t_j = 1
	for idx in active_customer_idxes:
		'''
		insering dropping j after dropping each idx
		'''
		if idx < active_customer_idxes[-1]:
			new_route_cost = cost_base \
				+ distance(customers[idx]['d'],customers[customer_j]['d']) \
				+ distance(customers[customer_j]['d'],customers[idx+1]['d']) \
				- distance(customers[idx]['d'],	customers[idx+1]['d']) 
		else:
			new_route_cost = cost_base \
				+ distance(customers[idx]['d'],customers[customer_j]['d'])

		if new_route_cost < opt_route_cost:
			opt_route_cost = new_route_cost
			t_j = idx + 1

	return t_j

def sum_previous_customer_shared_prices(customers,start_idx):
	summed_p_s = 0
	active_customer_idxes = active_customers_j(customers)
	for idx in active_customer_idxes:
		if idx >= start_idx:
			summed_p_s += customers[idx]['p_s']
	return summed_p_s

def get_incremental_profit_adding_j(x,customers,c_op,support_v,degradation_multiplier,EEPP_coeff,t_j):

	(prob_exclusive_val,prob_pool_val,incr_profit_exclusive_val,incr_profit_pool_val,expost_penalty_sum) = get_incremental_profit_adding_j_components(x,customers,c_op,support_v,degradation_multiplier,EEPP_coeff,t_j)

	return prob_exclusive_val*incr_profit_exclusive_val + prob_pool_val*incr_profit_pool_val

def get_incremental_profit_adding_j_components(x,customers,c_op,support_v,degradation_multiplier,EEPP_coeff,t_j):
	'''
	applicability: multi-source and multi-destination
	'''

	p_x,p_s = x[0],x[1]
	customer_j = len(customers)

	prob_exclusive_val = prob_exclusive_j(p_x,p_s,customers[customer_j]['delta_bar'],support_v,customers[customer_j]['k_delta_bar'])

	prob_pool_val,tempa,tempb = prob_pool_j(p_x,p_s,customers[customer_j]['delta_bar'],support_v,customers[customer_j]['k_delta_bar'])

	incr_profit_exclusive_val = (p_x - c_op)*customers[customer_j]['sd']


	expost_penalty_sum = 0
	for idx in customers:
		expost_penalty_sum += get_incremental_penalty(x,customers,idx,degradation_multiplier,support_v)

	incr_profit_pool_val = p_s*customers[customer_j]['sd']*(1 + customers[customer_j]['actual_detour_w_j']) \
		+ (sum_previous_customer_shared_prices(customers,1)-c_op)*source_detour_for_j(customers) \
		+ (sum_previous_customer_shared_prices(customers,t_j)-c_op)*destination_detour_for_j(customers,t_j) \
		- EEPP_coeff*expost_penalty_sum



	return (prob_exclusive_val,prob_pool_val,incr_profit_exclusive_val,incr_profit_pool_val,expost_penalty_sum)

def get_incremental_penalty(x,customers,idx,degradation_multiplier,support_v):

	if idx == len(customers):
		p_x = x[0]
		p_s = x[1]
	else:
		p_x = customers[idx]['p_x']
		p_s = customers[idx]['p_s']

	delta_ijm1 = customers[idx]['actual_detour_wo_j']
	delta_ij = customers[idx]['actual_detour_w_j']

	k_delta_ijm1 = degradation(delta_ijm1,degradation_multiplier)
	k_delta_ij = degradation(delta_ij,degradation_multiplier)

	v_ubar_before_j = p_s*(1+delta_ijm1)/k_delta_ijm1
	v_ubar_after_j = p_s*(1+delta_ij)/k_delta_ij

	prob_pool_val,v_ubar,v_lbar =  prob_pool_j(p_x,p_s,customers[idx]['delta_bar'],support_v,customers[idx]['k_delta_bar'])

	if customers[idx]['is_bootstrapped'] is True:
		'''
		different from other customer's expected ex post penalties
		expectation conditioned on ex ante IR being satisfied
		'''
		v_ubar = support_v[1]

	term1ub = min(min(v_ubar_before_j,v_ubar),support_v[1])
	term1lb = max(min(v_ubar_before_j,v_lbar),support_v[0])
	term1nr = integrate.quad(lambda vvar: customers[idx]['sd']*f_v(vvar,support_v)*(k_delta_ijm1*vvar - p_s*(1 + delta_ijm1)),
			term1lb,
			term1ub)


	term2ub = min(min(v_ubar_after_j,v_ubar),support_v[1])
	term2lb = max(min(v_ubar_after_j,v_lbar),support_v[0])
	term2nr = integrate.quad(lambda vvar: customers[idx]['sd']*f_v(vvar,support_v)*(k_delta_ij*vvar - p_s*(1 + delta_ij)),
			term2lb,
			term2ub)

	# print('term1',term1nr[0],'term1',term2nr[0],'denominator',prob_pool_val)
	expected_ex_post_penalty = (term1nr[0] - term2nr[0])/(prob_pool_val + 1e-8) #HARDCODE

	# print('expected_ex_post_penalty',expected_ex_post_penalty)
	return expected_ex_post_penalty

def maximize_incremental_profit_j(params,customers):

	solver_type = params['solver_type']
	c_op =	params['c_op']
	p_x_max =	params['p_x_max']
	EEPP_coeff = params['EEPP_coeff']
	gridsearch_resolution = params['gridsearch_resolution']
	support_v = params['support_v']
	degradation_multiplier = params['degradation_multiplier']

	customer_j = len(customers)
	delta_bar = customers[customer_j]['delta_bar']
	k_delta_bar = customers[customer_j]['k_delta_bar']

	px_lb = c_op
	px_ub = p_x_max
	ps_lb = 0
	ps_ub = k_delta_bar*px_ub/(1 + delta_bar)
	initial_guess = [min(px_ub,max(px_lb,phi_v_inv(c_op))),min(ps_ub,max(ps_lb,k_delta_bar*phi_v_inv(c_op)/(1+delta_bar)))]
	assert px_lb <= initial_guess[0] <= px_ub
	assert ps_lb <= initial_guess[1] <= ps_ub

	t_j = opt_customer_to_drop_after_j(customers)
	customers = set_actual_detours_w_j(customers,opt_customer_to_drop_after_j(customers))

	profit = get_incremental_profit_adding_j(initial_guess,customers,c_op,support_v,degradation_multiplier,EEPP_coeff,t_j)

	profit_surface = None
	p_x_opt = initial_guess[0]
	p_s_opt = initial_guess[1]

	if solver_type == 'gridsearch':
		# print('\nUsing Gridsearch:')
		px_gridsearch_num = int((px_ub-px_lb)/gridsearch_resolution)
		ps_gridsearch_num = int((ps_ub - ps_lb)/gridsearch_resolution)
		px_gridvals = np.linspace(px_lb,px_ub,num=px_gridsearch_num)
		ps_gridvals = np.linspace(ps_lb,ps_ub,num=ps_gridsearch_num)
		profit_surface = np.zeros((px_gridsearch_num,ps_gridsearch_num))

		for idxx,p_x_var in enumerate(px_gridvals):
			for idxs,p_s_var in enumerate(ps_gridvals):
				if pricing_feasibility_constraint([p_x_var,p_s_var],delta_bar,k_delta_bar) >= 0:

					profit_var = get_incremental_profit_adding_j([p_x_var,p_s_var],customers,c_op,support_v,degradation_multiplier,EEPP_coeff,t_j)

					profit_surface[idxx,idxs] = profit_var
					if profit_var > profit:
						profit = profit_var
						p_x_opt = p_x_var
						p_s_opt = p_s_var
	else:
		print('NO SOLVER!')

	return (profit,{'p_x':p_x_opt,'p_s':p_s_opt},profit_surface)

#=========================================

if __name__=='__main__':

	customers = OrderedDict()
	customers[1] = {}
	customers[2] = {}
	customers[1]['s'] = np.array([0,0])
	customers[1]['d'] = np.array([2.5,0])
	customers[2]['s'] = np.array([2,0])
	customers[2]['d'] = customers[1]['d'] #np.array([2.1,0]) #
	customers[1]['p_s'] = params['p_s_1']
	customers[1]['p_x'] = params['support_v'][1]

	for idx in customers:
		customers[idx]['sd']  = distance(customers[idx]['s'],customers[idx]['d'])
		customers[idx]['delta_bar'] = params['delta_same']
		customers[idx]['k_delta_bar'] = degradation(customers[idx]['delta_bar'],params['degradation_multiplier'])
		customers[idx]['actual_detour_wo_j'] = 0
		customers[idx]['is_bootstrapped'] = False

		print('customer ',idx,': sd',customers[idx]['sd'],'delta_bar',customers[idx]['delta_bar'],'k_delta_bar',customers[idx]['k_delta_bar'])

	customers[1]['is_bootstrapped'] = True
	assert_p_s_1_greater_than_c_op(customers[1]['p_s'],params['c_op'])
	assert_ex_ante_customer1_IR(params['support_v'],customers[1]['p_s'],customers[1]['delta_bar'],customers[1]['k_delta_bar'],customers[1]['sd'])


	customer_j = len(customers)
	active_customer_idxes = active_customers_j(customers)
	print(customer_j,active_customer_idxes)
	customers = set_actual_detours_w_j(customers,opt_customer_to_drop_after_j(customers))
	print(customers)

	# print(opt_customer_to_drop_after_j(customers))

	[incremental_profit_j,prices_j,incremental_profit_j_surface] = maximize_incremental_profit_j(params,customers)
	print('Incremental profit for j:',incremental_profit_j,'prices',prices_j)


	for idx in customers:
		print(idx,'wo_j',customers[idx]['actual_detour_wo_j'],'w_j',customers[idx]['actual_detour_w_j'])

	print(get_incremental_penalty([customers[1]['p_x'],customers[1]['p_s']],customers,1,params['degradation_multiplier'],params['support_v']))