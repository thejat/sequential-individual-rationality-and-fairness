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

def phi_v_inv(y,support_v):
	'''
	Inverse function of virtual valuation for the uniform distribution. 
	See https://en.wikipedia.org/wiki/Regular_distribution_(economics)
	'''
	if y > support_v[1] or y < 2*support_v[0] - support_v[1]:
		return None
	return 0.5*(support_v[1]+y)

def phi(v,support_v):
	if v is None or f_v(v,support_v) <= 0:
		return -np.inf
	return v - (1-F_v(v,support_v))/f_v(v,support_v)

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

def degradation(delta_i,degradation_multiplier,k_bar):
	'''
	delta_i is essentially a variable and this function needs to be reevaluated everytime
	'''
	# k_bar = .88
	degradation_coeff = k_bar - delta_i/degradation_multiplier
	return degradation_coeff

def assert_p_s_1_greater_than_c_op(p_s_1,c_op,s1d1):
	'''
	assuming they are bootstrapped. otherwise we don't need shared price to be greater than c_op
	'''
	assert p_s_1 >= c_op*s1d1

def assert_ex_ante_customer1_IR(support_v,p_s_1,delta_bar,k_delta_bar,s1d1):
	#customer 1's ex ante IR constraint
	ir_1_ante = s1d1*k_delta_bar*support_v[1] - p_s_1
	print('Ex ante customer 1 IR should be nonneg for support_v[1]:',ir_1_ante)
	assert ir_1_ante > 0

def active_customers_j(customers):
	'''
	assume all customers before j are active. i.e., 1.2,...,j-1
	'''
	return sorted(customers.keys())[:-1]

def pricing_feasibility_constraint(x,k_delta_bar):
	'''
	we need the computed value to be positive for the customer to even consider sharing. when its zero, p_s does not matter, so covers the exclusive vs declined case.
	'''
	p_x_var,p_s_var = x[0],x[1]
	return k_delta_bar*p_x_var - p_s_var

def prob_exclusive_j(p_x,p_s,sidi,support_v,k_delta_bar):
	'''
	applicability: multi-source and multi-destination
	'''
	v_ubar = (p_x - p_s)/((1-k_delta_bar)*sidi)
	prob_exclusive_val = 1 - F_v(v_ubar,support_v)
	return prob_exclusive_val

def prob_pool_j(p_x,p_s,sidi,support_v,k_delta_bar,flag_print_arguments=False):
	'''
	applicability: multi-source and multi-destination
	'''
	v_ubar = (p_x - p_s)/((1-k_delta_bar)*sidi)
	v_lbar = p_s/(k_delta_bar*sidi)

	# print('v_ubar',v_ubar)
	# print('v_lbar',v_lbar)

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
	'''
	t_j_minus is 0
	t_j_minus + 1 is customer 1
	D_0 is given by location_from_which_detour_starts
	'''

	customer_j = len(customers)
	active_customer_idxes = active_customers_j(customers)

	location_from_which_detour_starts = customers[last_customer_picked_up_or_dropped_off(active_customer_idxes)]['s']
	location_next_customer_drop = customers[1]['d']

	# print('loc detour starts',location_from_which_detour_starts)
	# print('loc next cust drop', location_next_customer_drop)

	source_detour_val = distance(location_from_which_detour_starts,customers[customer_j]['s']) + distance(customers[customer_j]['s'],location_next_customer_drop) - distance(location_from_which_detour_starts,location_next_customer_drop)
	return source_detour_val

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

	elif t_j > 1 and t_j < customer_j:
		destination_detour_val = distance(customers[t_j-1]['d'],customers[customer_j]['d']) \
								+ distance(customers[customer_j]['d'],customers[t_j]['d']) \
								- distance(customers[t_j]['d'],customers[t_j-1]['d'])
	else:
		destination_detour_val = distance(customers[active_customer_idxes[-1]]['d'],customers[customer_j]['d'])

	return destination_detour_val

def set_actual_detours_w_j(customers,t_j):

	customer_j = len(customers)
	active_customer_idxes = active_customers_j(customers)

	t_j_minus = 0 #TBD HARDCODED

	for idx in active_customer_idxes:
		# print('source detour',source_detour_for_j(customers))
		# print('dest detour',destination_detour_for_j(customers,t_j))
		customers[idx]['actual_detour_w_j'] = customers[idx]['actual_detour_wo_j'] + (indicator_of(idx > t_j_minus)*source_detour_for_j(customers) + indicator_of(idx >= t_j)*destination_detour_for_j(customers,t_j))/customers[idx]['sd']

	
	#new customer
	if t_j==1:
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
	notation t_j_plus used in the paper
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

	# print('returned t_j',t_j)
	return t_j,None #TODO: return the new route data structure

def get_incremental_profit_adding_j(x,customers,c_op,support_v,degradation_multiplier,EEPP_coeff,t_j,k_bar):

	# print('profit eval AA t_j',t_j)

	(prob_exclusive_val,prob_pool_val,incr_profit_exclusive_val,incr_profit_pool_val,expost_penalty_sum) = get_incremental_profit_adding_j_components(x,customers,c_op,support_v,degradation_multiplier,EEPP_coeff,t_j,k_bar)

	return prob_exclusive_val*incr_profit_exclusive_val + prob_pool_val*incr_profit_pool_val

def get_incremental_profit_adding_j_components(x,customers,c_op,support_v,degradation_multiplier,EEPP_coeff,t_j,k_bar):
	'''
	applicability: multi-source and multi-destination
	'''

	p_x,p_s = x[0],x[1]
	customer_j = len(customers)
	active_customer_idxes = active_customers_j(customers)

	prob_exclusive_val = prob_exclusive_j(p_x,p_s,customers[customer_j]['sd'],support_v,customers[customer_j]['k_delta_bar'])

	# print('DEBUG: call in profit adding j')
	prob_pool_val,tempa,tempb = prob_pool_j(p_x,p_s,customers[customer_j]['sd'],support_v,customers[customer_j]['k_delta_bar'])

	incr_profit_exclusive_val = p_x - c_op*customers[customer_j]['sd']

	expost_penalty_sum = 0
	for idx in active_customer_idxes:
		expost_penalty_sum += get_incremental_penalty(x,customers,idx,degradation_multiplier,support_v,k_bar)

	incr_profit_pool_val = p_s \
		- c_op*(source_detour_for_j(customers) + destination_detour_for_j(customers,t_j)) \
		- EEPP_coeff*expost_penalty_sum

	return (prob_exclusive_val,prob_pool_val,incr_profit_exclusive_val,incr_profit_pool_val,expost_penalty_sum)

def get_incremental_penalty(x,customers,idx,degradation_multiplier,support_v,k_bar):

	if idx == len(customers):
		p_x = x[0]
		p_s = x[1]
	else:
		p_x = customers[idx]['p_x']
		p_s = customers[idx]['p_s']

	delta_ijm1 = customers[idx]['actual_detour_wo_j']
	delta_ij = customers[idx]['actual_detour_w_j']

	k_delta_ijm1 = degradation(delta_ijm1,degradation_multiplier,k_bar)
	k_delta_ij = degradation(delta_ij,degradation_multiplier,k_bar)

	v_ubar_before_j = p_s/(k_delta_ijm1*customers[idx]['sd'])
	v_ubar_after_j = p_s/(k_delta_ij*customers[idx]['sd'])

	# print('DEBUG: call in incremental penalty. px',p_x,' p_s',p_s)
	prob_pool_val,v_ubar,v_lbar =  prob_pool_j(p_x,p_s,customers[idx]['sd'],support_v,customers[idx]['k_delta_bar'])

	if customers[idx]['is_bootstrapped'] is True:
		'''
		different from other customer's expected ex post penalties
		expectation conditioned on ex ante IR being satisfied
		'''
		v_ubar = support_v[1]

	term1ub = min(min(v_ubar_before_j,v_ubar),support_v[1])
	term1lb = max(min(v_ubar_before_j,v_lbar),support_v[0])
	term1nr = integrate.quad(lambda vvar: f_v(vvar,support_v)*(k_delta_ijm1*vvar*customers[idx]['sd'] - p_s),
			term1lb,
			term1ub)


	term2ub = min(min(v_ubar_after_j,v_ubar),support_v[1])
	term2lb = max(min(v_ubar_after_j,v_lbar),support_v[0])
	term2nr = integrate.quad(lambda vvar: f_v(vvar,support_v)*(k_delta_ij*vvar*customers[idx]['sd'] - p_s),
			term2lb,
			term2ub)

	# print('term1',term1nr[0],'term1',term2nr[0],'denominator',prob_pool_val)
	expected_ex_post_penalty = (term1nr[0] - term2nr[0])/(prob_pool_val + 1e-8) #HARDCODE

	# print('expected_ex_post_penalty',expected_ex_post_penalty)
	return expected_ex_post_penalty

def maximize_incremental_profit_j(params,customers):

	customer_j = len(customers)
	active_customer_idxes = active_customers_j(customers)
	k_delta_bar = customers[customer_j]['k_delta_bar']

	solver_type = params['solver_type']
	c_op =	params['c_op']
	p_x_max =	params['p_x_max_per_mile']*customers[customer_j]['sd']
	EEPP_coeff = params['EEPP_coeff']
	gridsearch_num = params['gridsearch_num']
	support_v = params['support_v']
	degradation_multiplier = params['degradation_multiplier']
	k_bar = params['k_bar']

	# print('p_x_max',p_x_max)

	px_lb = c_op*customers[customer_j]['sd']
	px_ub = p_x_max
	ps_lb = 0
	ps_ub = k_delta_bar*px_ub
	initial_guess = [px_ub,ps_ub]
	assert px_lb <= initial_guess[0] <= px_ub
	assert ps_lb <= initial_guess[1] <= ps_ub

	t_j,temp_route = opt_customer_to_drop_after_j(customers)
	# print('t_j',t_j)
	customers = set_actual_detours_w_j(customers,t_j)

	profit = get_incremental_profit_adding_j(initial_guess,customers,c_op,support_v,degradation_multiplier,EEPP_coeff,t_j,k_bar)

	profit_surface = None
	p_x_opt = initial_guess[0]
	p_s_opt = initial_guess[1]

	if solver_type == 'gridsearch':
		# print('\nUsing Gridsearch:')
		px_gridvals = np.linspace(px_lb,px_ub,num=gridsearch_num)
		ps_gridvals = np.linspace(ps_lb,ps_ub,num=gridsearch_num)
		# print('px_gridvals normalized',px_gridvals/customers[customer_j]['sd'])
		# print('ps_gridvals normalized',ps_gridvals/customers[customer_j]['sd'])
		profit_surface = np.zeros((gridsearch_num,gridsearch_num))

		for idxx,p_x_var in enumerate(px_gridvals):
			for idxs,p_s_var in enumerate(ps_gridvals):
				if pricing_feasibility_constraint([p_x_var,p_s_var],k_delta_bar) >= 0:

					profit_var = get_incremental_profit_adding_j([p_x_var,p_s_var],customers,c_op,support_v,degradation_multiplier,EEPP_coeff,t_j,k_bar)

					profit_surface[idxx,idxs] = profit_var
					if profit_var > profit:
						profit = profit_var
						p_x_opt = p_x_var
						p_s_opt = p_s_var
	elif solver_type == 'closed_form':

		#Solve for v_ubar_opt
		expost_penalty_sum = 0
		for idx in active_customer_idxes:
			expost_penalty_sum += get_incremental_penalty([None,None],customers,idx,degradation_multiplier,support_v,k_bar)

		temp_val_nr = c_op*(customers[customer_j]['sd'] - (source_detour_for_j(customers) + destination_detour_for_j(customers,t_j))) - EEPP_coeff*expost_penalty_sum
		temp_val_dr = (1 - k_delta_bar)*customers[customer_j]['sd']
		
		temp_threshold = temp_val_nr/temp_val_dr

		if temp_threshold >= support_v[1]:
			v_ubar_opt = support_v[1]
			# print('v_ubar_opt clipped to ', v_ubar_opt)
		elif temp_threshold <= 2*support_v[0] - support_v[1]:
			v_ubar_opt = support_v[0]
			# print('v_ubar_opt clipped to ', v_ubar_opt)
		else:
			# print('temp_threshold',temp_threshold)
			v_ubar_opt = phi_v_inv(temp_threshold,support_v)

		# print('v_ubar_opt',v_ubar_opt)

		# RHS of solution validity

		temp_val_nr = c_op*(source_detour_for_j(customers) + destination_detour_for_j(customers,t_j)) + EEPP_coeff*expost_penalty_sum
		temp_val_dr = k_delta_bar*customers[customer_j]['sd']

		temp_threshold = temp_val_nr/temp_val_dr


		if phi(v_ubar_opt,support_v) <= temp_threshold:
			#solution at boundary or exterior, no shared ride
			p_x_opt = v_ubar_opt*customers[customer_j]['sd']
			p_s_opt = k_delta_bar*p_x_opt
		else:
			v_lbar_opt = phi_v_inv(temp_threshold,support_v)
			p_s_opt = v_lbar_opt*k_delta_bar*customers[customer_j]['sd']
			p_x_opt = p_s_opt + v_ubar_opt*(1 - k_delta_bar)*customers[customer_j]['sd']

		profit = get_incremental_profit_adding_j([p_x_opt,p_s_opt],customers,c_op,support_v,degradation_multiplier,EEPP_coeff,t_j,k_bar)
	else:
		print('NO SOLVER!')

	return (profit,{'p_x':p_x_opt,'p_s':p_s_opt},profit_surface)

def solve_for_customer_j_wrapper(customers,params):
	customer_j = len(customers)
	active_customer_idxes = active_customers_j(customers)
	t_j,temp_route = opt_customer_to_drop_after_j(customers)
	customers = set_actual_detours_w_j(customers,t_j)
	[incremental_profit_j,prices_j,incremental_profit_j_surface] = maximize_incremental_profit_j(params,customers)

	print('customer_j',customer_j,'active_customer_idxes',active_customer_idxes)
	for idx in customers:
		print('customer ',idx,': sd',customers[idx]['sd'],'delta_bar',customers[idx]['delta_bar'],'k_delta_bar',customers[idx]['k_delta_bar'],'wo_j',customers[idx]['actual_detour_wo_j'],'w_j',customers[idx]['actual_detour_w_j'])
	print('t_j',t_j,'temp_route',temp_route)
	# print(customers)
	print('Incremental profit for j:',incremental_profit_j,'prices',prices_j)


	(prob_exclusive_val,prob_pool_val,incr_profit_exclusive_val,incr_profit_pool_val,expost_penalty_sum) = get_incremental_profit_adding_j_components([prices_j['p_x'],prices_j['p_s']],customers,params['c_op'],params['support_v'],params['degradation_multiplier'],params['EEPP_coeff'],t_j,params['k_bar'])
	print('prbx,probp,incrpex,incpp,expp',prob_exclusive_val,prob_pool_val,incr_profit_exclusive_val,incr_profit_pool_val,expost_penalty_sum)
	print('scaled: p_x',prices_j['p_x']/customers[customer_j]['sd'],'p_s',prices_j['p_s']/customers[customer_j]['sd'])

	return prices_j


def update_customer_information_sssd(customers,prices_j):

	customers[1]['actual_detour_wo_j'] = customers[1]['actual_detour_w_j']
	customers[2]['actual_detour_wo_j'] = customers[2]['actual_detour_w_j']

	customers[2]['p_x'] = prices_j['p_x']
	customers[2]['p_s'] = prices_j['p_s']

	del customers[1]['actual_detour_w_j']
	del customers[2]['actual_detour_w_j']

	return customers


#=========================================

if __name__=='__main__':


	params['scenario'] = 'sssd'
	# params['solver_type'] = 'closed_form'

	print('Run scenario: ',params['scenario'])
	print('Run solver type', params['solver_type'])

	#Initialize customer 1
	customers = OrderedDict()
	customers[1] = {}
	customers[1]['s'] = np.array([0,0])
	customers[1]['d'] = np.array([2.5,0])
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

	#Initialize customer 2
	customers[2] = {}	
	customers[2]['s'] = np.array([.5,.5])
	if params['scenario']=='ssd' or params['scenario']=='sssd':
		customers[2]['d'] = customers[1]['d']
	elif params['scenario']=='sdsd':
		customers[2]['d'] = np.array([2,-.5])
	customers[2]['sd']  = distance(customers[2]['s'],customers[2]['d'])
	customers[2]['delta_bar'] = params['delta_same']
	customers[2]['k_delta_bar'] = degradation(customers[2]['delta_bar'],params['degradation_multiplier'],params['k_bar'])
	customers[2]['actual_detour_wo_j'] = 0
	customers[2]['is_bootstrapped'] = False



	#Solving for prices for customer 2
	prices_j = solve_for_customer_j_wrapper(customers,params)

	print(customers)

	if params['scenario']=='sssd':

		customers = update_customer_information_sssd(customers,prices_j)
		print(customers)

		#Initialize customer 3		
		customers[3] = {}
		customers[3]['s'] = np.array([1.7,-.3])
		customers[3]['d'] = customers[1]['d']
		customers[3]['sd']  = distance(customers[3]['s'],customers[3]['d'])
		customers[3]['delta_bar'] = params['delta_same']
		customers[3]['k_delta_bar'] = degradation(customers[3]['delta_bar'],params['degradation_multiplier'],params['k_bar'])
		customers[3]['actual_detour_wo_j'] = 0
		customers[3]['is_bootstrapped'] = False


		solve_for_customer_j_wrapper(customers,params)

