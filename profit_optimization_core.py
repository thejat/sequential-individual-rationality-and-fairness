# Accounting for Expected Ex Post IR
'''
This file has the core function that maximizes profit of the firm in a two rider ride-sharing setting.

Author: Theja Tulabandhula
Year: 2018

'''
import numpy as np 
import math, time
from scipy.optimize import minimize, LinearConstraint
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

def degradation(delta_i,sidi):
	degradation_coeff = 1 - delta_i/(1.1*sidi)
	return degradation_coeff

def delta_i_max(delta_small,sidi):
	'''
	Design choice by the firm to promise delta_i_max as a scaled version of sidi
	'''
	return delta_small*sidi

def prob_exclusive_1(p_x_1,p_p_1,delta_small,s1d1,support_v,k):
	argument = (p_x_1 - p_p_1*(1+delta_small))/(1-k)
	prob_exclusive_val = 1 - F_v(argument,support_v)
	return prob_exclusive_val

def prob_pool_1(p_x_1,p_p_1,delta_small,s1d1,support_v,k,flag_print_arguments=False):
	argument1 = (p_x_1 - p_p_1*(1+delta_small))/(1-k)
	argument2 = p_p_1*(1+delta_small)/k
	prob_pool_val = F_v(argument1,support_v) - F_v(argument2,support_v)
	if flag_print_arguments is True:
		print('Need args between 0 and 1: ',argument1,argument2)
	return prob_pool_val

def profit_1_inner(x,delta_small,s1d1,c_op,support_v,k):
	p_x_1,p_p_1 = x[0],x[1]
	prob_exclusive_val = prob_exclusive_1(p_x_1,p_p_1,delta_small,s1d1,support_v,k)
	prob_pool_val = prob_pool_1(p_x_1,p_p_1,delta_small,s1d1,support_v,k)
	profit_exclusive_val = (p_x_1 - c_op)*s1d1
	profit_pool_val = (p_p_1 - c_op)*s1d1
	return (prob_exclusive_val,prob_pool_val,profit_exclusive_val,profit_pool_val)

def profit_1(x,delta_small,s1d1,c_op,support_v,k):
	prob_exclusive_val,prob_pool_val,profit_exclusive_val,profit_pool_val = profit_1_inner(x,delta_small,s1d1,c_op,support_v,k)
	return prob_exclusive_val*profit_exclusive_val + prob_pool_val*profit_pool_val

def pricing_feasibility_constraint(x,delta_small,sidi):
	p_x_i,p_p_i = x[0],x[1]
	return degradation(delta_i_max(delta_small,sidi),sidi)*p_x_i - p_p_i*(1 + delta_i_max(delta_small,sidi)/sidi) 


def maximise_profit_1(solver_type,c_op,p_max,gridsearch_resolution,delta_small,s1d1,support_v):

	k = degradation(delta_i_max(delta_small,s1d1),s1d1)
	px_lb = c_op
	px_ub = p_max
	pp_lb = 0
	pp_ub = k*p_max/(1 + delta_i_max(delta_small,s1d1)/s1d1)
	initial_guess = [px_ub, pp_lb]
	profit, p_x_opt,p_p_opt = -1,-1,-1

	if solver_type == 'gridsearch':
		print('\nUsing Gridsearch:')
		px_gridsearch_num = int((p_max-c_op)/gridsearch_resolution)
		pp_gridsearch_num = int((pp_ub - pp_lb)/gridsearch_resolution)
		px_gridvals = np.linspace(px_lb,px_ub,num=px_gridsearch_num)
		pp_gridvals = np.linspace(pp_lb,pp_ub,num=pp_gridsearch_num)

		profit = profit_1(initial_guess,delta_small,s1d1,c_op,support_v,k)
		for p_x_var in px_gridvals:
			for p_p_var in pp_gridvals:
				if pricing_feasibility_constraint([p_x_var,p_p_var],delta_small,s1d1) >= 0:
					profit_var = profit_1([p_x_var,p_p_var],delta_small,s1d1,c_op,support_v,k)
					if profit_var > profit:
						profit = profit_var
						p_x_opt = p_x_var
						p_p_opt = p_p_var
	elif solver_type == 'slsqp': #use a solver
		print('\nUsing Solver:')
		def profit_1_neg(x,delta_small,s1d1,c_op,support_v,k):
			return -1*profit_1(x,delta_small,s1d1,c_op,support_v,k)
		bnds = ((px_lb, px_ub), (pp_lb, pp_ub))


		cons = ({'type': 'ineq', 
				'fun': lambda x,delta_small,s1d1: pricing_feasibility_constraint(x,delta_small,s1d1),
		 		'args': (delta_small,s1d1)})
		res = minimize(profit_1_neg,
				initial_guess,
				args=(delta_small,s1d1,c_op,support_v,k),
				method='SLSQP', 
				bounds=bnds, 
				constraints=cons)
				#options={'maxiter': 50000, 'ftol': 1e-07,'disp': False, 'eps': 1.4901161193847656e-08})
		# https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html#optimize-minimize-slsqp
		# https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#tutorial-sqlsp

		# linear_constraint_trust_constr = LinearConstraint([[k, -(1 + delta_i_max(delta_small,s1d1)/s1d1)]], [0], [np.inf])
		# res = minimize(profit_1_neg, 
		# 		initial_guess, 
		# 		args=(delta_small,s1d1,c_op,support_v,k),
		# 		method='trust-constr', 
		# 		constraints=[linear_constraint_trust_constr],
		# 		options={'verbose': 1}, 
		# 		bounds=bnds)



		p_x_opt 	= res.x[0]
		p_p_opt 	= res.x[1]
		profit 		= -1*res.fun

	elif solver_type=='closed_form':
		print('\nUsing Closed-form solution:')
		p_x_opt = (1+ delta_small)*(c_op*(2+delta_small) + 2)/(4*(1+delta_small) - delta_small*delta_small*k/(1-k))
		p_p_opt = (k*(delta_small+2) + 2*c_op*(delta_small+1))/(4*(1+delta_small) - delta_small*delta_small*k/(1-k))
		profit = profit_1([p_x_opt,p_p_opt],delta_small,s1d1,c_op,support_v,k)



	return profit,{'p_x':p_x_opt,'p_p':p_p_opt}


if __name__=='__main__':


	s2   = np.array([1,1])
	s1s2 = distance(params['s1'],s2) #the drive by distance between customer 1 and customer 2
	s2d2  = distance(s2,params['d']) #the drive by distance between customer 2 and their destination
	s1d1  = distance(params['s1'],params['d']) #the drive by distance between customer 1 and their destination


	print('s1d1',s1d1,'delta_1_max',delta_i_max(params['delta_small'],s1d1))
	print('k',degradation(delta_i_max(params['delta_small'],s1d1),s1d1))

	for solver_type in ['gridsearch','closed_form']:
		params['solver_type'] = solver_type

		profit,prices = maximise_profit_1(
			params['solver_type'],
			params['c_op'],
			params['p_max'],
			params['gridsearch_resolution'],
			params['delta_small'],
			s1d1,
			params['support_v'])

		print('profit',profit,'prices',prices)
		print('pricing constraint should be positive: ',pricing_feasibility_constraint([prices['p_x'],prices['p_p']],params['delta_small'],s1d1))	

		prob_exclusive_val,prob_pool_val,profit_exclusive_val,profit_pool_val = profit_1_inner([prices['p_x'],prices['p_p']],params['delta_small'],s1d1,params['c_op'],params['support_v'],degradation(delta_i_max(params['delta_small'],s1d1),s1d1))
		print('prob exclusive: ',prob_exclusive_val,'prob pool: ',prob_pool_val,'profit exclusive: ',profit_exclusive_val,'profit pool: ',profit_pool_val)

		temp = prob_pool_1(prices['p_x'],prices['p_p'],params['delta_small'],s1d1,params['support_v'],degradation(delta_i_max(params['delta_small'],s1d1),s1d1),flag_print_arguments=True)
