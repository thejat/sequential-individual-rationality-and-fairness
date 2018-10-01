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

def prob_exclusive_1(p_x_1,p_p_1,delta_small,s1d1,support_v,k_1):
	argument = (p_x_1 - p_p_1*(1+delta_small))/(1-k_1)
	prob_exclusive_val = 1 - F_v(argument,support_v)
	return prob_exclusive_val

def prob_pool_1(p_x_1,p_p_1,delta_small,s1d1,support_v,k_1,flag_print_arguments=False):
	argument1 = (p_x_1 - p_p_1*(1+delta_small))/(1-k_1)
	argument2 = p_p_1*(1+delta_small)/k_1
	prob_pool_val = F_v(argument1,support_v) - F_v(argument2,support_v)
	if flag_print_arguments is True:
		print('Need args between 0 and 1: ',argument1,argument2)
	return prob_pool_val

def profit_1_inner(x,delta_small,s1d1,c_op,support_v,k_1):
	p_x_1,p_p_1 = x[0],x[1]
	prob_exclusive_val = prob_exclusive_1(p_x_1,p_p_1,delta_small,s1d1,support_v,k_1)
	prob_pool_val = prob_pool_1(p_x_1,p_p_1,delta_small,s1d1,support_v,k_1)
	profit_exclusive_val = (p_x_1 - c_op)*s1d1
	profit_pool_val = (p_p_1 - c_op)*s1d1
	return (prob_exclusive_val,prob_pool_val,profit_exclusive_val,profit_pool_val)

def profit_1(x,delta_small,s1d1,c_op,support_v,k_1):
	prob_exclusive_val,prob_pool_val,profit_exclusive_val,profit_pool_val = profit_1_inner(x,delta_small,s1d1,c_op,support_v,k_1)
	return prob_exclusive_val*profit_exclusive_val + prob_pool_val*profit_pool_val

def pricing_feasibility_constraint(x,delta_small,sidi,k_i):
	p_x_i,p_p_i = x[0],x[1]
	return k_i*p_x_i - p_p_i*(1 + delta_i_max(delta_small,sidi)/sidi) 


def maximise_profit_1(solver_type,c_op,p_max,gridsearch_resolution,delta_small,s1d1,support_v,k_1):

	px_lb = c_op
	px_ub = p_max
	pp_lb = 0
	pp_ub = k_1*p_max/(1 + delta_i_max(delta_small,s1d1)/s1d1)
	p_x_opt,p_p_opt = (1+c_op)/2, k_1*(1+c_op)/2
	initial_guess = [p_x_opt,p_p_opt]
	print('initial_guess',initial_guess)
	profit = profit_1(initial_guess,delta_small,s1d1,c_op,support_v,k_1)
	profit_surface = None


	if solver_type == 'gridsearch':
		print('\nUsing Gridsearch:')
		px_gridsearch_num = int((p_max-c_op)/gridsearch_resolution)
		pp_gridsearch_num = int((pp_ub - pp_lb)/gridsearch_resolution)
		profit_surface = np.zeros((px_gridsearch_num,pp_gridsearch_num))
		px_gridvals = np.linspace(px_lb,px_ub,num=px_gridsearch_num)
		pp_gridvals = np.linspace(pp_lb,pp_ub,num=pp_gridsearch_num)

		profit = profit_1(initial_guess,delta_small,s1d1,c_op,support_v,k_1)
		for idxx,p_x_var in enumerate(px_gridvals):
			for idxp,p_p_var in enumerate(pp_gridvals):
				if pricing_feasibility_constraint([p_x_var,p_p_var],delta_small,s1d1,k_1) >= 0:
					profit_var = profit_1([p_x_var,p_p_var],delta_small,s1d1,c_op,support_v,k_1)
					profit_surface[idxx,idxp] = profit_var
					# print('constraint satisfied',profit_surface[idxx,idxp])
					if profit_var > profit:
						profit = profit_var
						p_x_opt = p_x_var
						p_p_opt = p_p_var

	# elif solver_type == 'slsqp': #use a solver
	# 	print('\nUsing Solver:')
	# 	def profit_1_neg(x,delta_small,s1d1,c_op,support_v,k_1):
	# 		return -1*profit_1(x,delta_small,s1d1,c_op,support_v,k_1)
	# 	bnds = ((px_lb, px_ub), (pp_lb, pp_ub))


	# 	cons = ({'type': 'ineq', 
	# 			'fun': lambda x,delta_small,s1d1: pricing_feasibility_constraint(x,delta_small,s1d1,k_1),
	# 	 		'args': (delta_small,s1d1)})
	# 	res = minimize(profit_1_neg,
	# 			initial_guess,
	# 			args=(delta_small,s1d1,c_op,support_v,k_1),
	# 			method='SLSQP', 
	# 			bounds=bnds, 
	# 			constraints=cons)
	# 			#options={'maxiter': 50000, 'ftol': 1e-07,'disp': False, 'eps': 1.4901161193847656e-08})
	# 	# https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html#optimize-minimize-slsqp
	# 	# https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#tutorial-sqlsp

	# 	# linear_constraint_trust_constr = LinearConstraint([[k_1, -(1 + delta_i_max(delta_small,s1d1)/s1d1)]], [0], [np.inf])
	# 	# res = minimize(profit_1_neg, 
	# 	# 		initial_guess, 
	# 	# 		args=(delta_small,s1d1,c_op,support_v,k_1),
	# 	# 		method='trust-constr', 
	# 	# 		constraints=[linear_constraint_trust_constr],
	# 	# 		options={'verbose': 1}, 
	# 	# 		bounds=bnds)

	# 	p_x_opt 	= res.x[0]
	# 	p_p_opt 	= res.x[1]
	# 	profit 		= -1*res.fun

	elif solver_type=='closed_form':
		print('\nUsing Closed-form solution:')
		p_x_opt = (1+ delta_small)*(c_op*(2+delta_small) + 2)/(4*(1+delta_small) - delta_small*delta_small*k_1/(1-k_1))
		p_p_opt = (k_1*(delta_small+2) + 2*c_op*(delta_small+1))/(4*(1+delta_small) - delta_small*delta_small*k_1/(1-k_1))
		profit = profit_1([p_x_opt,p_p_opt],delta_small,s1d1,c_op,support_v,k_1)


	# print(np.max(profit_surface))
	# print(profit_surface)

	return [profit,{'p_x':p_x_opt,'p_p':p_p_opt},profit_surface]


if __name__=='__main__':

	s1d1  = distance(params['s1'],params['d']) #the drive by distance between customer 1 and their destination
	k_1 = degradation(delta_i_max(params['delta_small'],s1d1),s1d1)

	print('s1d1',s1d1,'delta_1_max',delta_i_max(params['delta_small'],s1d1),'k_1',k_1)

	for solver_type in ['closed_form','gridsearch']:
		params['solver_type'] = solver_type

		[profit,prices,profit_surface] = maximise_profit_1(
			params['solver_type'],
			params['c_op'],
			params['p_max'],
			params['gridsearch_resolution'],
			params['delta_small'],
			s1d1,
			params['support_v'],
			k_1)
		# print(profit_surface)

		print('profit',profit,'prices',prices)
		print('pricing constraint should be positive: ',pricing_feasibility_constraint([prices['p_x'],prices['p_p']],params['delta_small'],s1d1,k_1))	

		prob_exclusive_val,prob_pool_val,profit_exclusive_val,profit_pool_val = profit_1_inner([prices['p_x'],prices['p_p']],params['delta_small'],s1d1,params['c_op'],params['support_v'],k_1)
		print('prob exclusive: ',prob_exclusive_val,'prob pool: ',prob_pool_val,'profit exclusive: ',profit_exclusive_val,'profit pool: ',profit_pool_val)

		temp = prob_pool_1(prices['p_x'],prices['p_p'],params['delta_small'],s1d1,params['support_v'],k_1,flag_print_arguments=True)
