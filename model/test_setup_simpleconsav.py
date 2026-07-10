import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # without this Python may crash when plotting from matplotlib
import numpy as np
import torch 
import matplotlib.pyplot as plt

# %% CELL

from consav.quadrature import log_normal_gauss_hermite

# %% CELL

import EconDLSolvers

# %% CELL

class Model(EconDLSolvers.DLSolverClass): # inherit from DLSolverClass

    # setup and allocate
    def setup(self): pass
    def allocate(self): pass
    def setup_train(self): pass
    def allocate_train(self): pass

    # draw
    def draw_initial_states(self): pass
    def draw_shocks(self): pass

    # transition
    def outcomes(self): pass # states, actions -> (intermediary) outcomes
    def state_trans_pd(self): pass # states, actions, outcomes -> post-decision states
    def state_trans(self): pass # post-decision states, shocks -> next-period states
    def terminal_actions(self): pass # action in last period = zero savings rate = consume everything

    # reward
    def reward(self): pass # utility
    terminal_reward_pd = EconDLSolvers.terminal_reward_pd # default is just 0
    discount_factor = EconDLSolvers.discount_factor # default is just par.beta

    # exploration
    draw_exploration_shocks = EconDLSolvers.draw_exploration_shocks # default is normal(0,epsilon_sigma)
    exploration = EconDLSolvers.exploration # default is action + eps (clipping is also imposed)

    # FOC (only used in DeepFOC)
    def eval_equations_FOC(self): pass

# %% CELL

def setup(model):
    """ choose parameters """

    # a. unpack
    par = model.par
    sim = model.sim

    # b. seed
    par.seed = 1 # seed for random number generator

    # c. model
    par.T = 3 # number of periods

    # d. preferences
    par.beta = 1.0/1.04 # discount factor
    
    # e. income process
    par.kappa = 0.2 # income scale
    par.sigma_psi = 0.3 # shock, std

    # f. assets
    par.r = 0.02 # return rate

    # g. initial states
    par.mu_m0 = 1.0 # initial cash-on-hand, mean
    par.sigma_m0 = 0.1 # initial cash-on-hand, std

    # h. simulation
    sim.N = 10_000 # number of agents

    # i. states
    par.Nstates = 1 # number of states
    par.Nstates_pd = 1 # number of post decision states

    # j. number of actions and outcomes
    par.Nactions = 1 # number of actions, here the savings rate
    par.Noutcomes = 1 # number of outcomes, here consumption

    # k. number of shocks
    par.Nshocks = 1 # psi is the only shock
    par.Npsi = 4 # number of quadrature points - not used in DeepSimulate

Model.setup = setup

# %% CELL

def allocate(model):
    """ allocate arrays  """

    # a. unpack
    par = model.par
    sim = model.sim
    train = model.train
    dtype = train.dtype
    device = train.device        

    # b. quadrature (not used in DeepSimulate)
    par.psi,par.psi_w = log_normal_gauss_hermite(par.sigma_psi,par.Npsi) # returns arrays
    par.psi = torch.tensor(par.psi,dtype=dtype,device=device) # convert to tensor
    par.psi_w = torch.tensor(par.psi_w,dtype=dtype,device=device) # convert to tensor

    # c. simulation (same across models)
    sim.states = torch.zeros((par.T,sim.N,par.Nstates),dtype=dtype,device=device) # state-vector
    sim.states_pd = torch.zeros((par.T,sim.N,par.Nstates_pd),dtype=dtype,device=device) # post-decision state vector
    sim.shocks = torch.zeros((par.T,sim.N,par.Nshocks),dtype=dtype,device=device) # shock-vector
    sim.outcomes = torch.zeros((par.T,sim.N,par.Noutcomes),dtype=dtype,device=device) # outcomes tensor
    sim.actions = torch.zeros((par.T,sim.N,par.Nactions),dtype=dtype,device=device)  # actions tensor
    sim.reward = torch.zeros((par.T,sim.N),dtype=dtype,device=device) # tensor for utility rewards

    sim.R = np.nan # initialize average discounted utility

Model.allocate = allocate

# %% CELL

def setup_train(model):
    """ default parameters for training """
    
    # a. unpack
    par = model.par
    train = model.train
    dtype = train.dtype
    device = train.device

    # b. neural network
    train.Nneurons_policy = np.array([50,50]) # number of neurons in hidden layers
    train.Nneurons_value = np.array([50,50]) # number of neurons in hidden layers (only used in DeepVPD)
    
    # c. policy activation functions and clipping
    if train.algoname == 'DeepFOC':
        train.policy_activation_final = ['sigmoid','softplus'] # actions are savings rate in [0,1] and positive multiplier

        train.min_actions = torch.tensor([0.0, 0.0],dtype=dtype,device=device) # minimum action value
        train.max_actions = torch.tensor([0.9999, np.inf],dtype=dtype,device=device) # maximum action value		

    else:
        train.policy_activation_final = ['sigmoid'] # action is savings rate in [0,1]

        train.min_actions = torch.tensor([0.0],dtype=dtype,device=device) # minimum action value
        train.max_actions = torch.tensor([0.9999],dtype=dtype,device=device) # maximum action value				
    
    # d. exploration
    train.epsilon_sigma = torch.tensor([0.05]) # std for exploration shocks
    train.epsilon_sigma_decay = 1.0 # decay rate for epsilon_sigma
    train.epsilon_sigma_min = torch.tensor([0.0]) # minimum value for epsilon if decay is used
    train.explore_frac = torch.tensor([0.5*(1-t/(par.T-1)) for t in range(par.T)],dtype=dtype,device=device) # fraction of agents that explore in DeepSimulate
    
    # e. misc
    train.terminal_actions_known = True # not used in DeepSimulate
    train.only_time_termination = True
    train.K_time = 0.5 # run time in minutes

Model.setup_train = setup_train

# %% CELL

def allocate_train(model):
    """ allocate memory training """

    # a. unpack
    par = model.par
    train = model.train
    dtype = train.dtype
    device = train.device

    # b. training samples (same across models)
    train.states = torch.zeros((par.T,train.N,par.Nstates),dtype=dtype,device=device)
    train.states_pd = torch.zeros((par.T,train.N,par.Nstates_pd),dtype=dtype,device=device)
    train.shocks = torch.zeros((par.T,train.N,par.Nshocks),dtype=dtype,device=device)
    train.outcomes = torch.zeros((par.T,train.N,par.Noutcomes),dtype=dtype,device=device)
    train.actions = torch.zeros((par.T,train.N,par.Nactions),dtype=dtype,device=device)
    train.reward = torch.zeros((par.T,train.N),dtype=dtype,device=device)

Model.allocate_train = allocate_train

# %% CELL

def draw_initial_states(model,N,training=False):
    """ draw initial state (m,p,t) """

    # a. unpack
    par = model.par
    sigma_m0 = par.sigma_m0

    # b. draw cash-on-hand
    m0 = par.mu_m0*torch.exp(torch.normal(-0.5*sigma_m0**2,sigma_m0,size=(N,)))
 
    # c. store
    return torch.stack((m0,),dim=1) # (N,Nstates)

def draw_shocks(model,N):
    """ draw shocks """

    # a. unpack
    par = model.par

    # b. draw shocks
    psi_loc = -0.5*par.sigma_psi**2
    psi = torch.exp(torch.normal(psi_loc,par.sigma_psi,size=(par.T,N,)))

    return torch.stack((psi,),dim=-1) # (T,N,Nshocks)

Model.draw_initial_states = draw_initial_states
Model.draw_shocks = draw_shocks

# %% CELL

def numerical_integration(model): # not used in DeepSimulate
    """ quadrature nodes and weights """

    # a. unpack
    par = model.par

    # b. quadrature nodes and weights
    psi = par.psi
    psi_w = par.psi_w
    
    nodes = torch.stack((psi,),dim=1) # (Npsi,1)
    weigths = psi_w # (Npsi,)

    return nodes,weigths
    
Model.numerical_integration = numerical_integration

# %% CELL

def outcomes(model,states,actions,t=None,t0=0):
	""" outcomes """

	# a. unpack
	m = states[...,0]
	a = actions[...,0]

	# b. compute outcome
	c = m*(1-a) # intermediary outcome, not a state or action, but still useful

	return torch.stack((c,),dim=-1) # (T,N,Noutcomes)

Model.outcomes = outcomes

# %% CELL

def utility(par,c):
	""" utility """

	return torch.log(c)

def reward(model,states,actions,outcomes,t=None,t0=0):
	""" reward """

	# a. unpack
	par = model.par

	# b. consumption
	c = outcomes[...,0]

	# c. utility
	u = utility(par,c)

	return u 

Model.reward = reward

# %% CELL

def state_trans_pd(model,states,actions,outcomes,t=None,t0=0):
	""" transition to post-decision state """
	
	# a. unpack
	par = model.par

	# b. get cash-on-hand and consumption
	m = states[...,0]
	c = outcomes[...,0]

	# c. post-decision
	m_pd = m-c

	# d. finalize
	states_pd = torch.stack((m_pd,),dim=-1)
	
	return states_pd
	
	# Case I: shape = (T,...,Nstates_pd)
	# Case II: shape = (N,Nstates_pd)

Model.state_trans_pd = state_trans_pd

# %% CELL

def state_trans(model,states_pd,shocks,t=None):
	""" state transition with quadrature """

	# Case I: t is None -> t in 0,...,T-1 <= par.T-1:
	#  states_pd.shape = (T,N,1,Nstates_pd)
	#  shocks.shape = (1,1,Nnumint,Nshocks) [this is quadrature nodes]
	
	# Case II: t in 0,...,T-1, t0 irrelevant:
	#  states_pd.shape = (N,Nstates_pd)
	#  shocks.shape = (N,Nshocks) [this is actual shocks]

	# DeepSimulate: never t == None

	# a. unpack
	par = model.par
	train = model.train

	# b. get post-decision cash-on-hand and shock
	m_pd = states_pd[...,0]
	psi = shocks[...,0]

	# c. future cash-on-hand
	m_plus = (1+par.r)*m_pd + psi*par.kappa
	
	# d. finalize
	states_plus = torch.stack((m_plus,),dim=-1)

	return states_plus

	# Case I: states_plus.shape = (T,N,Nnumint,Nstates)
	# Case II: states_plus.shape = (N,Nstates)

Model.state_trans = state_trans

# %% CELL

def terminal_actions(model,states):
	""" terminal actions """

	# Case I: states.shape = (1,...,Nstates)
	# Case II: states.shape = (N,Nstates)
	
	# a. unpack
	par = model.par
	train = model.train
	dtype = train.dtype
	device = train.device
	
	# b. consume nothing in last period
	actions = torch.zeros((*states.shape[:-1],1),dtype=dtype,device=device)

	# c. multiplier action for DeepFOC
	if train.algoname == 'DeepFOC':
		multipliers = torch.zeros((*states.shape[:-1],1),dtype=dtype,device=device)
		actions = torch.cat((actions,multipliers),dim=-1)

	return actions 
	# Case I: shape = (1,...,Nactions)
	# Case II: shape = (N,Nactions)	

Model.terminal_actions = terminal_actions

# %% CELL


# --- Test de humo: DeepSimulate en el modelo consumo-ahorro simple (T=3) ---
import time
t0 = time.time()
model = Model(algoname='DeepSimulate', device='cpu', train={'K_time':0.3})
model.solve(do_print=True)
print(f"\n=== RESULTADO TEST ===")
print(f"tiempo: {time.time()-t0:.1f}s")
print(f"mejor R (recompensa vitalicia simulada): {model.sim.R:.6f}")
c = model.sim.outcomes[...,0].mean(dim=1).cpu().numpy()
m = model.sim.states[...,0].mean(dim=1).cpu().numpy()
print(f"consumo medio por periodo: {c}")
print(f"riqueza media por periodo: {m}")
# verificacion: en t=T-1 la politica terminal es consumir todo -> c_T = m_T
import numpy as np
err = abs(c[-1]-m[-1])/m[-1]
print(f"check politica terminal |c_T-m_T|/m_T = {err:.2e} (debe ser ~0)")
print("TEST OK" if err < 1e-6 else "TEST DUDOSO")
EOF_MARKER = True
