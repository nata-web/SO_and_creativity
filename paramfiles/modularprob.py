# -*- coding: utf-8 -*-
"""
Parameters for the abstract Modular weight matrix problem.

2024-03-20 10:41:10 (Wed), @author: Tesh
"""
import numpy as np
import os
import containers

def getOptions():     
    CO = containers.calcOptions()
    
    CO.w_order = [2,3][0]   # order of the weight matrix
    CO.num_alphas = 5 # amount of learning rates
    CO.alpha_max = 1e-4
    CO.alpha_min = 1e-7
    CO.alpha_base = (CO.alpha_max/CO.alpha_min)**(1/(CO.num_alphas-1))
    CO.alpha_arr = list((CO.alpha_base)**np.arange(0, CO.num_alphas) * CO.alpha_min)
    # CO.alpha_arr = [6e-7, 6.1e-7, 6.2e-7, 6.3e-7]      # learning rates alpha (for each size N) # [1e-4, 1e-6, 8e-7, 1.8e-7] 
    CO.N_arr = [100] * CO.num_alphas           # sizes N of the weights matrix
    CO.steps_arr = np.array(CO.N_arr) * 10
    
    CO.k_arr = [5] * CO.num_alphas            # module sizes k (for each size N)
    CO.p_arr = [0.1] * CO.num_alphas           # intermodule connections strength p (for each size N)
    CO.resets_arr = [1000] * CO.num_alphas    # amount of resets (for each size N)
    CO.runs = 20                # amount of simulation runs per each size N
    
    CO.seed_w = 12345
    
    CO.sim_seeds = np.random.randint(90000, size=(CO.runs))
    # CO.sim_seeds = [54321] * CO.runs
    CO.sat_seeds = [12345]
    
    return CO

def CO_per_N(CO, indN):
    CO.N = CO.N_arr[indN]
    CO.k, CO.p, CO.alpha = CO.k_arr[indN], CO.p_arr[indN], CO.alpha_arr[indN] 
    CO.resets = CO.resets_arr[indN]
    CO.eta = int(1/CO.alpha)
    CO.steps = CO.steps_arr[indN]
    CO.alphas = np.zeros((CO.resets, CO.steps), dtype=np.float64)
    CO.alphas[:,:] = CO.alpha
    CO.gamma = 1
    
    CO.path = os.path.join(CO.problemType, "N_" + str(CO.N), "k_" + str(CO.k))
    
    return CO

def CO_per_N_per_run(CO, indN, run):
    CO.seed_sim = CO.sim_seeds[run]

    # additional values for plots
    CO.main_title = f"{CO.problemType}, N={CO.N}, k={CO.k}, p={CO.p}"
    CO.main_title += f", ts={CO.steps}, resets={CO.resets}, α={round(1/CO.eta,11)}"
    CO.main_title += ", seed$_w$={}, seed$_{{sim}}$={}""".format(CO.seed_w, CO.seed_sim)
    
    
    CO.fig_name = f"{CO.problemType}_wo{CO.w_order}"
    CO.fig_name += f"_N{CO.N}_k{CO.k}_eta{CO.eta}_resets{CO.resets}_ts{CO.steps}"
    CO.fig_name += f"_sw{CO.seed_w}_ss{CO.seed_sim}"
    
    return CO
