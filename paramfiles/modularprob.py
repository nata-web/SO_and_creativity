"""
Parameters for the abstract Modular weight matrix problem.
"""
import numpy as np
import os
import containers

def getOptions():     
    CO = containers.calcOptions()
    
    # choose learning rates
    CO.alphaGrid = False
    CO.num_alphas = 64  # amount of learning rates
    CO.alpha_max = 1e-4
    CO.alpha_min = 1e-7
    if CO.num_alphas>1:
        CO.alpha_base = (CO.alpha_max/CO.alpha_min)**(1/(CO.num_alphas-1))
        CO.alpha_arr = list((CO.alpha_base)**np.arange(0, CO.num_alphas) * CO.alpha_min)
    else:
        if CO.alphaGrid:
            CO.alpha_arr = [2e-7,5e-7,5e-6,5e-5] # NN,NA; N,A; NN,A; N,NA 
        else:
            CO.alpha_arr = [5e-7]
    CO.N_arr = [100] * len(CO.alpha_arr)         # sizes N of the weights matrix
    CO.steps_arr = np.array(CO.N_arr) * 10
    
    CO.k_arr = [5]          * len(CO.alpha_arr)    # module sizes k (for each size N)
    CO.p_arr = [0.1]        * len(CO.alpha_arr)    # intermodule connections strength p (for each size N)
    if CO.num_alphas>1:
      CO.resets_arr = [1000]  * len(CO.alpha_arr)   # amount of resets (for each size N)
      CO.runs = 2000                                # amount of simulation runs per each size N
    else:
      CO.resets_arr = [1000]  * len(CO.alpha_arr)   # amount of resets (for each size N)
      CO.runs = 1                                   # amount of simulation runs per each size N
    
    CO.seed_w = [12345,23456][0]
    
    if CO.runs>1:
        CO.sim_seeds = np.arange(CO.num_alphas*CO.runs).reshape(CO.num_alphas,CO.runs)
    else:
        if CO.alphaGrid:
            CO.sim_seeds = [[53833]]*4
        else:
            CO.sim_seeds = [[1914]]
    
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
    CO.seed_sim = CO.sim_seeds[indN][run]

    # additional values for plots
    CO.main_title = f"{CO.problemType}, N={CO.N}, k={CO.k}, p={CO.p}"
    CO.main_title += f", ts={CO.steps}, resets={CO.resets}, α={round(1/CO.eta,11)}"
    CO.main_title += ", seed$_w$={}, seed$_{{sim}}$={}""".format(CO.seed_w, CO.seed_sim)
    
    CO.fig_name = f"{CO.problemType}"
    CO.fig_name += f"_N{CO.N}_k{CO.k}_eta{CO.eta}_resets{CO.resets}_ts{CO.steps}"
    CO.fig_name += f"_sw{CO.seed_w}_ss{CO.seed_sim}"
    
    return CO
