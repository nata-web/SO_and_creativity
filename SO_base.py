# Load necessary libraries
import os
import sys
import numpy as np
import pickle as pickle      # Func: dump, load
import importlib
import shutil
import datetime
import subprocess

# Load custom functions 
import SO_base_FUNC
import SO_base_plots
import containers

### Define the path for exporting images and data
path = os.path.abspath(os.getcwd()) 
path = os.path.join(path,'output','output_python')

### Choose the problem
problemType_options = ["Modular","Sparse"][0]

### Choose style for plots
plotsStyle = ['regular', 'IEEEplots'][0]
plot6 = False       # Make one figure  with all simulation results in 6 subplots (per N, per run)
                    # When 'False' all simulation results are generated as separate figures
alphaPlot = True    # Make figures 5 and 6 
saveFigures = True

def main(CO, PO, firstNind, lastNind):
    
    # Run the simulation for all parameters (all Ns, alphas, runs)
    for indN in range(firstNind,lastNind):
        CO = problem.CO_per_N(CO, indN)

        # Initialize an array for converged energies for all runs, all stages 
        Econv_runs = np.zeros((CO.runs, 3*CO.resets_arr[0]), dtype=np.float64) 
        
        # Run the simulation for individual seeds
        for run in range(CO.runs):
            print(f'{run}/{CO.runs},{indN}/{len(CO.N_arr)}')
            CO = problem.CO_per_N_per_run(CO, indN, run)
            
            result = SO_base_FUNC.simulate(CO, PO=PO)
            Econv_runs[run,:] = result.energies[:,-1] # Only save the last energy of each reset
            
            CO.resultF = result
            
            if PO.plot6:
                SO_base_plots.plot_6(CO, result, CO.runs, PO=PO)
            if CO.alphaGrid:
                SO_base_plots.gridPlots(CO, result, CO.runs, PO=PO)
        
        # Save the results
        if CO.runs>1:
            output_name = 'output_{}_N{}_k{}_sw{}_Neta{}_ieta{}.txt'
            with open(os.path.join(PO.path, output_name.format(CO.problemType, CO.N, CO.k, CO.seed_w,CO.num_alphas,indN)),'wb') as out:
                pickle.dump(Econv_runs, out)
        else:    
            output_name = 'output_{}_N{}_k{}_eta{}_sw{}_ss{}.txt'
            with open(os.path.join(PO.path, output_name.format(CO.problemType, CO.N, CO.k,CO.eta,CO.seed_w,CO.seed_sim)),'wb') as out:
                pickle.dump(Econv_runs, out)
    
    # Make figures 5 and 6 in the paper
    if alphaPlot:
        # Load the energy data from each file
        all_energies = np.zeros((CO.num_alphas,CO.runs,3*CO.resets_arr[0])) # last energies of each reset for all stages (non-learning, learning, after learning)
        for alpha_num in range(CO.num_alphas):
            CO = problem.CO_per_N(CO, alpha_num)
            result = containers.container()
            with open(os.path.join(PO.path,'output_{}_N{}_k{}_sw{}_Neta{}_ieta{}.txt'.format(CO.problemType, CO.N, CO.k, CO.seed_w, CO.num_alphas, alpha_num)),'rb') as inData:
                result.energies = pickle.load(inData)
                all_energies[alpha_num]=result.energies

        L_energies = all_energies[:,:, 1000:2000]       # energies in learning stage, all resets
        F_energies = all_energies[:,:, -1]              # energies after learning, last reset
        NL_energies = all_energies[:,:, 0:1000]         # energies before learning, all resets
        NL2_energies = all_energies[:,:, 2000:3000]     # energies after learning, all resets
        eMin_NL, eMax_NL = (NL_energies.min(), NL_energies.max()) # the band on energies before learning
        eMeanMin_NL, eMeanMax_NL = (NL_energies.min(axis=2).mean(), NL_energies.max(axis=2).mean()) # the band on energies before learning
        MinMax_NL = (eMin_NL, eMax_NL, eMeanMin_NL, eMeanMax_NL)

        # probPlot, convPlot = False, False
        # fig_name = 'E_alphas_{}_N{}_k{}_{}as_{}ss_{}sw'.format(
        #         CO.problemType, CO.N, CO.k_arr[0], CO.num_alphas, CO.runs, CO.seed_w)
        # SO_base_plots.alphasPlot(CO, PO, L_energies, NL_energies, NL2_energies, MinMax_NL, (probPlot, convPlot), fig_name)

        # probPlot, convPlot = False, False
        # fig_name = 'E_alphas_{}_N{}_k{}_{}as_{}ss_{}sw_lastReset'.format(
        #         CO.problemType, CO.N, CO.k_arr[0], CO.num_alphas, CO.runs, CO.seed_w)
        # SO_base_plots.alphasPlot(CO, PO, F_energies, NL_energies, NL2_energies, MinMax_NL, (probPlot, convPlot), fig_name)
        
        # Generate figures 5 and 6 in the paper
        probPlot, convPlot = True, False
        fig_name = 'E_alphas_{}_N{}_k{}_{}as_{}ss_{}sw_lastReset_probPlot'.format(
                CO.problemType, CO.N, CO.k_arr[0], CO.num_alphas, CO.runs, CO.seed_w)
        SO_base_plots.alphasPlot(CO, PO, F_energies, NL_energies, NL2_energies, MinMax_NL, (probPlot, convPlot), fig_name)

        # probPlot, convPlot = False, True
        # fig_name = 'E_alphas_{}_N{}_k{}_{}as_{}ss_{}sw_lastReset_convPlot'.format(
        #         CO.problemType, CO.N, CO.k_arr[0], CO.num_alphas, CO.runs, CO.seed_w)
        # SO_base_plots.alphasPlot(CO, PO, F_energies, NL_energies, NL2_energies, MinMax_NL, (probPlot, convPlot), fig_name)
            
if __name__ == '__main__':
    
    PO = containers.plotOptions()
    PO.path = path
    PO.plot6 = plot6
    PO.saveFigures = saveFigures 
    
    # import parameters per chosen problem
    problemType = problemType_options
    problem = importlib.import_module('paramfiles.'+problemType.lower()+'prob')
    CO = problem.getOptions()
    CO.problemType = problemType
    
    # Store the parameters files with each commit
    date_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S") 
    git_hash = subprocess.check_output(['git','rev-parse','HEAD'],encoding='utf-8').strip('\n')
    path_params = os.path.join(path, CO.problemType, "Parameters")
    if not os.path.exists(path_params):
        os.makedirs(path_params, exist_ok=True)
    shutil.copyfile(problem.__file__, os.path.join(path_params, date_str + "_" + git_hash + ".py"))
    
    print(f"Selected problem type: {problemType}")

    # Read first and second parameter as limits for alpha scan
    # SO_base can be called like
    # python SO_base.py 3 7
    # to only compute alpha_arr[3:7].
    # Given CPU_NUMBER CPUs and JOBS=len(alpha_arr)/CPU_NUMBER jobs per CPU,
    # this can be used to run SO_base in parallel:
    # for ((i=0;i<CPU_NUMBER;i++)) ; do python SO_base.py $((8*i)) $((8*(i+1)) & done
    # Once this is done, alpha plots can be generated with:
    # python SO_base.py 0 0
    if len(sys.argv) > 1:
      firstNind = int(sys.argv[1])
    else:
      firstNind = 0
    if len(sys.argv) > 2:
      lastNind = int(sys.argv[2])
    else:
      lastNind = len(CO.N_arr)
    if lastNind == firstNind:
      alphaPlot = True
    else:
      alphaPlot = False
    main(CO, PO, firstNind, lastNind)
