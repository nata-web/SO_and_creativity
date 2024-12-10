import os
import sys
import numpy as np
import pickle as pickle      # Func: dump, load
import scipy.sparse as sparse

import containers

F = 1           # '1' to load Fortran module, 'compile' to dynamically compile it,
                # 'compile' to dynamically compile it,
speed = 1       # speed up by not explicitly adding w=w+dw
isBipolar = 1   # change to '0' is the state is binary (i.e. {0,1})
       
################## Functions to generate random matrices ##################

rf = lambda n: np.random.choice(np.arange(-1,2,1), n, p=[0.5, 0, 0.5])
theta = lambda x: 1 if x >= 0 else -1

w_type = lambda x: "sparse" if(x) else "modular" 
                      
def sparseRandom(m, n, density=0.01, format='coo', dtype=None, random_state=None, data_rvs=None):
    "Generates a m*n random sparse symmetric matrix from {-1,0,1} with density d of nonzero values"
    def my_func(N, seed=None): 
        return random_state.choice(np.arange(-1,2,1), N, p=[0.5, 0, 0.5])
        # 'my_func' needs to be defined inside 'sparseRandom' to make sure that 
        # sampling the values of the structurally nonzero entries of the matrix 
        # uses the same random_state as the one used for sampling the sparsity structure   
    return sparse.random(m, n, density, format, dtype, random_state, my_func)
        
def w_sparse(N, d, seed_w):
    rng = np.random.default_rng(seed_w)
    w = sparseRandom(N, N, density = d, 
                      random_state = rng, # If seed is given, it will produce same result
                      dtype='f'           # use float32 first
                      # ).astype('int8')    # then convert to int8
                      ).astype('float64')
    w = w.toarray()
    w = np.triu(w) # make it upper triangular 
    w = w + w.T - np.diag(np.diag(w)) # make the matrix symmetric
    Ws = (w,)

    return Ws

def w_modular(N, k, p, seed_w):
    np.random.seed(seed_w)
    w = np.random.choice((-p,p),(N,N))
    for i in range(N):
        for j in range(N):
            if np.floor(i/k) == np.floor(j/k):
                w[i,j] = rf(1)
    w = np.triu(w) # make it upper triangular 
    w = w + w.T - np.diag(np.diag(w))

    Ws = (w,)
        
    return Ws
    
  
def beginRun(CO, Ws, WsOrig, energies, startState, doLearn):
    
    start = np.array(os.times())
    np.random.seed(CO.seed_sim)
    
    if F: # run Fortran routine
        if not isBipolar:
            raise ValueError('Fortran hebbF.runsimple only works for bipolar states')
        state = np.zeros(CO.N, dtype=np.int8, order='F')
        randoms = np.zeros((CO.N+CO.steps ,CO.resets), dtype=int, order='F')
        # Prefill random values for Fortran learning in the same order as
        # python learn 
        for r in range(CO.resets):
            randoms[:CO.N, r] = 2*np.random.randint(0,2, CO.N) - 1
            randoms[CO.N:, r] = np.random.randint(0, CO.N, CO.steps) + 1 #Fortran uses 1 base indexing
        hebbF.hebb.runsimple(*Ws[::-1], *WsOrig[::-1], energies.T, doLearn, CO.alpha, state, randoms)
        # transpose of energies because Fortran array ordering is reversed
    else:
        raise ValueError('The simulation requires the Fortran hebbF.runsimple routine that is enabled with F=1')
    
    duration = np.array(os.times()) - start
    print("""\nExecution time for N={} (ts={}, resets={}) with L={}:\n""".format(CO.N, CO.steps, CO.resets, doLearn), 
          [round(d, 4) for d in duration])
    if doLearn:
        print("eta=", CO.eta, ", α=" + str(CO.alpha) + "\n")
    sys.stdout.flush()
    
    return state

def simulate(CO, startState=None, PO=containers.plotOptions()):
    energies = np.zeros((3*CO.resets, CO.steps), dtype=np.float64)

    np.random.seed(CO.seed_w)
    if CO.problemType == "Sparse":
        Ws = w_sparse(CO.N, CO.d, CO.seed_w)
    else:
        Ws = w_modular(CO.N, CO.k, CO.p, CO.seed_w)
    I = np.zeros(CO.N)
    Ws=tuple(w.copy(order='F') for w in Ws)
    Ws=(0,I)+Ws
    WsOrig=(0,)+tuple(w.copy() for w in Ws[1:])
        
    state = beginRun(CO, Ws, WsOrig, energies[:CO.resets], startState, False)
    stateLearn = beginRun(CO, Ws, WsOrig, energies[CO.resets:CO.resets*2], startState, True)
    stateEnd = beginRun(CO, Ws, WsOrig, energies[CO.resets*2:CO.resets*3], startState, False)

    if PO.dump:
        with open(os.path.join(PO.path,'output_{}_{}_{}_ss{}'.format(CO.problemType, CO.N, CO.eta, CO.seed_sim)),'wb') as out:
          for thing in [Ws, WsOrig, state, energies, stateLearn]:
            pickle.dump(thing, out)
    
    result = containers.container()
    result.energies = energies
    result.Ws = Ws
    result.WsOrig = WsOrig
    result.state = state
    result.stateLearn = stateLearn

    return result

def load_data(CO):
    result = containers.container()
    with open(os.path.join(CO.path,'output_{}_{}_{}_ss{}'.format(CO.problemType, CO.N, CO.eta, CO.seed_sim)),'rb') as inData:
        result.Ws = pickle.load(inData)
        result.WsOrig = pickle.load(inData)
        result.state = pickle.load(inData)
        result.energies = pickle.load(inData)
        result.oInfo = pickle.load(inData)
        result.sInfo = pickle.load(inData)
        result.stateLearn = pickle.load(inData)
        
    return result

    
if F == 'compile':
    import importlib, subprocess
    
    modFile = 'hebbclean.F90'
    stamp = int(os.path.getmtime(modFile))
    module = 'hebbF{}'.format(stamp)
    
    try:
      hebbF = importlib.import_module(module)
    except ModuleNotFoundError:
      print('Compiling', modFile)
      try:
          res=subprocess.check_output('f2py --f90flags="-g -fdefault-integer-8 -O3" -m {} -c {}'.format(module,modFile),shell=True,stderr=subprocess.STDOUT)
      except subprocess.CalledProcessError as ex:
          print(ex.output.decode('utf-8'))
          raise ValueError() from None
      hebbF = importlib.import_module(module)
elif F==1:
    try:
        import hebbF
    except ModuleNotFoundError:
        print('hebbF.so compiled Fortran module not found\nTry compiling it with:\nf2py3 --f90flags="-g -fdefault-integer-8 -O3" -m hebbF -c hebbclean.F90')
        raise