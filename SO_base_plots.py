import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.optimize,scipy.special
import numpy as np
import os
import containers 

# For creating your own colormap
from matplotlib.colors import LinearSegmentedColormap

SVGcolors = ["#ddaa33","#66ccee", "#004488", "#bb5566"]
# Create the 'iridescent' scheme (https://personal.sron.nl/~pault/)
clrs = ['#FEFBE9', '#FCF7D5', '#F5F3C1', '#EAF0B5', '#DDECBF',
        '#D0E7CA', '#C2E3D2', '#B5DDD8', '#A8D8DC', '#9BD2E1',
        '#8DCBE4', '#81C4E7', '#7BBCE7', '#7EB2E4', '#88A5DD',
        '#9398D2', '#9B8AC4', '#9D7DB2', '#9A709E', '#906388',
        '#805770', '#684957', '#46353A']

cmap = LinearSegmentedColormap.from_list("",clrs)
cmap.set_bad('#999999')

def plotEnergies(CO, ax, energies, runs, colors=['#004488','#BB5566','#66ccee'], labels=["Before Learning","Learning","After Learning"]):
  for which in range(3):
    ax.plot(np.arange(which*CO.resets, (which+1)*CO.resets), 
            energies[which*CO.resets:(which+1)*CO.resets,-1], c = colors[which], 
            ls='None', marker='o', markersize=1, label = labels[which])

def margins(ax,fig,CO,xLabel,yLabel,img=None):
    """ Function that adjusts the margins of the figures to create plots of equal size"""
    CBpad = 0.05
    if CO.alpha in [2e-7,5e-6]:
        ax.set_ylabel(yLabel)
        shared_margins = {'left' :0.1,'right':0.98}
        if img is not None:
            shared_margins['left'] += CBpad/5+0.05*(0.97-0.15)
    else:
        ax.set_yticklabels([])
        shared_margins = {'left' :0.02,'right':0.9}
        if img is not None:
          # create an axes on the right side of ax. The width of cax will be 5%
          # of ax and the padding between cax and ax will be fixed at 0.05 inch.
          divider = make_axes_locatable(ax)
          cax = divider.append_axes("right", size="5%", pad=CBpad)
          cb = fig.colorbar(img, ax = ax, cax=cax)
    if CO.alpha in [2e-7,5e-5]:
        ax.set_xticklabels([])
        shared_margins.update({'top' :0.9,'bottom':0.03})
    else:
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xlabel(xLabel)
        shared_margins.update({'top' :.97,'bottom':0.1})
    fig.subplots_adjust(**shared_margins)


def gridPlots(CO, result, runs, PO=containers.plotOptions(), mainTitle=False):
    """ Generates separate figures for visited attractors, their histograms and finals weights"""
    # Attractors visited
    fig, ax = plt.subplots(figsize=[5,5])
    plotEnergies(CO, ax, result.energies, runs)
    ax.set_ylim([-158,-95])
    margins(ax,fig,CO,"Resets","Energy")
 
    if mainTitle:
        if CO.startState is not None:
            CO.main_title += ", startState=True"
        fig.subplots_adjust(top=0.9)
        fig.suptitle(CO.main_title, y=0.99)
    
    if PO.saveFigures:  
        if CO.startState is not None:
            CO.fig_name += "_startState" 
        plt.savefig(os.path.join(PO.path, CO.fig_name + '_attractors.png'))
        plt.savefig(os.path.join(PO.path, CO.fig_name + '_attractors.pdf'))

    resets = result.energies.shape[0]//3
    
    # Histogram of attractors visited
    fig, ax = plt.subplots(figsize=[5,5])
    reset_wind = int(resets/10)             # width of reset window
    avg_fac = int(3*resets / reset_wind)    # number of reset windows
    eMin,eMax = (result.energies[:,-1].min(),result.energies[:,-1].max())
    eMin=-158
    eMax=-95
    binN = 30
    h = np.array([np.histogram(r,binN,range=(eMin,eMax))[0] for r in result.energies[:,-1].reshape((avg_fac,reset_wind))])

    img = ax.matshow(h.T,origin="lower",extent=[0,avg_fac,eMin,eMax], aspect='auto')
    margins(ax,fig,CO,"Reset bins","Energy",img)
    
    if PO.saveFigures:  
        if CO.startState is not None:
            CO.fig_name += "_startState" 
        plt.savefig(os.path.join(PO.path, CO.fig_name + '_hist.png'))
        plt.savefig(os.path.join(PO.path, CO.fig_name + '_hist.pdf'))

    # Final weights
    fig, ax = plt.subplots(figsize=[5,5])
    img = ax.matshow(result.Ws[2], vmin=result.WsOrig[2].min(), vmax=result.WsOrig[2].max(), interpolation="none", cmap=cmap,aspect="equal")
    margins(ax,fig,CO,"i","j",img)
    if PO.saveFigures:
        plt.savefig(os.path.join(PO.path, CO.fig_name + '_W_fin.png'))
        plt.savefig(os.path.join(PO.path, CO.fig_name + '_W_fin.pdf'))
        # plt.savefig(os.path.join(PO.path, CO.fig_name + '_W_fin.svg'))

def plot_6(CO, result, runs, PO=containers.plotOptions(), mainTitle=True):
    
    Ws = result.Ws
    WsOrig = result.WsOrig
    
    resets = result.energies.shape[0]//3
    steps = result.energies.shape[1]
    
    E_NL = result.energies[resets-50:resets]
    mean_E_NL = np.mean(E_NL.T,axis=1)
    std_E_NL = np.std(E_NL.T,axis=1)

    E_L = result.energies[resets+resets-50:resets*2]
    mean_E_L = np.mean(E_L.T,axis=1)
    std_E_L = np.std(E_L.T,axis=1)

    separatePlots = True
    weightsPlots = False

    if separatePlots:
        #### Separate figures energies with and without learning
        # Energies without learning
        fig, ax = plt.subplots(figsize=[5,5])
        ax.plot(np.arange(steps), E_NL.T, c=[0.90, 0.90, 0.90], alpha=1, zorder=1)
        ax.plot(np.arange(steps), mean_E_NL, c='navy', alpha=0.5, zorder=3)
        ax.fill_between(np.arange(steps), mean_E_NL - std_E_NL,
                                                mean_E_NL + std_E_NL,
                                alpha=0.2, edgecolor="steelblue", facecolor="steelblue", zorder=2)
        # ax.set_title('Energies Without learning')
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Energy")
        plt.tight_layout()
        if PO.saveFigures:  
            if CO.startState is not None:
                CO.fig_name += "_startState" 
            plt.savefig(os.path.join(PO.path, CO.fig_name + '_Es_one_NL.png'))
            plt.savefig(os.path.join(PO.path, CO.fig_name + '_Es_one_NL.pdf'))

        # Energies with learning
        fig, ax = plt.subplots(figsize=[5,5])
        ax.plot(np.arange(steps), E_L.T, c=[0.90, 0.90, 0.90], alpha=1, zorder=1)
        ax.plot(np.arange(steps), mean_E_L, c='navy', alpha=0.5, zorder=3)
        ax.fill_between(np.arange(steps), mean_E_L - std_E_L,
                                                mean_E_L + std_E_L,
                                alpha=0.2, edgecolor="steelblue", facecolor="steelblue", zorder=2)
        
        # ax.set_title('Energies With Learning')
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Energy")
        plt.tight_layout()
        if PO.saveFigures:  
            if CO.startState is not None:
                CO.fig_name += "_startState" 
            plt.savefig(os.path.join(PO.path, CO.fig_name + '_Es_one_L.png'))
            plt.savefig(os.path.join(PO.path, CO.fig_name + '_Es_one_L.pdf'))

    ###### A 2-by-2 figure with energies before and after learning, visited attractor and their hoistogram
    fig, axs = plt.subplots(2,2,figsize=[10,10])
    
    # subplot(221) - Energies without learning
    axs[0,0].plot(np.arange(steps), E_NL.T, c=[0.90, 0.90, 0.90], alpha=1, zorder=1)
    axs[0,0].plot(np.arange(steps), mean_E_NL, c='navy', alpha=0.2, zorder=3)
    
    axs[0,0].fill_between(np.arange(steps), mean_E_NL - std_E_NL,
                                            mean_E_NL + std_E_NL,
                            alpha=0.2, edgecolor="steelblue", facecolor="steelblue", zorder=2)
    axs[0,0].set_title('Energies WoL')
    axs[0,0].set_xlabel("Timesteps")
    axs[0,0].set_ylabel("Energy")
    
    # subplot(222) - Energies with learning  
    axs[0,1].plot(np.arange(steps), E_L.T, c=[0.90, 0.90, 0.90], alpha=1, zorder=1)
    axs[0,1].plot(np.arange(steps), mean_E_L, c='navy', alpha=0.2, zorder=3)
    axs[0,1].fill_between(np.arange(steps), mean_E_L - std_E_L,
                                            mean_E_L + std_E_L,
                            alpha=0.2, edgecolor="steelblue", facecolor="steelblue", zorder=2)
    
    axs[0,1].set_title('Energies WL')
    axs[0,1].set_xlabel("Timesteps")
    axs[0,1].set_ylabel("Energy")
    
    # subplot(223) - Attractors visited
    plotEnergies(CO, axs[1,0], result.energies, runs)
    axs[1,0].set_title('Attractor states visited')
    axs[1,0].set_xlabel("Resets")
    axs[1,0].set_ylabel("Energy")
    
    # subplot(224) - Histogram of attractors visited
    reset_wind = int(resets/10)
    avg_fac = int(3*resets / reset_wind)
    eMin,eMax = (result.energies[:,-1].min(),result.energies[:,-1].max())
    h = np.array([np.histogram(r,30,range=(eMin,eMax))[0] for r in result.energies[:,-1].reshape((avg_fac,reset_wind))])

    img = axs[1,1].matshow(h.T,origin="lower",extent=[0,avg_fac,eMin,eMax], aspect='auto') 
    axs[1,1].set_title('Histogram of attractor energies')
    divider = make_axes_locatable(axs[1,1])
    cax = divider.append_axes("right", size="5%", pad=0.05)     
    fig.colorbar(img, ax = axs[1,1], cax=cax)
    axs[1,1].set_xlabel("Time window")
    axs[1,1].set_ylabel("Energy")
        
    plt.tight_layout()
        
    if mainTitle:
        if CO.startState is not None:
            CO.main_title += ", startState=True"
        fig.subplots_adjust(top=0.9)
        fig.suptitle(CO.main_title, y=0.99)
    
    if PO.saveFigures:  
        if CO.startState is not None:
            CO.fig_name += "_startState" 
        plt.savefig(os.path.join(PO.path, CO.fig_name + '_Es_four.png'))
        plt.savefig(os.path.join(PO.path, CO.fig_name + '_Es_four.pdf'))
    
    if weightsPlots:
        # FIGURE with weights only
        fig, axs = plt.subplots(1,2,figsize=[10,5])  
        # subplot(121) - Initial weights
        img = axs[0].matshow(WsOrig[2], interpolation="none", cmap=cmap);        
        axs[0].set_title('Initial weights')
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)     
        fig.colorbar(img, ax = axs[0], cax=cax)
    
        # subplot(232) - Final weights
        img = axs[1].matshow(Ws[2], vmin=WsOrig[2].min(), vmax=WsOrig[2].max(), interpolation="none", cmap=cmap);
        axs[1].set_title('Final weights')
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)     
        fig.colorbar(img, ax = axs[1], cax=cax)      
            
        plt.tight_layout()
            
        if mainTitle:
            if CO.startState is not None:
                CO.main_title += ", startState=True"
            fig.subplots_adjust(top=0.9)
            fig.suptitle(CO.main_title, y=0.99)
        
        if PO.saveFigures:
            plt.savefig(os.path.join(PO.path, CO.fig_name + '_Ws.png'))
            plt.savefig(os.path.join(PO.path, CO.fig_name + '_Ws.pdf'))

# Define a formatter for scientific notation
def scientific_notation(x, pos):
    return f"${x:.0e}$"  # Format as scientific notation in LaTeX

def alphasPlot(CO, PO, L_energies, NL_energies, NL2_energies, MinMax_NL, im_param, fig_name):

    probPlot, convPlot = im_param
    eMin_NL, eMax_NL, eMeanMin_NL, eMeanMax_NL = MinMax_NL
    eMin, eMax = (L_energies.min()-2, L_energies.max())
    eMax = -80.
    eSteps = int(min((eMax-eMin), L_energies.shape[1]**0.5))
    print(eSteps,L_energies.shape[1])
    eRange = np.arange(eMin, eMax, int((eMax-eMin)/eSteps)+1)

    fig, ax = plt.subplots(figsize=(8, 6),constrained_layout=True)
    ax.set_ylabel("count")
    ax.set_xlabel("Energy")
    
    hist, bins = np.histogram(NL_energies, bins=np.arange(eMin_NL,eMax_NL,0.2)-0.1)
    ax.hist(bins[:-1],bins,weights=hist, label="Non learning energies", alpha=0.6, color="green")
    print(bins[:-1])
    print(hist,len(hist))

    fitBins = 0.5*(bins[1:]+bins[:-1])
    fitHist = hist
    if 0:
        fitBins = fitBins[hist>0]
        fitHist = fitHist[hist>0]
        print(fitBins)
        print(fitHist)
    gauss = lambda x,x0,sig,A:A*np.exp(-0.5*((x-x0)/sig)**2)
    poisson = lambda x,x0,lamb,A:A*lamb**(x-x0)*np.exp(-lamb)/scipy.special.gamma(x-x0+1)
    p0={
      gauss:[NL_energies.mean(),NL_energies.std(),np.prod(NL_energies.shape)],
      poisson:[NL_energies.min(),NL_energies.mean()-NL_energies.min(),1e7]
    }
    def doFit(fitFunc):
      res = scipy.optimize.curve_fit(fitFunc,fitBins,fitHist,p0[fitFunc])
      print(fitFunc(bins[:-1],*p0[fitFunc]))

      if fitFunc == gauss:
        x0,sig, A = res[0]
      else:
        x0 = res[0][0]+res[0][1]
        sig = res[0][1]**0.5
        print('x0',res[0][0])
      print('mu',x0,'sig',sig)
      lineStyle = {gauss:'-',poisson:'--'}[fitFunc]
      ax.axvline(x=x0, color="blue", linestyle=lineStyle, linewidth=1.5, label=f"$\mu$ = {x0:.1f}")
      for ind,col in enumerate(['red','orange','yellow']):
          for pm in [+1,-1]:
              if pm==+1:
                  label=f"$\mu\pm\,{ind+1}\sigma$"
              else:
                  label=None
              ax.axvline(x=x0+pm*(ind+1)*sig, color=col, linestyle=lineStyle, linewidth=1.5, label=label)

      energies=np.linspace(eMin_NL,eMax_NL,eSteps*4)
      ax.plot(energies,fitFunc(energies,*res[0]),label={gauss:"Gaussian",poisson:"Poisson"}[fitFunc]+" fit")
      return x0,sig

    if 0:
        fitFunc = gauss
        x0,sig=doFit(fitFunc)
    else:
        fitFunc = poisson
        x0,sig = doFit(fitFunc)

    ax.legend(loc="upper right", fontsize="small")

    if PO.saveFigures:
        plt.savefig(os.path.join(PO.path, fig_name + '_hist.png'))
        plt.savefig(os.path.join(PO.path, fig_name + '_hist.pdf'))

    eHist = []

    for data in L_energies:
        combined_data = data.reshape(-1)
        hist, _ = np.histogram(combined_data, bins=eRange)
        eHist.append(hist)

    h = np.array(eHist)
    fig, ax = plt.subplots(figsize=(8, 6),constrained_layout=True)
    ax2 = ax.twinx() # right y-axis for probability plot
    cax = ax.matshow(h.T, cmap='viridis', origin='lower', aspect='auto',
                        extent=[0,len(CO.alpha_arr),eMin,eMax],alpha=0.8)

    # Add horizontal lines for mu pm 1-,2-,3-sigmas
    ax.axhline(y=x0-1*sig, color="red", linestyle="--", linewidth=1.5, label=r"$\mu\pm\,\sigma$")
    ax.axhline(y=x0+1*sig, color="red", linestyle="--", linewidth=1.5)
    ax.axhline(y=x0-2*sig, color="orange", linestyle="--", linewidth=1.5, label=r"$\mu\pm2\sigma$")
    ax.axhline(y=x0+2*sig, color="orange", linestyle="--", linewidth=1.5)
    ax.axhline(y=x0-3*sig, color="yellow", linestyle="--", linewidth=1.5, label=r"$\mu\pm3\sigma$")
    ax.axhline(y=x0+3*sig, color="yellow", linestyle="--", linewidth=1.5)

    # Configure axis labels
    tickStep = max(1,int(len(CO.alpha_arr)/5))
    ticks = list(range(len(CO.alpha_arr)))[::tickStep]
    ax.set_ylim(eMin,eMax)
    ax.set_xticks(ticks)
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()
    ax.set_xticklabels([scientific_notation(lr, None) for lr in CO.alpha_arr[::tickStep]])
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Energy")

    if convPlot:
        meanStd = NL2_energies.std(axis=2).mean(axis=1)
        mean = NL2_energies.mean(axis=(1,2))
        ax.fill_between(np.arange(len(CO.alpha_arr)), mean - meanStd,
                                                      mean + meanStd,
                                alpha=0.2, edgecolor="orange", facecolor="orange")#, zorder=2)

    ax.legend(loc="upper right", fontsize="small")

    if probPlot:
        pdf={gauss:scipy.stats.norm(0,sig),poisson:scipy.stats.poisson(sig**2,-sig**2)}[fitFunc]
        loweringProbability = (L_energies<x0-sig).sum(axis=1)/L_energies.shape[1]
        print('1sig',loweringProbability.max(),pdf.cdf(-sig),1-pdf.cdf(-sig),end='   ')
        ax2.plot(range(len(CO.alpha_arr)),loweringProbability,color="red")
        loweringProbability = (L_energies<x0-2*sig).sum(axis=1)/L_energies.shape[1]
        print('2sig',loweringProbability.max(),pdf.cdf(-2*sig),1-pdf.cdf(-2*sig),end='   ')
        ax2.plot(range(len(CO.alpha_arr)),loweringProbability,color="orange")
        loweringProbability = (L_energies<x0-3*sig).sum(axis=1)/L_energies.shape[1]
        print('3sig',loweringProbability.max(),pdf.cdf(-3*sig),1-pdf.cdf(-3*sig))
        ax2.plot(range(len(CO.alpha_arr)),loweringProbability,color="yellow")

    fig.colorbar(cax, ax=ax, label='Frequency')

    plt.show()

    if PO.saveFigures:
        plt.savefig(os.path.join(PO.path, fig_name + '.png'))
        plt.savefig(os.path.join(PO.path, fig_name + '.pdf'))
