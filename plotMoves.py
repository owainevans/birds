from utils import *
from train_birds import *
import matplotlib.pylab as plt
import numpy as np

def plotMoves(baseDirectory=''):
    fileName = baseDirectory + 'moves.dat'
    with open(fileName,'r') as f:
        fString = f.read()
    moves = eval( fString[ fString.find('=')+1: ] )
    no_days_run = max([key[1] for key in moves.keys()]) + 1

    allMoves = checkMoves(moves,no_days=no_days_run)

    no_days = 19
    params['days'] = range(no_days)
    ground_moves = readReconstruction(params)
    allMovesGround = checkMoves(ground_moves,no_days=no_days)

    fig,ax = plt.subplots(no_days_run,1,figsize=(10,3*no_days_run) )
    for i in range(no_days_run):
        ax[i].plot(allMoves[i],label='Inferred')
        ax[i].plot(allMovesGround[i],label='Ground')
        ax[i].set_xlim(0,100)
        ax[i].set_title('# Birds leaving cell on day %i'%i)
        ax[i].set_xlabel('Cell number')
        ax[i].set_ylabel('# birds leaving cell')
        ax[i].legend()
        
    fig.tight_layout()
    return allMoves,allMovesGround,fig


def plotSamples(baseDirectory,noIters=1):
    fileName = baseDirectory+'posteriorRunsDump.py'
    with open(fileName,'r') as f:
        fString = f.read()
    runs = eval( fString[ fString.rfind('=')+1: ] )
    
    allSamples=[]; priorTriples=[]
    
    for run in runs:
        for i,triple in enumerate(run):
            if not np.mod(i,noIters+1)==0:
                allSamples.append(triple)
            else:
                priorTriples.append(triple)

    allSamples=np.array(allSamples)
    logscores = allSamples[:,0]
    l2 = allSamples[:,1]
    
    allPriorRuns = np.array(priorTriples)
    logscoresPrior = allPriorRuns[:,0]
    l2Prior = allPriorRuns[:,1]

    fig,ax = plt.subplots(2,1,sharex=True)
    ax[0].hist(l2,label='L2_post',color='r')
    ax[1].hist(l2Prior,label='L2_prior',alpha=.4,color='y')
    [ax[i].legend() for i in range(2)]

    fig,ax = plt.subplots()
    ax.hist(logscores,label='log_post',color='r');
    ax.hist(logscoresPrior,label='log_prior',alpha=.4,color='y')
    ax.legend()
    fig,ax = plt.subplots()
    ax.hist(l2,label='L2_post',color='r')
    ax.hist(l2Prior,label='L2_prior',alpha=.4,color='y')
    ax.legend()
