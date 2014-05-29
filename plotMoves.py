from utils import *
from train_birds import *
import matplotlib.pylab as plt

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
        ax[i].plot(allMoves[i])
        ax[i].plot(allMovesGround[i])
        ax[i].set_xlim(0,100)

    return allMoves,allMovesGround,fig


def plotSamples(baseDirectory,noIters=1):
    execfile(baseDirectory+'posteriorRunsDump.py')
    runs= posteriorLogs
    
    allSamples=[]; priorTriples=[]
    
    for run in runs:
        for i,triple in enumerate(run):
            if not mod(i,noIters+1)==0:
                allSamples.append(triple)
            else:
                priorTriples.append(triple)

    allSamples=np.array(allSamples)
    logscores = allSamples[:,0]
    l2 = allSamples[:,1]

    allPriorRuns = np.array(allPriorRuns)
    logscoresPrior = allPriorRuns[:,0]
    l2Prior = allPriorRuns[:,1]

    fig,ax = plt.subplots()
    ax.hist(logscores,label='post'); ax.hist(logscoresPrior,label='prior')
    ax.legend()
    fig,ax = plt.subplots()
    ax.hist(l2,label='post'), ax.hist(l2Prior,label='prior'), 
    ax.legend()
