

import scipy.stats
import numpy as np
import matplotlib.pylab as plt
import os


best_day_logscore = [-14.91, -98.79, -239.21]

prior_val = 'ds3'#'ds2_long'#,'ds3'#, 'g05_005/'#, 'g1_05/'

names20='''
getMoves_2787/  getMoves_4705/  getMoves_6694/  getMoves_8212/ getMoves_3493/  getMoves_5686/  getMoves_7116/  getMoves_8748/ getMoves_3870/  getMoves_5803/  getMoves_7512/  getMoves_8986/ getMoves_4373/  getMoves_6061/  getMoves_7562/  getMoves_9108/'''

names30='''getMoves_1777/  getMoves_6081/  getMoves_6919/  getMoves_9933/ getMoves_2793/  getMoves_6308/  getMoves_6968/ getMoves_3853/  getMoves_6525/  getMoves_7070/ getMoves_5295/  getMoves_6602/  getMoves_7340/'''

names_g1_05 = '''getMoves_1835/  getMoves_2255/  getMoves_6905/  getMoves_8929/ getMoves_1857/  getMoves_2441/  getMoves_7114/  getMoves_9636/'''

names_g05_005='''getMoves_1414/  getMoves_3889/  getMoves_6239/  getMoves_9074/ getMoves_143/   getMoves_463/   getMoves_7256/  getMoves_9219/'''

names_ds3_05_005 = '''getMoves_141/   getMoves_3157/  getMoves_5010/  getMoves_7901/ getMoves_2576/  getMoves_3942/  getMoves_5591/  getMoves_8918/ getMoves_2833/  getMoves_4401/  getMoves_5878/  getMoves_9453/'''

names_ds2_05_005='''getMoves_2143/  getMoves_4208/  getMoves_6265/  getMoves_9856/ getMoves_3321/  getMoves_4529/  getMoves_6329/  getMoves_9922/ getMoves_4115/  getMoves_5277/  getMoves_6467/'''

if prior_val == 30:
    prior_name = 'n0_30/'
    path = '/home/owainevans/birds/vari_prior2/'
    names = names30
elif prior_val == 20:
    prior_name = 'n0_20/'
    path = '/home/owainevans/birds/vari_prior2/'
    names =  names20
elif prior_val in ['g1_05/','g05_005/']:
    prior_name = prior_val
    path = '/home/owainevans/birds/gamma_prior/'
    names = eval('names_'+prior_val[:-1])
    print 'Prior: ' + prior_name
elif prior_val in 'ds3':
    prior_name = 'g05_005/'
    path = '/home/owainevans/birds/ds3_long_gamma_prior/'
    names = names_ds3_05_005
    print 'Prior: ',prior_name+' ', prior_val
elif prior_val in 'ds2_long':
    prior_name = 'g05_005/'
    path = '/home/owainevans/birds/long_gamma_prior/'
    names = names_ds2_05_005
    print 'Prior: ',prior_name+' ', prior_val

    

names= names.split()
dump_names = []
for name in names:
    filename = path + prior_name + name + 'posteriorRunsDump.py'
    if os.path.isfile(filename):
        dump_names.append(filename)

no_runs = len(dump_names)

run_allParams = []

run_logscore_day_k = []; k=2
means = []


burn_in = 199; cut_off = 200 #min(40,allParams.shape[0])

    
for name in dump_names:
    with open(name,'r') as f: 
        dump = f.read()
    logs = eval( dump[ dump.rfind('=')+1: ] )
    logs = logs[0]

    allParams = []
    allLogscores = []
    for line in logs:
        allParams.append( line[5] )

        day = line[0]
        if day==k:
            allLogscores.append( line[3] )
    allParams = np.array(allParams)
    run_allParams.append(  allParams ) 

    run_logscore_day_k.append( allLogscores )

    means.append( np.mean(allParams,axis=0) )

    cut_allParams = allParams[burn_in:cut_off,:]
    #print 'mean:',
    print np.round(np.mean(cut_allParams,axis=0),2)
    #print 'std:',np.round( np.std(cut_allParams,axis=0), 2)
    #print 'no_samples:',cut_allParams.shape[0]

print 'sample from logs:'
for el in logs[0:10]+logs[-10:]:
    print map(np.round,el)

print '--------\n'

# very little variance after day 1, so we cut off second half
run_length = len( run_allParams[0] )
run_allParams = np.array( [run[ burn_in:cut_off ] for run in run_allParams] )
run_logscore_day_k = [run[ burn_in: ] for run in run_logscore_day_k]

flat_run_allParams = np.array( [line for run in run_allParams for line in run] )


#thin_flat = flat_run_allParams[ range(0,180,10), :]

final_ar = np.array([run[0] for run in run_allParams])
print 'mean of final samples:',np.round(np.mean(final_ar,axis=0),2)
print 'stderr of final samples:',np.round(scipy.stats.sem(final_ar,axis=0),2)

    
fig,ax = plt.subplots(4, 2, figsize=(16,12))
for i in range(4):

    for count,run in enumerate(run_allParams[:2]):
        ax[i,0].hist(run[:,i],bins=20,alpha=.6, label='Run %i (N=%i)'%(count,len(run[:,i])))

        ax[i,0].set_title('Param %i'%i)
        ax[i,0].legend()
        #ax[i,0].set_xticklabels(visible=True)
        
    assert len(flat_run_allParams[:,i]) == no_runs * (cut_off - burn_in)
    ax[i,1].hist(flat_run_allParams[:,i], alpha=.6, bins=30)
    ax[i,1].set_title('Param %i, all runs'%i)
    #ax[i,1].set_xticklabels(visible=True)

for j in range(2):
    ax[0,j].set_title('Prior: Gamma(0.5,.05), Runs: %i, Samples per run: %i.'%(no_runs,cut_off))
fig.tight_layout()

print '\n Mean collapsing samples from all runs:',np.round( np.mean( flat_run_allParams, axis=0), 2)
print '\n std of collapsed samples',np.round( np.std( flat_run_allParams, axis=0), 2)
print '\n Total samples',flat_run_allParams.shape[0]

print 'mean logscores (day %i): '%k, map(np.mean, run_logscore_day_k)
print '\n best logscores: ', best_day_logscore



plt.show()




