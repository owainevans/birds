import numpy as np
import matplotlib.pylab as plt
import os

best_day_logscore = [-14.91, -98.79, -239.21]


Reconstruction problem writeup
1. Poisson approximation to multinomial: Motivation
For problems 2 and 3, we switch from a multinomial to a poisson model. For the multinomial generative model, the probability of a bird moving from i to j is a single-trial multinomial over cells j in the reachable neighborhood around j. Since bird moves are iid across birds, the probability of k birds moving i to j is given by an M-trial multinomial over cells j, where M is the number of birds at cell i (on a given day). We approximate this multinomial for moves between cells i and its reachable neighbors j with a set of independent poisson distributions, one for each pair i,j. The lambda parameter for poissoni,j = M*phi(i,j). Thus the expectation for each poisson is the same as for the multinomial. 

On the poisson model, prob_move(i,j) and prob_movej+1) are independent given M and the parameters determining the unnormalized phi probabilities. Hence the number of birds moving from i (including those that move from i to i) will not always be conserved, in contrast to the multinomial model. Lack of conservation is not a problem in this setting because of our observations of bird counts. While our Poisson model will generate sequeneces of states in wich the total bird count changes substantially, such sequences will be rejected by our inference as inconsistent with observed bird counts.

Our immediate motivation for the Poisson model is to speed up inference. The multinomial model makes all latents i,j (for all j reachable from i) dependent. If they are independent, as in our Poisson model, we can make MH proposals to each latent independently (rather than treating a large number of latents as a block). This works better with Venture's built-in proposal distributions for MH (which resample from the prior).

A further extension of our approach (which is not implemented here) would be sum over the latent moves(i,j) pairs and instead just represent the inflow to a given cell. This would be a sum of the incoming independent Poissons and so also a Poisson. Thus we eliminate all but #cell latent variables, which may be a large reduction if O(#cells^2) have positive expected bird count. 
 

(We could also motivate our Poisson model on theoretical grounds. If the starting bird count were unknown and assumed to be Poisson, then the multinomial transition probabilities would be identical to a set of independent Poissons as in our model. Cite: ). 

2. Implementation of Poisson Model in Venture
We describe the key functions in the Venture implementation of the model. Definitions of utility functions are found in model.py (Class Poisson). 
'''
[assume hypers0 5]
[assume hypers1 10]
[assume hypers2 10]
[assume hypers3 10]
; these are the beta parameters. we set them to the groundtruth values. in section on parameter estimation we show how to place a prior on them. 

[assume phi (mem (lambda (y d i j)
              (if (> (cell_dist2 i j) max_dist2) 0
                (let ((fs (lookup features (array y d i j))))
                  (exp (+ (* hypers0 (lookup fs 0)
                           (* hypers1 (lookup fs 1)
                            (* hypers2 (lookup fs 2)
                             (* hypers2 (lookup fs 3)
                                                      ))))))))))]
; we load the features from the given csv file


[assume bird_movements_loc
      (mem (lambda (y d i)
          (let
           ((normalize (foldl + 0 0 cells (lambda (j) (phi y d i j)))))
            (mem (lambda (j)
              (if (= (phi y d i j) 0) 0
                (let ((n (* (count_birds y d i) (/ (phi y d i j) normalize))))
                  (scope_include d (array y d i j)
                    (poisson n))))))))))
# even more simplified version that leaves out foldl and other stuff
[assume bird_movements_loc
      (mem (lambda (y d i)
        (let
         ((normalize (sum (map (lambda (j) (phi y d i j)) all_cells) ) )
            (mem (lambda (j)
              (let ((n (* (count_birds y d i) (/ (phi y d i j) normalize))))
                  (scope_include d (array y d i j)
                    (poisson n)))))))))]

; this function specifies how the latent bird_movements from i to j are generated. we compute a normalizing constant *normalize* by summing over all phi)i.j) pairs. then for a given cell j, we generate its count bird_movements(i,j) by drawing from poisson n, where n is the total number of birds at i * the normalized probability of i,j. 


[assume observe_birds
          (mem (lambda (y d i) (poisson (+ (count_birds y d i) 0.0001))))')


3. Reconstruction inference
For the reconstruction task we implement a filtering inference program on the latent bird_moves. For a given day d, we observe all counts for day d (via the function *observe_birds* above). We then run MH only on the latent states for that day (holding fixed the values of all latent states for previous days). We implement filtering in Venture using scope annotations in the Venture program. All bird_moves(d,i,j) are included in a scope named 'd', where d is the number of the day. See definiton of bird_movements_loc, above.

We then use these scopes in the inference program:
for d in days:
   model.updateObserves(d)  # observe_birds(d,i) for each cell i for day d  
   model.ripl.infer('(mh d one transitions)')
   # where transitions is a parameter we control for number of MH transitions, d is the day, 'one' specifies that we transition involves picking from all variables in scope d uniformly for the MH proposal.


          





prior_val = 'ds2_long'#,'ds3'#, 'g05_005/'#, 'g1_05/'

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


burn_in = 180; cut_off = 200 #min(40,allParams.shape[0])

    
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
    print 'mean:',np.round(np.mean(cut_allParams,axis=0),2)
    print 'std:',np.round( np.std(cut_allParams,axis=0), 2)
    print 'no_samples:',cut_allParams.shape[0]


# very little variance after day 1, so we cut off second half
run_length = len( run_allParams[0] )
run_allParams = np.array( [run[ burn_in:cut_off ] for run in run_allParams] )
run_logscore_day_k = [run[ burn_in: ] for run in run_logscore_day_k]

flat_run_allParams = np.array( [line for run in run_allParams for line in run] )


thin_flat = flat_run_allParams[ range(0,180,10), :]

    
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




