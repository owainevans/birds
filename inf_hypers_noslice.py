from train_birds import *
priors=['(gamma .5 .05)','(gamma .5 .03)','(gamma 1 .05)']
labels=['g05_005','g05_003','g1_05']
for prior,label in zip(priors,labels):
    model = makeModel(D=3,learnHypers=True,hyperPrior=prior)
    
    out = getMoves(model,slice_hypers=False,transitions=200,iterations=25,
               label='gamma_prior/%s/'%label )

# print 'NEW JOB, prior normal 0 20 with slice hypers'
# model = makeModel(D=6,learnHypers=True,hyperPrior='(normal 0 30)')
# out = getMoves(model,slice_hypers=False,transitions=200,iterations=8,
#                label='prior30_7day_noslice_')
