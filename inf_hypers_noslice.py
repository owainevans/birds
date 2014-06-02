from train_birds import *
priors=['(gamma .25 .05)']
labels=['g025_005']
for prior,label in zip(priors,labels):
    model = makeModel(dataset=3, D=3, learnHypers=True, hyperPrior=prior)
    
    out = getMoves(model,slice_hypers=False,transitions=200,iterations=40,
               label='gamma_prior_ds3/%s/'%label )

# print 'NEW JOB, prior normal 0 20 with slice hypers'
# model = makeModel(D=6,learnHypers=True,hyperPrior='(normal 0 30)')
# out = getMoves(model,slice_hypers=False,transitions=200,iterations=8,
#                label='prior30_7day_noslice_')
