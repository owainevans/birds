from train_birds import *
model = makeModel(D=6,learnHypers=True,hyperPrior='(normal 0 30)')
out = getMoves(model,slice_hypers=False,transitions=200,iterations=8,
               label='prior30_7day_noslice_')

print 'NEW JOB, prior normal 0 20 with slice hypers'
model = makeModel(D=6,learnHypers=True,hyperPrior='(normal 0 30)')
out = getMoves(model,slice_hypers=False,transitions=200,iterations=8,
               label='prior30_7day_noslice_')
