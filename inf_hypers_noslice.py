from train_birds import *
model = makeModel(D=7,learnHypers=True,hyperPrior='(normal 0 10)')
out = getMoves(model,slice_hypers=False,transitions=400,iterations=3,
               label='prior10_7day_noslice_')

print 'NEW JOB, prior normal 0 20 with slice hypers'
model = makeModel(D=7,learnHypers=True,hyperPrior='(normal 0 20)')
out = getMoves(model,slice_hypers=False,transitions=400,iterations=3,
               label='prior20_7day_noslice_')
