from train_birds import *
priors=['(normal 0 20)','(normal 0 30)','(normal 0 100)']
labels=['n0_20','n0_30','n0_100']
for prior,label in zip(priors,labels):
    model = makeModel(D=4,learnHypers=True,hyperPrior=prior)
    
    out = getMoves(model,slice_hypers=False,transitions=200,iterations=50,
               label='vari_prior2/%s/'%label )

# print 'NEW JOB, prior normal 0 20 with slice hypers'
# model = makeModel(D=6,learnHypers=True,hyperPrior='(normal 0 30)')
# out = getMoves(model,slice_hypers=False,transitions=200,iterations=8,
#                label='prior30_7day_noslice_')
