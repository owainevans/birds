from train_birds import *
priors=['(gamma .5 .05)']
labels=['g05_005']
for prior,label in zip(priors,labels):
    model = makeModel(dataset=2, D=3, learnHypers=True, hyperPrior=prior)
    
    out = getMoves(model,slice_hypers=False,transitions=100,iterations=100,
               label='new_cycle/%s/'%label )


