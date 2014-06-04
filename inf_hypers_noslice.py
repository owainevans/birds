from train_birds import *
priors=['(gamma 1 .1)']
labels=['g1_01']
for prior,label in zip(priors,labels):
    model = makeModel(dataset=2, D=3, learnHypers=True, hyperPrior=prior)
    
    out = getMoves(model,slice_hypers=False,transitions=(100,100,25),iterations=50,
               label='new_cycle/%s/'%label )


[assume phi (mem (lambda (y d i j)
              (if (> (cell_dist2 i j) max_dist2) 0
                (let ((fs (lookup features (array y d i j))))
                  (exp (+ (* hypers0 (lookup fs 0))
                          (* hypers1 (lookup fs 1))
                          (* hypers2 (lookup fs 2))
                          (* hypers2 (lookup fs 3)) ))))))]


[assume phi (mem (lambda (y d i j)
              (if (> (cell_dist2 i j) max_dist2) 0
                (let ((fs (lookup all_features (array y d i j))))
                  (exp (dot_product hypers fs ))))))]  
