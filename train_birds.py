import time
import venture.shortcuts as s
ripl = s.make_puma_church_prime_ripl()

from utils import *
from model import Poisson, num_features
from IPython.parallel import Client
from IPython.parallel.util import interactive

width = 10
height = 10
cells = width * height

dataset = 2
total_birds = 1000 if dataset == 2 else 1000000
name = "%dx%dx%d-train" % (width, height, total_birds)
Y = 1
D = 6 # CHANGE TO 10 from 6

runs = 1

hypers = [5, 10, 10, 10] # these are ground truths

params = {
  "name":name,
  "width":width,
  "height":height,
  "cells":cells,
  "dataset":dataset,
  "total_birds":total_birds,
  "years":range(Y),
  "days":[],
  "hypers":hypers,
}

def makePoisson():
  model = Poisson(ripl, params)
  return model

model = Poisson(ripl, params)

def run(days=None,iterations=5, transitions=1000, baseDirectory=''):
  if days is not None:
    D = days
  
  print "Starting run"
  ripl.clear()
  model.loadAssumes()
  model.updateObserves(0)

  #model.loadObserves()
  #ripl.infer('(incorporate)') #print "Loading observations"
  #for (y, d, i, n) in observes:
  #  ripl.observe('(observe_birds %d %d %d)' % (y, d, i), n)
  logs = []
  t = [time.time()] 
  
  def log():
    dt = time.time() - t[0]
    logs.append((ripl.get_global_logscore(),
                 model.computeScoreDay(model.days[-2]), dt))
    print logs[-1]
    t[0] += dt

  
  for d in range(1, D):
    print "Day %d" % d
    model.updateObserves(d)  # self.days.append(d)
    log()
    
    for i in range(iterations): # iterate inference (could reduce from 5)
      ripl.infer({"kernel":"mh", "scope":d-1, "block":"one", "transitions": Y*transitions})
      log()
      continue
      bird_locs = model.getBirdLocations(days=[d])
      
      for y in range(Y):  # save data for each year
        path = baseDirectory+'/bird_moves%d/%d/%02d/' % (dataset, y, d)
        ensure(path)
        drawBirds(bird_locs[y][d], path + '%02d.png' % i, **params)
  
  model.drawBirdLocations()

  return logs, model



def priorSamples(runs=4):
  priorLogs = []
  
  for run_i in range(runs):
    logs,_ = run(iterations=0)  # scores for each day before inference
    priorLogs.append( logs ) # list of logs for iid draws from prior

  with open('priorRuns.data', 'w') as f:
    f.write(str(priorLogs) )

  return priorLogs


def posteriorSamples( baseDirectory=None,
                     runs=10, iterations=5, transitions=1000):
  
  if baseDirectory is None:
    baseDirectory = 'posteriorSamples_'+str(np.random.randint(10**4))+'/'

  infoString='''\n\n PosteriorSamples: runs=%i,iterations=%i,
  transitions=%i, time=%.3f\n'''%(runs,iterations,transitions,time.time())

  ensure(baseDirectory)
  with open(baseDirectory+'posteriorAppend.dat','a') as f:
    f.write(infoString)
  
  posteriorLogs = []

  for run_i in range(runs):
    
    logs,_ = run(iterations=iterations,transitions=transitions,
                 baseDirectory=baseDirectory) 
    posteriorLogs.append( logs ) # list of logs for iid draws from prior
    
    with open(baseDirectory+'posteriorRuns.dat','a') as f:
      f.write('\n Run #:'+str(run_i)+'\n logs:\n'+str(logs))
  
  # dump whole thing to a file
  with open('posteriorRunsDump.dat', 'w') as f:
    f.write(infoString + str(posteriorLogs) )

  return posteriorLogs



def getMoves():
  basedir = 'getMoves_'+str(np.random.randint(10**4))+'/'
  print 'getMoves basedir:', basedir
  kwargs = dict(days=2,iterations=1,transitions=100,baseDirectory=basedir)
  logs,model = run(**kwargs)
  bird_moves = model.getBirdMoves()
  bird_locs = model.getBirdLocations()

  path = basedir
  ensure(path)
  with open(path+'moves.dat','w') as f:
    f.wrote('moves='+bird_moves)

  return logs,model,bird_moves,bird_locs
  

def checkMoves(moves,no_days=5):
  allMoves = {}
  for day in range(no_days):
    allMoves[day] = []
    for i in range(100):
      fromi = sum( [moves[(0,day,i,j)] for j in range(100)] )
      allMoves[day].append(fromi)
    print 'allMoves total for day %i: %i'%(day,sum(allMoves[day]))
  
  return allMoves





def loadFromPrior():
  model.loadAssumes()

  print "Predicting observes"
  observes = []
  for y in range(Y):
    for d in range(D):
      for i in range(cells):
        n = ripl.predict('(observe_birds %d %d %d)' % (y, d, i))
        observes.append((y, d, i, n))
  
  return observes

#observes = loadFromPrior()
#true_bird_moves = getBirdMoves()

import multiprocessing
#p = multiprocessing.cpu_count() / 2
p = 2

print "Using %d particles" % p

def sweep(r, *args):
  t0 = time.time()
  for y in range(Y):
    r.infer("(pgibbs %d ordered %d 1)" % (y, p))
  
  t1 = time.time()
  #for y in range(Y):
    #r.infer("(mh %d one %d)" % (y, 1))
  r.infer("(mh default one %d)" % 1000)
  
  t2 = time.time()
  
  print "pgibbs: %f, mh: %f" % (t1-t0, t2-t1)
