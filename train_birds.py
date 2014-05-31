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
D = 12 # run inference on days 1 to (D-1)

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
  "maxDay":D}


def hypersInfer():
  global hypers
  hypers = ['(gamma 5 1)']*4
  params['hypers']=hypers


if __name__ == '__main__':
  hypersInfer()
  model = Poisson(ripl, params)
  

def log(t,day,iteration,transitions,ripl):
  dt = time.time() - t[0]
  data = (day, iteration, transitions, ripl.get_global_logscore(),
          model.computeScoreDay(model.days[-2]), dt)
  print data
  t[0] += dt
  return data


def stepThru():
  ripl.clear()
  model.loadAssumes()
  model.updateObserves(0)

  t=[time.time()]
  logs = []
  daysRange = range(1,D)

  for d in daysRange:
    print 'day %d'% d
    model.updateObserves(d)
    logs.append( log(t,d,0,10,model.ripl) )
    yield
    model.forceBirdMoves(d)
    #model.ripl.infer(10)
    logs.append( log(t,d,1,10,model.ripl) )
    yield
  

def run(days=None,iterations=5, transitions=1000, baseDirectory=''):
  
  print "\n Starting run"
  print 'params:',params,'\n'
  ripl.clear()
  model.loadAssumes()
  model.updateObserves(0)

  logs = []
  t = [time.time()]
  daysRange = range(1,D) if days is None else range(1,days)
  
    
  for d in daysRange:
    print "Day %d" % d
    model.updateObserves(d)  # self.days.append(d)
    logs.append( log(t,d,0,transitions,ripl) )
    
    for i in range(iterations): # iterate inference (could reduce from 5)
      ripl.infer({"kernel":"mh", "scope":d-1, "block":"one", "transitions": Y*transitions})
      logs.append( log(t,d,i+1,transitions,ripl) )
      continue
      bird_locs = model.getBirdLocations(days=[d])
      
      for y in range(Y):  # save data for each year
        path = baseDirectory+'/bird_moves%d/%d/%02d/' % (dataset, y, d)
        ensure(path)
        drawBirds(bird_locs[y][d], path + '%02d.png' % i, **params)
        
  
  model.drawBirdLocations()

  return logs, model




def posteriorSamples(runs=10, baseDirectory=None, days=None,
                      iterations=5, transitions=1000):
  
  if baseDirectory is None:
    baseDirectory = 'posteriorSamples_'+str(np.random.randint(10**4))+'/'

  infoString='''PostSamples:runs=%i,iters=%i, transitions=%i,
  time=%.3f\n'''%(runs,iterations,transitions,time.time())

  ensure(baseDirectory)
  with open(baseDirectory+'posteriorRuns.dat','a') as f:
    f.write(infoString)
  
  posteriorLogs = []

  for run_i in range(runs):
    
    logs,lastModel = run(days=days,iterations=iterations,
                         transitions=transitions,
                         baseDirectory=baseDirectory)
    posteriorLogs.append( logs ) # list of logs for iid draws from prior
    
    with open(baseDirectory+'posteriorRuns.dat','a') as f:
      f.write('\n Run #:'+str(run_i)+'\n logs:\n'+str(logs))
  

  with open(baseDirectory+'posteriorRunsDump.py', 'w') as f:
    info = 'info="""%s"""'%infoString
    #params = '\nparams=%s'%params
    logs = '\n\nlogs=%s'%posteriorLogs
    f.write(info+logs) # dump to file

    
  return posteriorLogs,lastModel



def getMoves(days=None,transitions=1000,iterations=1,label=''):
  
  basedir = label + 'getMoves_'+str(np.random.randint(10**4))+'/'
  print '====\n getMoves basedir:', basedir
  print '\n getMoves args:'
  print 'days=%s,transitions=%i,iterations=%i'%(str(days), transitions,iterations)
  
  kwargs = dict(runs=1, days=days, iterations=iterations,
                transitions=transitions,baseDirectory=basedir)
  posteriorLogs,lastModel = posteriorSamples(**kwargs)
  bird_moves = model.getBirdMoves()
  bird_locs = model.getBirdLocations()

  ensure(basedir)
  with open(basedir+'moves.dat','w') as f:
    f.write('moves='+str(bird_moves))

  return posteriorLogs,model,bird_moves,bird_locs
  

def checkMoves(moves,no_days=5):
  allMoves = {}
  for day in range(no_days):
    allMoves[day] = []
    for i in range(100):
      fromi = sum( [moves[(0,day,i,j)] for j in range(100)] )
      allMoves[day].append(fromi)
      
    if day<6:
      print 'allMoves total for day %i (up to 6): %i'%(day,
                                                       sum(allMoves[day]))
  
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
