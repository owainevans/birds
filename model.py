import venture.shortcuts as s
from utils import *
from venture.unit import VentureUnit
from venture.ripl.ripl import _strip_types
from itertools import product

num_features = 4
    


def day_features(features,width,y=0,d=0,summary=None):
  lst = [features[(y,d,i,j)] for (i,j) in product(range(cells),range(cells))]
  return lst


def loadFeatures(dataset, name, years, days, maxDay=None):
  
  features_file = "data/input/dataset%d/%s-features.csv" % (dataset, name)
  print "Loading features from %s" % features_file
  
  features = readFeatures(features_file, maxYear= max(years)+1, maxDay=maxDay)
  
  for (y, d, i, j) in features.keys():
    if y not in years:# or d not in days:
      del features[(y, d, i, j)]
  
  return toVenture(features)


def loadObservations(ripl, dataset, name, years, days):
  observations_file = "data/input/dataset%d/%s-observations.csv" % (dataset, name)
  observations = readObservations(observations_file)

  for y in years:
    for (d, ns) in observations[y]:
      if d not in days: continue
      for i, n in enumerate(ns):
        #print y, d, i
        ripl.observe('(observe_birds %d %d %d)' % (y, d, i), n)


def drawBirdLocations(bird_locs,name,years,days,height,width):
    for y in years:
      path = 'bird_moves_%s/%d/' % (name, y)
      ensure(path)
      for d in days:
        drawBirds(bird_locs[y][d], path+'%02d.png'%d, height=height, width=width)



class OneBird(VentureUnit):
  
  def __init__(self, ripl, params):
    self.name = params['name']
    self.cells = params['cells']
    self.years = params['years']
    self.days = params['days']

    if 'features' in params:
      self.features = params['features']
      self.num_features = params['num_features']
    else:
      self.features = loadFeatures(1, self.name, self.years, self.days)
      self.num_features = num_features

    self.learnHypers = params['learnHypers']
    if not self.learnHypers:
      self.hypers = params['hypers']
      
    self.load_observes_file=params.get('load_observes_file',True)
    self.num_birds=params.get('num_birds',1)

    super(OneBird, self).__init__(ripl, params)


# automatically load assumes onto ripl - we can later cancel this if dealing
# with large number of features
  def makeAssumes(self):
    self.loadAssumes(self) # note this hackish thing where we send self and so
    # ripl takes value self, so we use venture unit assume which is wrapper
    # round assume and then assumes are there for analytics coz we are mutating
    # the relevant datastructure. gnarly.
  
  def makeObserves(self):
    if self.load_observes_file:
      self.loadObserves(ripl=self)
    else:
      pass
  
  def loadAssumes(self, ripl = None):
    if ripl is None:
      ripl = self.ripl
    
    print "Loading assumes"

    ripl.assume('filter',"""
      (lambda (pred lst)
        (if (not (is_pair lst)) (list)
          (if (pred (first lst))
            (pair (first lst) (filter pred (rest lst)) )
            (filter pred (rest lst)))))""")
    ripl.assume('map',"""
      (lambda (f lst)
        (if (is_nil lst) (list)
          (pair (f (first lst)) (map f (rest lst))) ) )""")

    if not self.learnHypers:
      for k, value_k in enumerate(self.hypers):
        ripl.assume('hypers%d' % k,  value_k)
    else:
      for k in range(self.num_features):
        ripl.assume('scale', '(scope_include (quote hypers) (quote scale) (gamma 1 1))')
        ripl.assume('hypers%d' % k, '(scope_include (quote hypers) %d (* scale (normal 0 10)))' % k)

    
    ripl.assume('features', self.features)
    ripl.assume('num_birds',self.num_birds)
    
    bird_ids = ' '.join(map(str,range(self.num_birds)))
    ripl.assume('bird_ids','(list %s)'%bird_ids)

    ripl.assume('phi', """
      (mem (lambda (y d i j)
        (let ((fs (lookup features (array y d i j))))
          (exp %s))))"""
       % fold('+', '(* hypers_k_ (lookup fs _k_))', '_k_', self.num_features))

    ripl.assume('get_bird_move_dist',
      '(mem (lambda (y d i) ' +
        fold('simplex', '(phi y d i j)', 'j', self.cells) +
      '))')
    
    ripl.assume('cell_array', fold('array', 'j', 'j', self.cells))

    ripl.assume('move', """
      (lambda (y d i)
        (let ((dist (get_bird_move_dist y d i)))
          (scope_include (quote move) (array y d)
            (categorical dist cell_array))))""")

    ripl.assume('move2', """
      (mem (lambda (bird_id y d i)
        (let ((dist (get_bird_move_dist y d i)))
          (scope_include (quote move) (array bird_id y d i)
            (categorical dist cell_array)))))""")


# for any day=0, bird starts at 0
    ripl.assume('get_bird_pos', """
      (mem (lambda (y d)
        (if (= d 0) 0
          (move y (- d 1) (get_bird_pos y (- d 1))))))""")

    ripl.assume('count_birds', """
      (lambda (y d i)
        (if (= (get_bird_pos y d) i) 1 0))""")

    ripl.assume('observe_birds', '(lambda (y d i) (poisson (+ (count_birds y d i) 0.0001)))')

# multi-bird version
    ripl.assume('get_bird_pos2', """
      (mem (lambda (bird_id y d)
        (if (= d 0) 0
          (move2 bird_id y (- d 1) (get_bird_pos2 bird_id y (- d 1))))))""")

    ripl.assume('all_bird_pos',"""
       (mem (lambda (y d) 
         (map (lambda (bird_id) (get_bird_pos2 bird_id y d)) bird_ids)))""")

    ripl.assume('count_birds2', """
      (lambda (y d i)
        (size (filter 
                 (lambda (bird_id) (= i (get_bird_pos2 bird_id y d)) )
                  bird_ids) ) ) """)

    ripl.assume('count_birds22', """
      (mem (lambda (y d i)
        (size (filter
                (lambda (x) (= x i)) (all_bird_pos y d)))))""" )

    ripl.assume('observe_birds2', '(lambda (y d i) (poisson (+ (count_birds2 y d i) 0.0001)))')

  def bird_to_pos(self,day,year=0,hist=False):
    l=[]
    for bird_id in self.ripl.sample('bird_ids'):
      l.append(ripl.sample('(get_bird_pos2 %i %i %i)'%(bird_id,year,day)))

    all_bird_l = self.ripl.sample('(all_bird_pos year day)')
    assert all( np.array(all_bird_l)==np.array(l) )

    if hist:
      hist,_ = np.histogram(l,bins=range(self.cells))
      return hist
    else:
      return l


  def getBirdLocations(self, years=None, days=None):
    if years is None: years = self.years
    if days is None: days = self.days
    
    bird_locations = {}
    for y in years:
      bird_locations[y] = {}
      for d in days:
        bird_locations[y][d] = bird_to_pos(d,year=y,hist=True)
    
    return bird_locations



  def loadObserves(self, ripl = None):
    if ripl is None:
      ripl = self.ripl
  
    observations_file = "data/input/dataset%d/%s-observations.csv" % (1, self.name)
    observations = readObservations(observations_file)

    self.unconstrained = []

    for y in self.years:
      for (d, ns) in observations[y]:
        if d not in self.days: continue
        if d == 0: continue
        
        loc = None
        
        for i, n in enumerate(ns):
          if n > 0:
            loc = i
            break
        
        if loc is None:
          self.unconstrained.append((y, d-1))
          #ripl.predict('(get_bird_pos %d %d)' % (y, d))
        else:
          ripl.observe('(get_bird_pos %d %d)' % (y, d), loc)
  
  def inferMove(self, ripl = None):
    if ripl is None:
      ripl = self.ripl
    
    for block in self.unconstrained:
      ripl.infer({'kernel': 'gibbs', 'scope': 'move', 'block': block, 'transitions': 1})
  


class Poisson(VentureUnit):

  def __init__(self, ripl, params):
    self.name = params['name']
    self.width = params['width']
    self.height = params['height']
    self.cells = params['cells']
    
    self.dataset = params['dataset']
    self.total_birds = params['total_birds']
    self.years = params['years']
    self.days = params['days']
    self.hypers = params["hypers"]
    self.learnHypers = True if isinstance(self.hypers[0],str) else False
    self.ground = readReconstruction(params) if 'ground' in params else None
    self.maxDay = params.get('maxDay',None)

    if 'features' in params:
      self.features = params['features']
      self.num_features = params['num_features']
    else:
      self.features = loadFeatures(self.dataset, self.name, self.years, self.days,
                                   maxDay=self.maxDay)
      self.num_features = num_features
      
    self.load_observes_file=params.get('load_observes_file',True)

    val_features = self.features['value']
    self.parsedFeatures = {k:_strip_types(v) for k,v in val_features.items() }


    super(Poisson, self).__init__(ripl, params)


  def feat_i(y,d,i,feat=2):
    'Input *feat in range(3) (default=wind), return all values i,j for fixed i'
    return [self.parsedFeatures[(y,d,i,j)][feat] for j in range(100)] 


  def loadAssumes(self, ripl = None):
    if ripl is None:
      ripl = self.ripl
    
    print "Loading assumes"
    
    ripl.assume('total_birds', self.total_birds)
    ripl.assume('cells', self.cells)

    #ripl.assume('num_features', num_features)

    
    if not self.learnHypers:
      for k, b in enumerate(self.hypers):
        ripl.assume('hypers%d' % k,  b)
    else:
      for k, prior in enumerate(self.hypers):
        ripl.assume('hypers%d' % k,'(scope_include (quote hypers) 0 %s )'%prior)
        #ripl.assume('hypers%d' % k,'(scope_include (quote hypers) %d %s )'%(k,prior) )

    ripl.assume('features', self.features)

    ripl.assume('width', self.width)
    ripl.assume('height', self.height)
    ripl.assume('max_dist2', '18')

    ripl.assume('cell2X', '(lambda (cell) (int_div cell height))')
    ripl.assume('cell2Y', '(lambda (cell) (int_mod cell height))')
    #ripl.assume('cell2P', '(lambda (cell) (make_pair (cell2X cell) (cell2Y cell)))')
    ripl.assume('XY2cell', '(lambda (x y) (+ (* height x) y))')

    ripl.assume('square', '(lambda (x) (* x x))')

    ripl.assume('dist2', """
      (lambda (x1 y1 x2 y2)
        (+ (square (- x1 x2)) (square (- y1 y2))))""")

    ripl.assume('cell_dist2', """
      (lambda (i j)
        (dist2
          (cell2X i) (cell2Y i)
          (cell2X j) (cell2Y j)))""")
    
    # phi is the unnormalized probability of a bird moving from cell i to cell j on day d
    ripl.assume('phi', """
      (mem (lambda (y d i j)
        (if (> (cell_dist2 i j) max_dist2) 0
          (let ((fs (lookup features (array y d i j))))
            (exp %s)))))"""
            % fold('+', '(* hypers__k (lookup fs __k))', '__k', self.num_features))
    

    ripl.assume('get_bird_move_dist', """
      (lambda (y d i)
        (lambda (j)
          (phi y d i j)))""")
    
    ripl.assume('foldl', """
      (lambda (op x min max f)
        (if (= min max) x
          (foldl op (op x (f min)) (+ min 1) max f)))""")

    ripl.assume('multinomial_func', """
      (lambda (n min max f)
        (let ((normalize (foldl + 0 min max f)))
          (mem (lambda (i)
            (poisson (* n (/ (f i) normalize)))))))""")
                  
    ripl.assume('count_birds', """
      (mem (lambda (y d i)
        (if (= d 0) (if (= i 0) total_birds 0)""" +
          fold('+', '(get_birds_moving y (- d 1) __j i)', '__j', self.cells) + ")))")
    
    # bird_movements_loc
    # if no birds at i, no movement to any j from i
    # normalize is normalizing constant for probms from i
    # n is product of count at position i * normed probility of i to j
    # return lambda that takes j and return poisson of this n
  
    ripl.assume('bird_movements_loc', """
      (mem (lambda (y d i)
        (if (= (count_birds y d i) 0)
          (lambda (j) 0)
          (let ((normalize (foldl + 0 0 cells (lambda (j) (phi y d i j)))))
            (mem (lambda (j)
              (if (= (phi y d i j) 0) 0
                (let ((n (* (count_birds y d i) (/ (phi y d i j) normalize))))
                  (scope_include d (array y d i j)
                    (poisson n))))))))))""")

    

## ADD REPEAT AND FILTER
    # ripl.assume('multinomial_mem', """
    #   (lambda (trials simplex)
    #       (let ( (draws (repeat (lambda() (categorical simplex) ) ) ) )
    #       (mem (lambda (i)
    #          (count_instances i draws) ) ) ) )"""

    # ripl.assume('count_instances',"""
    #             (lambda (i lst) 
    #                   (size (filter (lambda (x) (= x i)) lst) ) )""")

    # ripl.assume('bird_movements_loc', """
    #   (mem (lambda (y d i)
    #     (if (= (count_birds y d i) 0)
    #       (lambda (j) 0)
    #         (let ((normalize (foldl + 0 0 cells (lambda (j) (phi y d i j)))))
    #           (scope_include d (array y d i)
    #             (let ( (counts (multinomial_mem (count_birds y d i) normalize)) )
    #               (mem (lambda (j) (counts j) ) ) ) ) ) ) ) ) """ )
    

    #ripl.assume('bird_movements', '(mem (lambda (y d) %s))' % fold('array', '(bird_movements_loc y d __i)', '__i', self.cells))
    
    ripl.assume('observe_birds', '(mem (lambda (y d i) (poisson (+ (count_birds y d i) 0.0001))))')

    # returns number birds from i,j (we want to force this value)
    ripl.assume('get_birds_moving', """
      (lambda (y d i j)
        ((bird_movements_loc y d i) j))""")
    
    ripl.assume('get_birds_moving1', '(lambda (y d i) %s)' % fold('array', '(get_birds_moving y d i __j)', '__j', self.cells))
    ripl.assume('get_birds_moving2', '(lambda (y d) %s)' % fold('array', '(get_birds_moving1 y d __i)', '__i', self.cells))
    ripl.assume('get_birds_moving3', '(lambda (d) %s)' % fold('array', '(get_birds_moving2 __y d)', '__y', len(self.years)))
    ripl.assume('get_birds_moving4', '(lambda () %s)' % fold('array', '(get_birds_moving3 __d)', '__d', len(self.days)-1))
  
  def loadObserves(self, ripl = None):
    if ripl is None:
      ripl = self.ripl
    
    print "Loading observations"
    loadObservations(ripl, self.dataset, self.name, self.years, self.days)
  
  def loadModel(self, ripl = None):
    if ripl is None:
      ripl = self.ripl
    self.loadAssumes(ripl)
    self.loadObserves(ripl)
  
  def makeAssumes(self):
    self.loadAssumes(ripl=self)
  
  def makeObserves(self):
    self.loadObserves(ripl=self)
  
  def updateObserves(self, d):
    self.days.append(d)
    #if d > 0: self.ripl.forget('bird_moves')
    
    loadObservations(self.ripl, self.dataset, self.name, self.years, [d])
    self.ripl.infer('(incorporate)')
    #self.ripl.predict(fold('array', '(get_birds_moving3 __d)', '__d', len(self.days)-1), label='bird_moves')
  
  def getBirdLocations(self, years=None, days=None):
    if years is None: years = self.years
    if days is None: days = self.days
    
    bird_locations = {}
    for y in years:
      bird_locations[y] = {}
      for d in days:
        bird_locations[y][d] = [self.ripl.sample('(count_birds %d %d %d)' % (y, d, i)) for i in range(self.cells)]
    
    return bird_locations
  

  def drawBirdLocations(self):
    bird_locs = self.getBirdLocations()
  
    for y in self.years:
      path = 'bird_moves%d/%d/' % (self.dataset, y)
      ensure(path)
      for d in self.days:
        drawBirds(bird_locs[y][d], path + '%02d.png' % d, **self.parameters)
  

  def getBirdMoves(self):
    
    bird_moves = {}
    
    for d in self.days[:-1]:
      bird_moves_raw = self.ripl.sample('(get_birds_moving3 %d)' % d)
      for y in self.years:
        for i in range(self.cells):
          for j in range(self.cells):
            bird_moves[(y, d, i, j)] = bird_moves_raw[y][i][j]
    
    return bird_moves
  
  def forceBirdMoves(self,d,cell_limit=100):
    # currently ignore including years also
    detvalues = 0
    
    for i in range(self.cells)[:cell_limit]:
      for j in range(self.cells)[:cell_limit]:
        ground = self.ground[(0,d,i,j)]
        current = self.ripl.sample('(get_birds_moving 0 %d %d %d)'%(d,i,j))
        
        if ground>0 and current>0:
          self.ripl.force('(get_birds_moving 0 %d %d %d)'%(d,i,j),ground)
          print 'force: moving(0 %d %d %d) from %f to %f'%(d,i,j,current,ground)
          
    #     try:
    #       self.ripl.force('(get_birds_moving 0 %d %d %d)'%(d,i,j),
    #                       self.ground[(0,d,i,j)] )
    #     except:
    #       detvalues += 1
    # print 'detvalues total =  %d'%detvalues

  def computeScoreDay(self, d):
    bird_moves = self.ripl.sample('(get_birds_moving3 %d)' % d)
    
    score = 0
    
    for y in self.years:
      for i in range(self.cells):
        for j in range(self.cells):
          score += (bird_moves[y][i][j] - self.ground[(y, d, i, j)]) ** 2
    
    return score
  
  def computeScore(self):
    infer_bird_moves = self.getBirdMoves()

    score = 0
    
    for key in infer_bird_moves:
      score += (infer_bird_moves[key] - self.ground[key]) ** 2

    return score




