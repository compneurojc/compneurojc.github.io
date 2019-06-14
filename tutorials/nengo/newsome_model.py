import nengo
import numpy as np
from nengo.dists import Uniform


# following are required for computing 
# the conditioned avg response
cond_colour = []
cond_motion = []


#Monkey A inputs
class Experiment(object):
    def __init__(self, seed=None, interval=0.75, delay=0.75, blk_length=36):
        self.rng = np.random.RandomState(seed=seed)
        self.interval = interval
        self.delay = delay
        self.blk_interval = blk_length * (interval + delay)
        self.col_index = None
        self.mot_index = None
        self.cont_index = None
        self.colour = 0
        self.motion = 0
        self.context = self.rng.choice([2,-2])
        
    def context_in(self, t): 
        index = int(t / self.blk_interval) % 2
        if self.cont_index != index:
            self.context *= -1
            self.cont_index = index
        return self.context    

    def colour_in(self, t):
        index = int(t / self.interval) % 6 
        if self.col_index != index:
            if self.colour != 0:
                self.colour = 0
            else:
                self.colour = self.rng.choice([0.06,0.18,0.50,-0.06,-0.18,-0.50])
                cond_colour.append(self.colour)
            self.col_index = index
        return self.colour   
        
    def motion_in(self, t):
        index = int(t / self.interval) % 6 
        if self.mot_index != index:
            if self.motion != 0:
                self.motion = 0
            else:
                self.motion = self.rng.choice([0.05,0.15,0.50,-0.05,-0.15,-0.50])
                cond_motion.append(self.motion)
            self.mot_index = index
        return self.motion    
    
    def correct_ans(self, t):
        if self.context == -2:
            if self.colour == 0:
                ans = 0
            elif self.colour>0:
                ans = 1
            else:
                ans = -1
        else:
            if self.motion == 0:
                ans = 0
            elif self.motion>0:
                ans = 1
            else:
                ans = -1   
        return ans


seed=31     
model = nengo.Network(seed=seed)  #This seed could also be different
with model:
    exp = Experiment(seed=seed)
    
    stim_colour = nengo.Node(exp.colour_in)
    stim_motion = nengo.Node(exp.motion_in)
    stim_context = nengo.Node(exp.context_in)

    
    pfc = nengo.Ensemble(n_neurons=1000, dimensions=4, 
                         radius=2, max_rates=Uniform(20, 120))
        
    thresh = nengo.Ensemble(n_neurons=200, dimensions=1, radius=1, 
                            intercepts=Uniform(0.2, 1),
                            max_rates=Uniform(20,120))
    
    tau = 0.2
    nengo.Connection(stim_colour, pfc[0], synapse=tau) 
    nengo.Connection(stim_motion, pfc[1], synapse=tau) 
    nengo.Connection(stim_context, pfc[2], synapse=tau)
    
    #multiplicative gating 
    def response(x):
        colour, motion, context, choice = x
        choice = 1*((2-context)*colour + (2+context)*motion)    # mem effect: +.75*choice
        return 0, 0, 0, choice
        
    nengo.Connection(pfc, pfc, synapse=0.1, function=response)
    nengo.Connection(pfc[3], thresh)
    
    corr_ans = nengo.Node(exp.correct_ans, size_out=1)
