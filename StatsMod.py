import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Define a class for an idealised experiment.
class IdealisedExperiment:
    
    # Class constructor taking in distribution 
    # along with any optional arguments.
    def __init__(self,distribution = stats.norm(2,0.3)):
        self.distribution = distribution
 
    # return the result of n repetitions of the experiment.
    def Sample(self,n):
        return self.distribution.rvs(size =n)
    

