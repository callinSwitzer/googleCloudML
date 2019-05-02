import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import os
import pandas as pd
import seaborn as sns
from scipy.integrate import odeint
import random
import time
from datetime import datetime
import sys
from multiprocessing import Pool, cpu_count
import simUtils # note that this is a custom-written file 
import importlib
import functools

print(sys.version)

now = datetime.now()
print("last run on " + str(now))


np.random.seed(123)

from collections import OrderedDict

globalDict = OrderedDict({"bhead": 0.507,
            "ahead": 0.908,
            "bbutt": 0.1295,
            "abutt": 1.7475, 
            "rho": 1, 
            "rhoA": 0.00118, 
            "muA": 0.000186, 
            "L1": 0.908, 
            "L2": 1.7475,  
            "L3": 0.75,
            "K": 29.3,
            "c":  14075.8,
            "g": 980.0,
            "betaR":  0.0,
            "nstep": 100,
            "nrun" : 2500  # (max) number of  trajectories.
            })

# Calculated variables
globalDict['m1'] = globalDict['rho']*(4/3)*np.pi*(globalDict['bhead']**2)*globalDict['ahead']
globalDict["m2"] = globalDict["rho"]*(4/3)*np.pi*(globalDict["bbutt"]**2)*globalDict["abutt"]
globalDict["echead"] = globalDict["ahead"]/globalDict["bhead"]
globalDict['ecbutt'] = globalDict['abutt']/globalDict['bbutt']
globalDict['I1'] = (1/5)*globalDict['m1']*(globalDict['bhead']**2)*(1 + globalDict['echead']**2)
globalDict['I2'] = (1/5)*globalDict['m2']*(globalDict['bbutt']**2)*(1 + globalDict['ecbutt']**2)
globalDict['S_head'] = np.pi*globalDict['bhead']**2
globalDict['S_butt'] = np.pi*globalDict['bbutt'] **2
t = np.linspace(0, 0.02, num = globalDict["nstep"], endpoint = True)

# ranges for variables
rangeDict = {"Fmin": 0,
             "Fmax": 44300,
            "alphaMin":  0,
             "alphaMax":2*np.pi, 
            "tau0Min": -100000, 
             "tau0Max": 100000}

# ranges for initial conditions
ranges = np.array([[rangeDict["Fmin"], rangeDict["Fmax"]], 
                   [rangeDict["alphaMin"], rangeDict["alphaMax"]], 
                   [rangeDict["tau0Min"], rangeDict["tau0Max"] ]])


# generate initial conditions for state 0
# x,xd,y,yd,theta,thetad,phi,phid
state0_ICs = [0.0, 0.0001, 0.0, 0.0001, 0.0, 0.0001, 0.0, 0.0001]

# F, alpha, tau
FAlphaTau_list = np.random.uniform(ranges[:, 0], ranges[:, 1], 
                                   size=(globalDict["nrun"], ranges.shape[0]))

# convert dict to list, since @jit works better with lists
globalList = [ v for v in globalDict.values() ]

# try to run simulations in parallel
p = Pool(cpu_count() - 1)
stt = time.time()
bb = p.map(functools.partial(simUtils.flyBug_dictInput, t=t, 
                              state0_ICs = state0_ICs, 
                              FAlphaTau_list= FAlphaTau_list, 
                              globalList = globalList), range(globalDict["nrun"]))
print(time.time() - stt)
p.close()
p.join()