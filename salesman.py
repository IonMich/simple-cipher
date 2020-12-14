#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 14:26:47 2018

@author: yannis
"""
###BLECHA'S CODE
### If you're using Anaconda python, the vpython visualization module
### (updated equivalent of the 'visual' module) can be installed with 
### 'conda -c vpython vpython' at the command line.
### Note that the 'random' module has to be imported after vpython to work.

### Comment this out (and similarly marked lines below)
### if not using visualization module:
from vpython import canvas, sphere,vec,curve,vector,rate
###
import numpy as np
import random

N = 25
R = 0.02
Tmax = 10.0
Tmin = 1e-3
tau = 1e4

# Function to calculate the magnitude of a vector
def mag(x):
    return np.sqrt(x[0]**2+x[1]**2)

# Function to calculate the total length of the tour
def distance():
    s = 0.0
    for i in range(N):
        s += mag(r[i+1]-r[i])
    return s

# Choose N city locations and calculate the initial distance
r = np.empty([N+1,2],float)
for i in range(N):
    r[i,0] = random.random()
    r[i,1] = random.random()
r[N] = r[0]
D = distance()

### Comment this out if not using visualization module:
# Set up the graphics
canvas(center=vector(0.5,0.5,0))
for i in range(N):
    sphere(pos=vec(r[i,0],r[i,1],0),radius=R)
l = curve(pos=[vec(r[i,0],r[i,1],0) for i in range(N+1)], 
          radius=R/2, retain=N+1)
###

# Main loop
t = 0
T = Tmax

while T>Tmin:

    # Cooling
    t += 1
    T = Tmax*np.exp(-t/tau)

    ### Comment this out if not using visualization module:
    # Update the visualization every 100 moves
    if t%100==0:
        for i in range(N+1):
            l.append(pos=vec(r[i,0],r[i,1],0),retain=N+1)
        rate(25)
    ###

    # Choose two cities to swap and make sure they are distinct
    i,j = random.randrange(1,N), random.randrange(1,N)
    while i==j:
        j = random.randrange(1,N)

    # Swap them and calculate the change in distance
    oldD = D
    r[i,0],r[j,0] = r[j,0],r[i,0]
    r[i,1],r[j,1] = r[j,1],r[i,1]
    D = distance()
    deltaD = D - oldD
    print(deltaD)
    # If the move is rejected, swap them back again
    if random.random()>np.exp(-deltaD/T):
        r[i,0],r[j,0] = r[j,0],r[i,0]
        r[i,1],r[j,1] = r[j,1],r[i,1]
        D = oldD

# If not using the visualization module, you'll want to plot
# your result here.
