import numpy as np

L = 10                 # box size
N = 100                 # number of particles
eta = 0.72              # volume fraction
rho = N/L**2            # number density
sigma = 2*np.sqrt(eta/N/np.pi)   # particle diameter
dr = sigma/20           # binning interval for the correlation function
Nsteps = 1000   #Number of steps involved in the simulation