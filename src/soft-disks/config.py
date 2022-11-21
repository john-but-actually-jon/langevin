import numpy as np

scaler = 1e20

k_b_real = 1.380649e-23      # Real units (m^2 kg s^-2 K^-1)
k_b = k_b_real * scaler
T = 274                 # Kelvin
m = 1                   # Mass of particles
L = 10                  # box size in nm?
N = 100                 # number of particles
eta = 0.72              # volume fraction
rho = N/L**2            # number density
sigma = 2*np.sqrt(eta/N/np.pi)   # particle diameter
dr = sigma/20           # binning interval for the correlation function
Nsteps = 1000   #Number of steps involved in the simulation

dt = 1e-9


init_spacing = L/(np.sqrt(N)+0.2)
velocity_variance = k_b_real*T 