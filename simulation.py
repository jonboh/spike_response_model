import pdb

import numpy as np
import matplotlib.pyplot as plt

import neuron_parameters as param

# Inputs
steps = 100
t_vec = np.arange(0,steps)

i = np.ones((steps,))


# Parameters
filter_steps = 100
kappa, eta, theta1, s_vec = param.inhibitory(filter_steps)

theta0 = 2.75 * np.ones(t_vec.shape)
theta = theta0


# 
u = np.convolve(kappa, i, mode='full')[0:len(t_vec)]
u_latent = np.zeros(t_vec.shape)
for t in range(len(t_vec)):
    if u[t]>=theta[t]: # FIRE
        # Apply eta
        if steps - t >= filter_steps:
            u_latent[t:t+filter_steps] += eta
            theta[t:t+filter_steps] += theta1
        else:
            u_latent[t:] +=  eta[0:steps - t]
            # Apply theta1
            theta[t:] += theta1[0:steps - t]
    u[t] += u_latent[t]

# Filters Plot
plt.subplot(3,1,1)
plt.plot(s_vec, kappa, label = 'kappa(s)')
#plt.title('kappa(s)')
plt.legend()
plt.subplot(3,1,2)
plt.plot(s_vec, eta, label='eta(s)')
#plt.title('eta(s)')
plt.subplot(3,1,3)
plt.plot(s_vec, theta1, label='theta1(s)')
plt.legend()

plt.figure()
plt.plot(t_vec, u, label='u(t)')
plt.plot(t_vec, theta, label='threshold', ls='-.', c='k', alpha=0.5)
plt.legend()
plt.show()