import pdb

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fmin_cg

import simulation


def error_fun(u):
    return np.sum((u_data - u)**2)

def error_param(param):
    kappa = param[0:filter_lenght]
    eta = param[filter_lenght:]
    u = np.matmul(I, kappa) + np.matmul(A, eta)
    return error_fun(u)


steps = 1000
t = np.arange(0,steps)
i = np.ones((steps,))

filter_lenght = 100

neuron_gen = simulation.spr_neuron(filter_lenght, 'e')

u_data, a, _ = neuron_gen.simulate(i)

neuron_fit = simulation.spr_neuron(filter_lenght, 'i')


# u = I * k
I = np.nan * np.ones((steps, filter_lenght))
A = np.nan * np.ones((steps, filter_lenght))
for n in range(I.shape[0]):
    if n<filter_lenght:
        i_n = np.concatenate((i[n::-1], np.zeros((filter_lenght-n-1))), axis=0)
        a_n = np.concatenate((a[n::-1], np.zeros((filter_lenght-n-1))), axis=0) 
    else:
        i_n = i[n::-1][0:filter_lenght]
        a_n = a[n::-1][0:filter_lenght]
    I[n,:] = i_n
    A[n,:] = a_n

iterations = 2
for _ in range(iterations):

    
    u_vec = np.matmul(I, neuron_fit.kappa) + np.matmul(A, neuron_fit.eta)
    u = np.convolve(neuron_fit.kappa, i, mode='full') + np.convolve(neuron_fit.eta, a, mode='full')
    error = error_fun(u_vec)

    print('error: {0}'.format(error))



    ## PLOTTING
    plt.figure(figsize=(9,9))

    # Input
    plt.subplot(4,1,1)
    plt.plot(t, i)
    plt.title('I(t)')

    # Filters
    plt.subplot(4,2,3)
    plt.plot(neuron_fit.s, neuron_fit.kappa, label = 'kappa(s)')
    #plt.title('kappa(s)')
    plt.legend()
    plt.subplot(4,2,4)
    plt.plot(neuron_fit.s, neuron_fit.eta, label='eta(s)')
    #plt.title('eta(s)')
    plt.legend()

    # Membrane Potential
    plt.subplot(4,1,3)
    plt.plot(u_data, label='data')
    plt.plot(u[0:steps], label='u(t) = conv(I, k) + conv(eta, a)')
    plt.plot(u_vec, label='u(t) = I * kappa + A * eta')
    #plt.title('u(t)')
    plt.legend()

    # Activations
    plt.subplot(4,1,4)
    plt.plot(t, a, label='activations')
    plt.legend()
    plt.show()

    param = np.concatenate((neuron_fit.kappa, neuron_fit.eta), axis=0)
    param = fmin_cg(error_param, param)
    neuron_fit.kappa = param[0:filter_lenght]
    neuron_fit.eta = param[filter_lenght:]

