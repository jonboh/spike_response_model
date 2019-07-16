import pdb

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fmin_cg
from scipy.optimize import minimize

import simulation

def numerical_gradient(param):
    e = error_u_param(param)
    eps = 10**-8
    J_approx = np.nan * np.ones((2*filter_lenght,))
    for i in range(len(J_approx)):
        diff = np.zeros(J_approx.shape)
        diff[i] += eps
        J_approx[i] = (error_u_param(param+diff) - e)/eps
    return J_approx
          
def u_compute(kappa, eta):
    return np.matmul(I, kappa) + np.matmul(A, eta)

def error_u(u):
    return np.sum((u_data - u)**2)

def error_u_param(param):
    kappa = param[0:filter_lenght]
    eta = param[filter_lenght:]
    u = u_compute(kappa, eta)
    return error_u(u)


steps = 300
t = np.arange(0,steps)
intensity = np.ones((steps,)) + np.random.randn(steps)/10
#intensity[0:1] = 0

filter_lenght = 15

neuron_gen = simulation.spr_neuron(filter_lenght, 'e')

u_data, a_data, _ = neuron_gen.simulate(intensity)

neuron_fit = simulation.spr_neuron(filter_lenght, 'i')


# u = I * k
I = np.nan * np.ones((steps, filter_lenght))
A = np.nan * np.ones((steps, filter_lenght))
for n in range(I.shape[0]):
    if n<filter_lenght:
        i_n = np.concatenate((intensity[n::-1], np.zeros((filter_lenght-n-1))), axis=0)
        a_n = np.concatenate((a_data[n::-1], np.zeros((filter_lenght-n-1))), axis=0) 
    else:
        i_n = intensity[n::-1][0:filter_lenght]
        a_n = a_data[n::-1][0:filter_lenght]
    I[n,:] = i_n
    A[n,:] = a_n

# Calculate Hessian
H = np.nan * np.ones((2*filter_lenght, 2*filter_lenght))
for i in range(0,filter_lenght): # kappa
    for j in range(0, filter_lenght): # kappa kappa
        H[i, j] = 2 * np.sum(I[:,i]*I[:,j])
    for j in range(filter_lenght, 2*filter_lenght): # kappa eta
        H[i, j] = 2 * np.sum(I[:,i]*A[:,j-filter_lenght])
for i in range(filter_lenght, 2*filter_lenght): # eta
    for j in range(0, filter_lenght): # eta kappa
        H[i,j] = 2 * np.sum(A[:,i-filter_lenght]*I[:,j])
    for j in range(filter_lenght, 2*filter_lenght): # eta eta
        H[i,j] = 2 * np.sum(A[:,i-filter_lenght]*A[:,j-filter_lenght])

def hessian(param):
    return H

def jacobian(param):
    u = u_compute(param[0:filter_lenght],param[filter_lenght:2*filter_lenght])
    J = np.nan * np.ones((2*filter_lenght,))
    for i in range(0, filter_lenght):
        J[i] = -2 * np.sum((u_data - u)*I[:,i])
    for i in range(filter_lenght, 2*filter_lenght):
        J[i] = -2 * np.sum((u_data - u)*A[:,i-filter_lenght])
    #pdb.set_trace()
    return J


iterations = 10
for iterat in range(iterations):

    
    u, a, _ = neuron_fit.simulate(intensity)
    error_vmem = error_u(u)
    print('Iteration {0:4} -> Error_u: {1:.2f}'.format(iterat,error_vmem/steps))



    ## PLOTTING
    plt.figure(figsize=(9,9))

    # Input
    plt.subplot(4,1,1)
    plt.plot(t, intensity)
    plt.title('I(t)')

    # Filters
    plt.subplot(4,2,3)
    plt.plot(neuron_fit.s, neuron_fit.kappa, label = 'kappa(s) (fit)')
    plt.plot(neuron_gen.s, neuron_gen.kappa, label = 'kappa(s) (gen)')
    #plt.title('kappa(s)')
    plt.legend()
    plt.subplot(4,2,4)
    plt.plot(neuron_fit.s, neuron_fit.eta, label='eta(s) (fit)')
    plt.plot(neuron_gen.s, neuron_gen.eta, label='eta(s) (gen)')
    #plt.title('eta(s)')
    plt.legend()

    # Membrane Potential
    plt.subplot(4,1,3)
    plt.plot(u_data, label='u_data')
    plt.plot(u[0:steps], label='u(t) simulation')
    #plt.plot(u_vec, label='u(t) = I * kappa + A * eta')
    #plt.title('u(t)')
    plt.legend()

    # Activations
    plt.subplot(4,1,4)
    plt.plot(t, a_data, label='a_data')
    plt.plot(t, a, label='a(t) simulation')
    plt.legend()
    plt.show()

    param = np.concatenate((neuron_fit.kappa, neuron_fit.eta), axis=0)
    result = minimize(error_u_param, param, method ='Newton-CG', options={'maxiter':1}, jac=jacobian)#, hess=hessian)
    #pdb.set_trace()
    neuron_fit.kappa = result.x[0:filter_lenght]
    neuron_fit.eta = result.x[filter_lenght:]

