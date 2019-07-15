import pdb

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fmin_cg


def error_fun(u):
    return np.sum((u_data - u)**2)

def error_kappa(kappa):
    u = np.matmul(I, kappa)
    return error_fun(u)


t_sim = 1000
t = np.arange(0,t_sim)

u_data = np.zeros((t_sim,))
u_data[2] = 1
u_data[4] = 2
u_data[7] = 1

i = np.ones((t_sim,))/3
i[0] = 0 
i[300:] = 0

a = np.zeros((t_sim,))
a[75] = 1
a[250] = 1


kappa = 0.45*np.exp(-t/18)
# Original: eta = 7*(np.exp(-t/18)-np.exp(-t/45))
eta = 7*(np.exp(-(t-1)/18)-np.exp(-(t-1)/45))
eta[0] = 15
theta1 = 12*np.exp(-t/37)+2*np.exp(-t/500)

# u = I * k
I = np.nan * np.ones((t_sim, len(kappa)))
for n in range(I.shape[0]):
    if n<len(kappa):
        i_n = np.concatenate((i[n::-1], np.zeros((len(kappa)-n-1))), axis=0)
    else:
        i_n = i[n::-1][0:k]
    I[n,:] = i_n

iterations = 2
for _ in range(iterations):

    
    u_vec = np.matmul(I, kappa)
    u = np.convolve(kappa, i, mode='full') + np.convolve(eta, a, mode='full')
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
    plt.plot(t, kappa, label = 'kappa(s)')
    #plt.title('kappa(s)')
    plt.legend()
    plt.subplot(4,2,4)
    plt.plot(t, eta, label='eta(s)')
    #plt.title('eta(s)')
    plt.legend()

    # Membrane Potential
    plt.subplot(4,1,3)
    #plt.plot(u_data, label='data')
    plt.plot(u[0:t_sim], label='u(t) = conv(I, k) + conv(eta, a)')
    plt.plot(u_vec, label='u(t) = I * k')
    #plt.title('u(t)')
    plt.legend()

    # Activations
    plt.subplot(4,1,4)
    plt.plot(t, a, label='activations')
    plt.legend()
    plt.show()

    kappa = fmin_cg(error_kappa, kappa)
