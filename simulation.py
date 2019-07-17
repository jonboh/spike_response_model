import pdb

import numpy as np
import matplotlib.pyplot as plt

import neuron_parameters as param

class spr_neuron:
    def __init__(self, lenght_filter=100, parameters='e'):
        if parameters.lower() == 'e':
            kappa, eta, theta0, theta1, s = param.excitatory(lenght_filter)
        elif parameters.lower() == 'i':
            kappa, eta, theta0, theta1, s = param.inhibitory(lenght_filter)
        else:
            raise ValueError('Custom Parameters not implemented in initialization')
        self.kappa = kappa
        self.eta = eta
        self.theta0 = theta0
        self.theta1 = theta1
        self.s = s

    def simulate(self, i):
        steps = len(i)
        filter_steps = len(self.kappa)
        t_vec = np.arange(0, steps)

        u = np.convolve(self.kappa, i, mode='full')[0:len(t_vec)]
        u_latent = np.zeros(t_vec.shape)
        a = np.zeros(t_vec.shape)
        theta = self.theta0 * np.ones(t_vec.shape)
        for t in range(len(t_vec)):
            if u[t]>=theta[t]: # FIRE
                a[t] = 1
                # Apply eta
                if steps - t >= filter_steps:
                    u_latent[t:t+filter_steps] += self.eta
                    theta[t:t+filter_steps] += self.theta1
                else:
                    u_latent[t:] +=  self.eta[0:steps - t]
                    # Apply theta1
                    theta[t:] += self.theta1[0:steps - t]
            u[t] += u_latent[t]
        return u, a, theta


if __name__ == "__main__":

    # Inputs
    steps = 400
    t_vec = np.arange(0,steps)

    intensity = np.ones((steps,)) + np.random.randn(steps)/10
    intensity[0:50] = 0


    # Neuron
    filter_steps = 400
    neuron = spr_neuron(filter_steps, 'e')

    u, a, theta = neuron.simulate(intensity)


    ## PLOTTING
    # Filters Plot
    plt.subplot(3,1,1)
    plt.plot(neuron.s, neuron.kappa, label = 'kappa(s)')
    plt.legend()
    plt.subplot(3,1,2)
    plt.plot(neuron.s, neuron.eta, label='eta(s)')
    plt.subplot(3,1,3)
    plt.plot(neuron.s, neuron.theta1, label='theta1(s)')
    plt.legend()

    # Simulation Plot
    # Input
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t_vec, intensity)
    plt.title('I(t)')

    plt.subplot(3,1,2)
    plt.plot(t_vec, u, label='u(t)')
    plt.plot(t_vec, theta, label='threshold', ls='-.', c='k', alpha=0.5)
    plt.title('u(t)')
    plt.legend()
    
    plt.subplot(3,1,3)
    plt.plot(t_vec, a, label='a(t)')
    plt.title('a(t)')
    plt.legend()

    plt.show()