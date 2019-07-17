import pdb

import matplotlib.pyplot as plt

import numpy as np

import neuron_parameters as param


class nonlinear_neuron:
    def __init__(self, lenght_filter=100, parameters='e', delta=10**-8):
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
        self.delta = delta

    def simulate(self, i):
        steps = len(i)
        filter_steps = len(self.kappa)
        t_vec = np.arange(0, steps)

        u = np.convolve(self.kappa, i, mode='full')[0:len(t_vec)]
        u_latent = np.zeros(t_vec.shape)
        a = np.zeros(t_vec.shape)
        theta = self.theta0 * np.ones(t_vec.shape)
        for t in range(len(t_vec)):
            a[t] = 1 / (1 + self.delta * np.exp(-(u[t]-theta[t])/self.delta))
            # Apply Filters
            if steps - t >= filter_steps:
                u_latent[t:t+filter_steps] += self.eta * a[t]
                theta[t:t+filter_steps] += self.theta1 * a[t]
            else:
                u_latent[t:] +=  self.eta[0:steps - t] * a[t]
                # Apply theta1
                theta[t:] += self.theta1[0:steps - t] * a[t]
            u[t] += u_latent[t]
        return u, a, theta


if __name__ == "__main__":

    # Inputs
    steps = 400
    t_vec = np.arange(0,steps)

    i = np.ones((steps,))


    # Neuron
    filter_steps = 400
    neuron = nonlinear_neuron(filter_steps, 'e', delta = 10**-4)

    u, a, theta = neuron.simulate(i)


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
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(t_vec, u, label='u(t)')
    plt.plot(t_vec, theta, label='threshold', ls='-.', c='k', alpha=0.5)
    plt.subplot(2,1,2)
    plt.plot(t_vec, a, label='a(t)')
    plt.legend()
    plt.show()