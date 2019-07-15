import numpy as np

def inhibitory(filter_lenght):
    # Domain
    s = np.arange(0, filter_lenght)
    
    # Kappa. I(t-s)
    kappa = 6*np.exp(-s/9)

    # eta. S(t-s)
    # Original: eta = 3*np.exp(-s/9) - 8*np.exp(-s/37) + 4*np.exp(-s/62)
    eta = 3*np.exp(-s/9) - 8*np.exp(-s/37) + 4*np.exp(-s/62)
    eta[0] = 12.5

    # theta1. S(t-s)
    theta1 = 12*np.exp(-s/37)+2*np.exp(-s/500)

    return kappa, eta, theta1, s


def excitatory(filter_lenght):
    # Domain
    s = np.arange(0, filter_lenght)
    
    # Kappa. I(t-s)
    kappa = 0.45*np.exp(-s/18)

    # eta. S(t-s)
    # Original: eta = 7*(np.exp(-t/18)-np.exp(-t/45))
    eta = 7*(np.exp(-(s-1)/18)-np.exp(-(s-1)/45))
    eta[0] = 12.5

    # theta1. S(t-s)
    theta1 = 12*np.exp(-s/37)+2*np.exp(-s/500)

    return kappa, eta, theta1, s