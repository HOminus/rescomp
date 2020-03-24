# -*- coding: utf-8 -*-
""" Calculating the lorenz63-attractor and other chaotic systems using 4th order runge kutta method

@author: aumeier, baur, mabrouk
"""

import numpy as np


def _roessler(x):
    '''
    returns (dx/dt, dy/dt, dz/dt) for given (x,y,z)
    '''
    a = 0.5
    b = 2.
    c = 4.
    np.array(x)
    if x.shape == (3,):
        return np.array([-x[1] - x[2], x[0] + a * x[1], b + x[2] * (x[0] - c)])
    else:
        raise Exception('check shape of x, should have 3 components')
        
def _roessler_sprott(x):
    '''
    Returns (dx/dt, dy/dt, dz/dt) for given (x,y,z).
    This version is identical to roessler(), but uses the parameters from Sprott.
    Implemented as its own function for now to make sure not to interfere.
    '''
    a = 0.2
    b = 0.2
    c = 5.7
    np.array(x)
    if x.shape == (3,):
        return np.array([-x[1] - x[2], x[0] + a * x[1], b + x[2] * (x[0] - c)])
    else:
        raise Exception('check shape of x, should have 3 components')


def _normal_lorenz(x):
    '''
    returns (dx/dt, dy/dt, dz/dt) for given (x,y,z)
    '''
    sigma = 10.
    rho = 28.
    b = 8 / 3
    np.array(x)
    if x.shape == (3,):
        return np.array([sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - b * x[2]])
    else:
        raise Exception('check shape of x, should have 3 components')


def _mod_lorenz(x):
    '''
    returns (dx/dt, dy/dt, dz/dt) for given (x,y,z)
    with dz/dt += x(t) to break symmetry
    
    >>> _mod_lorenz(np.array([1.,2.,3.]))
    array([10., 23., -5.])
    '''
    sigma = 10.
    rho = 28.
    b = 8 / 3
    # np.array(x)
    # print('cheese')
    if x.shape == (3,):
        return np.array([sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - b * x[2] + x[0]])
    else:
        raise Exception('check shape of x, should have 3 components')


def _mod_lorenz_wrong(x):
    '''
    
    returns (dx/dt, dy/dt, dz/dt) for given (x,y,z)
    with dz/dt += x(t) to break symmetry
    '''
    sigma = 10.
    rho = 28.
    b = 8 / 3
    np.array(x)
    if x.shape == (3,):
        return np.array([sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - b * x[2]] + x[0])
    else:
        raise Exception('check shape of x, should have 3 components')
        
def _chua(x):
    '''
    Returns (dx/dt, dy/dt, dz/dt) for given (x,y,z)
    '''
    alpha=9.
    beta=100./7.
    a=8./7.
    b=5./7.
    
    np.array(x)
    if x.shape == (3,):
        return np.array([alpha*(x[1]-x[0]+b*x[0]+0.5*(a-b)*(np.abs(x[0]+1)
                        -np.abs(x[0]-1))),x[0]-x[1]+x[2], -beta*x[1]])
    else:
        raise Exception('check shape of x, should have 3 components')


#def g_chua(x):
#    '''
#    function needed for Chua ODEs
#    '''
#    m0=-0.5
#    m1=-0.8
#    Bp=1.
#    return m0*x+0.5*(m1-m0)*(np.abs(x+Bp)-np.abs(x-Bp))
    
def _ueda(x):
    '''
    returns (dx/dt, dy/dt, dz/dt) for given (x,y,z)
    '''
    delta = 0.05
    beta = 0.
    alpha = 1.
    gamma=7.5
    omega=1.
    np.array(x)
    if x.shape == (3,):
        return np.array([x[1],
                -delta*x[1] - beta*x[0] - alpha*x[0]**3 + gamma*np.cos(x[2]),
                omega])
    else:
        raise Exception('check shape of x, should have 3 components')
        
def _complex_butterfly(x):
    '''
    returns (dx/dt, dy/dt, dz/dt) for given (x,y,z)
    '''
    a=0.55
    np.array(x)
    if x.shape == (3,):
        return np.array([a*(x[1]-x[0]), -x[2]*np.sign(x[0]), np.abs(x[0]) - 1])
    else:
        raise Exception('check shape of x, should have 3 components')

def _chen(x):
    '''
    returns (dx/dt, dy/dt, dz/dt) for given (x,y,z)
    '''
    a = 35.
    b = 3.
    c = 28.
    np.array(x)
    if x.shape == (3,):
        return np.array([a * (x[1] - x[0]), (c-a)*x[0] -x[0]*x[2]+c*x[1], x[0] * x[1] - b * x[2]])
    else:
        raise Exception('check shape of x, should have 3 components')
   
def _rucklidge(x):
    '''
    returns (dx/dt, dy/dt, dz/dt) for given (x,y,z)
    '''
    kappa = 2.
    lam = 6.7
    
    np.array(x)
    if x.shape == (3,):
        return np.array([-kappa*x[0] + lam*x[1] - x[1]*x[2],
                         x[0], -x[2] + x[1]**2])
    else:
        raise Exception('check shape of x, should have 3 components')
        
def _rabinovich(x):
    '''
    returns (dx/dt, dy/dt, dz/dt) for given (x,y,z)
    '''
    alpha = 1.1
    gamma = 0.87
    
    np.array(x)
    if x.shape == (3,):
        return np.array([x[1]*(x[2]-1+x[0]**2)+gamma*x[0],
                         x[0]*(3*x[2]+1-x[0]**2)+gamma*x[1],
                         -2*x[2]*(alpha + x[1]*x[0])])
    else:
        raise Exception('check shape of x, should have 3 components')
        
def _thomas(x):
    '''
    returns (dx/dt, dy/dt, dz/dt) for given (x,y,z)
    '''
    b=0.18
    np.array(x)
    if x.shape == (3,):
        return np.array([-b*x[0]+np.sin(x[1]), -b*x[1]+np.sin(x[2]), -b*x[2]+np.sin(x[0])])
    else:
        raise Exception('check shape of x, should have 3 components')


#TODO: Rewrite using numpy vectorization to make faster
def _lorenz_96(x, system_dimension=11, force=8):
    # compute state derivatives
    derivative = np.zeros(system_dimension)

    # Periodic Boundary Conditions for the 3 edge cases i=1,2,system_dimension
    derivative[0] = (x[1] - x[system_dimension - 2]) * x[system_dimension - 1] - x[0]
    derivative[1] = (x[2] - x[system_dimension - 1]) * x[0] - x[1]
    derivative[system_dimension - 1] = (x[0] - x[system_dimension - 3]) * x[system_dimension - 2] - x[system_dimension - 1]

    # then the general case
    for i in range(2, system_dimension - 1):
        derivative[i] = (x[i + 1] - x[i - 2]) * x[i - 1] - x[i]

    # add the forcing term
    derivative = derivative + force

    # return the state derivatives
    return derivative

def _runge_kutta(f, dt, y=np.array([2.2, -3.5, 4.3])):
    '''
    the function approximates differential equations of the form dy/dt = f(t,y)
    returns y(t + dt)
    '''
    # print(y)
    k1 = dt * f(y)
    k2 = dt * f(y + k1 / 2)
    k3 = dt * f(y + k2 / 2)
    k4 = dt * f(y + k3)
    return y + 1. / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_trajectory(sys_flag='mod_lorenz', dt=2e-2, time_steps=int(2e4),
                        starting_point=None, **kwargs):
    """ Simulate a trajectory in an artificial chaotic system

    Args:
        sys_flag (str): The system to be simulated. Acceptable flags are:
            mod_lorenz
            mod_lorenz_wrong
            normal_lorenz
            roessler
            lorenz_96
            ueda
            chua
            complex_butterfly
            chen
            rucklidge
            rabinovich
            thomas
            roessler_sprott
            kuramoto_sivashinsky
        dt (float): Size of time steps
        time_steps (int): Number of time steps to simulate
        starting_point (np.ndarray): Starting point of the trajectory
        **kwargs (): Further Arguments passed to the simulating function,
            usually not needed

    Returns:
        trajectory (np.ndarray) the full trajectory, ready to be used for RC

    """
    if starting_point is None: starting_point = np.array([1, 2, 3])

    if sys_flag == 'mod_lorenz':
        f = _mod_lorenz
    elif sys_flag == 'mod_lorenz_wrong':
        print('YOU ARE USING AN UNUSUAL KIND OF LORENZ EQUATION! USE WITH CARE')
        f = _mod_lorenz_wrong
    elif sys_flag == 'normal_lorenz':
        f = _normal_lorenz
    elif sys_flag == 'roessler':
        f = _roessler
    elif sys_flag == 'lorenz_96':
        # Starting point is ignored here atm
        f = lambda x: _lorenz_96(x, **kwargs)
    elif sys_flag == 'ueda':
        f = lambda x: _ueda(x)
    elif sys_flag == 'chua':
        f = lambda x: _chua(x)
    elif sys_flag == 'complex_butterfly':
        f = lambda x: _complex_butterfly(x)
    elif sys_flag == 'chen':
        f = lambda x: _chen(x)
    elif sys_flag == 'rucklidge':
        f = lambda x: _rucklidge(x)
    elif sys_flag == 'rabinovich':
        f = lambda x: _rabinovich(x)
    elif sys_flag == 'thomas':
        f = lambda x: _thomas(x)
    elif sys_flag == 'roessler_sprott':
        f = lambda x: _roessler_sprott(x)
    elif sys_flag == 'kuramoto_sivashinsky':
        # Starting point is ignored here atm
        return _kuramoto_sivashinsky(dt=dt, time_steps=time_steps - 1, **kwargs)
    else:
        raise Exception('sys_flag not recoginized')

    traj_size = ((time_steps, starting_point.shape[0]))
    traj = np.zeros(traj_size)
    # if print_switch:
    #     print('record_trajector received and used: starting_point: ', starting_point)
    y = starting_point

    for t in range(traj_size[0]):
        traj[t] = y
        y = _runge_kutta(f, dt, y=y)
    return traj

# not used currently:
# def save_trajectory(time_steps, dt): #starting_point=np.array([-13.5,10.8,-17.9])
#     np.savetxt('lorenz_attractor_'+str(time_steps)+'_a_'+str(dt)+'.dat', record_trajectory(time_steps=time_steps, dt=dt))


# TODO: Refactor to make more legible, then add a docstring, remove/add print statements etc.
def _kuramoto_sivashinsky(dimensions, system_size, dt, time_steps):
    # This function simulates the Kuramoto–Sivashinsky PDE
    # reference for the numerical integration : "fourth order time stepping for stiff pde-kassam trefethen 2005" at
    # https://people.maths.ox.ac.uk/trefethen/publication/PDF/2005_111.pdf
    # Python implementation at: https://github.com/E-Renshaw/kuramoto-sivashinsky

    # print("Start PDE simulation")

    n = dimensions  # No. of grid points in real space (and hence dimensionality of the output)
    size = system_size  # grid points for the PDE simulation

    # Define initial conditions and Fourier Transform them
    x = np.transpose(np.conj(np.arange(1, n + 1))) / n
    u = np.cos(2 * np.pi * x / size) * (1 + np.sin(2 * np.pi * x / size))
    v = np.fft.fft(u)

    h = dt  # time step
    nmax = time_steps # No. of time steps to simulate
    # Wave numbers
    k = np.transpose(
        np.conj(np.concatenate((np.arange(0, n / 2), np.array([0]), np.arange(-n / 2 + 1, 0))))) * 2 * np.pi / size

    # Just copied from the paper, it works
    L = k ** 2 - k ** 4
    E = np.exp(h * L)
    E_2 = np.exp(h * L / 2)
    M = 16
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = h * np.transpose(np.repeat([L], M, axis=0)) + np.repeat([r], n, axis=0)
    Q = h * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
    f1 = h * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, axis=1))
    f2 = h * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, axis=1))
    f3 = h * np.real(np.mean((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3, axis=1))

    uu = [np.array(u)]  # List of Real space solutions, later converted to a np.array

    g = -0.5j * k  # TODO: Meaning?

    # See paper for details
    for n in range(1, nmax + 1):
        Nv = g * np.fft.fft(np.real(np.fft.ifft(v)) ** 2)
        a = E_2 * v + Q * Nv
        Na = g * np.fft.fft(np.real(np.fft.ifft(a)) ** 2)
        b = E_2 * v + Q * Na
        Nb = g * np.fft.fft(np.real(np.fft.ifft(b)) ** 2)
        c = E_2 * a + Q * (2 * Nb - Nv)
        Nc = g * np.fft.fft(np.real(np.fft.ifft(c)) ** 2)

        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        u = np.real(np.fft.ifft(v))
        uu.append(np.array(u))

    uu = np.array(uu)
    # print("PDE simulation finished")

    return uu
