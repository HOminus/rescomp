from numpy.lib.function_base import average
import rescomp
import numpy as np
import scipy

from scipy.sparse.linalg.eigen.arpack.arpack \
    import ArpackNoConvergence as _ArpackNoConvergence

from matplotlib import pyplot as plt
import math
import copy
import base64

def guan(starting_point, time_steps, dt=0.02, a = 5, b = 15, c = 3, d = 12):
    def guan_derivative(x):
        dx = np.zeros(3)
        dx[0] = a * x[0] - x[1] * x[2] - x[1] + d
        dx[1] = - b * x[1] + x[0] * x[2]
        dx[2] = - c * x[2] + x[0] * x[1]
        return dx

    traj = np.zeros((time_steps, starting_point.shape[0]))
    y = starting_point
    for t in range(time_steps):
        traj[t] = y
        y = rescomp.simulations._runge_kutta(guan_derivative, dt, y)
    return traj

def halvorsen(starting_point, time_steps, dt = 0.02, sigma=1.3):
    def halvorsen_derivative(x):
        dx = np.zeros(x.shape)
        dx[0] = -sigma * x[0] - 4 * x[1] - 4 * x[2] - x[1]**2
        dx[1] = -sigma * x[1] - 4 * x[2] - 4 * x[0] - x[2]**2
        dx[2] = -sigma * x[2] - 4 * x[0] - 4 * x[1] - x[0]**2
        return dx

    traj = np.zeros((time_steps, starting_point.shape[0]))
    y = starting_point
    for t in range(time_steps):
        traj[t] = y
        y = rescomp.simulations._runge_kutta(halvorsen_derivative, dt, y)
    return traj

def circle(time_steps, t0 = 0, dt = 0.02, omega = 1., s = 1.):
    traj = np.zeros((time_steps, 2))

    for step in range(time_steps):
        t = step * dt
        traj[step] = np.array([s * np.cos(omega * (t + t0)), s * np.sin(omega * (t + t0))])
    return traj

def spr(data, err, dim_params, lyap_params, scale, pos, rot):
    mean = np.mean(data, axis = 0)
    data = data - mean
    if type(rot) is np.ndarray and rot[0] != 0:
        rx = rot[0]
        m = np.array([[1., 0. , 0.], [0.0, cos(rx), -sin(rx)], [0.0, sin(rx), cos(rx)]])
        data[:] = data.dot(m.T)
        if type(err) is np.ndarray:
            err[:] = np.abs(err.dot(m.T))
    if type(rot) is np.ndarray and rot[1] != 0:
        ry = rot[1]
        m = np.array([[cos(ry), 0., -sin(ry)], [0., 1., 0.], [sin(ry), 0., cos(ry)]])
        data[:] = data.dot(m.T)
        if type(err) is np.ndarray:
            err[:] = np.abs(err.dot(m.T))
    if type(rot) is np.ndarray and rot[2] != 0:
        rz = rot[2]
        m = np.array([[cos(rz), -sin(rz), 0.], [sin(rz), cos(rz), 0.], [0., 0., 1.]])
        data[:] = data.dot(m.T)
        if type(err) is np.ndarray:
            err[:] = np.abs(err.dot(m.T))
    if scale != None:
        size = max([np.linalg.norm(x) for x in data])
        data = scale/size * data
        if type(err) is np.ndarray:
            err = scale/size * err
            dim_params.r_min = scale/size * dim_params.r_min
            dim_params.r_max = scale/size * dim_params.r_max
        if lyap_params != None and lyap_params.epsilon != None:
            lyap_params.epsilon = scale/size * lyap_params.epsilon
    if type(pos) is np.ndarray:
        data = data + pos
    else:
        data = data + mean
    
    return data, err, dim_params, lyap_params

#TODO: Use numpy to measure the autocorrelation
def autocorrelation(data, size):
    mean = np.mean(data, axis = 0)
    var = np.var(data, axis = 0)

    autocorr = []
    for i in range(size):
        corr_i = []
        for k in range(i, data.shape[0]):
            corr_i += [data[k - i] * data[k]]
        autocorr += [(np.mean(np.array(corr_i), axis = 0) - mean**2) / var]
    return autocorr

'''
Based on "A robust method to estimate the maximal Lyapunov exponent of a time series" Kantz, 1994
Basically the same idea as the Rosenstein algorithm.
'''
def estimate_lyapunov_kantz(data, dt, minimum_time_distance, epsilon_distance, tau_begin, tau_end, tau_points):
    minimum_index_distance = math.ceil(minimum_time_distance / dt)

    taus = np.linspace(tau_begin, tau_end, tau_points)
    tau_index_offsets = np.array([round(tau/dt) for tau in taus])
    S_tau = np.zeros(tau_points)

    tree = scipy.spatial.KDTree(data)
    data_point_counts = 0
    for index in range(data.shape[0] - tau_index_offsets[-1]):
        point = data[index,]
        neighbors = tree.query_ball_point(point, epsilon_distance)
        neighbors = list(filter(lambda i: abs(index - i) >= minimum_index_distance and i + tau_index_offsets[-1] < data.shape[0], neighbors))
        neighbors_len = len(neighbors)

        if neighbors_len != 0:
            data_point_counts += 1
            for tau_index, time_offset in enumerate(tau_index_offsets):
                distance_sum = 0
                for neighbor in neighbors:
                    distance_sum += np.linalg.norm(data[index + time_offset] - data[neighbor + time_offset])
                S_tau[tau_index] += np.log(distance_sum / neighbors_len)
    return 1. / data_point_counts * S_tau, taus

def lyapunov_kantz_and_correlation_dimension(data, dt, minimum_time_distance, epsilon_distance, tau_begin, tau_end, tau_points, r_min, r_max, r_points):
    if tau_points != 2 or r_points != 2:
        raise f"At the moment only 2 is a valid choice for tau_points and r_points. Found: tau: {tau_points} r: {r_points}"

    minimum_index_distance = math.ceil(minimum_time_distance / dt)
    rs = np.logspace(np.log10(r_min), np.log(r_max), r_points)
    
    tree = scipy.spatial.KDTree(data)
    N_r = np.array(tree.count_neighbors(tree, rs), dtype=float) / data.shape[0]
    N_r = np.vstack((rs, N_r))

    dimension = (np.log(N_r[1,1]) - np.log(N_r[1,0]))/(np.log(N_r[0,1]) - np.log(N_r[0,0]))

    taus = np.linspace(tau_begin, tau_end, tau_points)
    tau_index_offsets = np.array([round(tau/dt) for tau in taus])
    S_tau = np.zeros(tau_points)

    data_point_counts = 0
    for index in range(data.shape[0] - tau_index_offsets[-1]):
        point = data[index,]
        neighbors = tree.query_ball_point(point, epsilon_distance)
        neighbors = list(filter(lambda i: abs(index - i) >= minimum_index_distance and i + tau_index_offsets[-1] < data.shape[0], neighbors))
        neighbors_len = len(neighbors)

        if neighbors_len != 0:
            data_point_counts += 1
            for tau_index, time_offset in enumerate(tau_index_offsets):
                distance_sum = 0
                for neighbor in neighbors:
                    distance_sum += np.linalg.norm(data[index + time_offset] - data[neighbor + time_offset])
                S_tau[tau_index] += np.log(distance_sum / neighbors_len)

    if data_point_counts != 0:
        lyapunov = 1. / data_point_counts * (S_tau[-1] - S_tau[0]) / (taus[-1] - taus[0])
    else:
        lyapunov = None
    return lyapunov, dimension


def _simplify_adjacency_matrix(esn, spectral_radius: float):
    network = esn._network.todense()
    for i in range(network.shape[0]): # Zerofy lower triangle
        for j in range(i):
            network[i, j] = 0

    for i in range(network.shape[0]):
        for j in range(i, network.shape[1]):
            if network[i, j] != 0:
                val = np.random.choice([-1, 1])
                network[i, j] = val
                network[j, i] = val
    esn._network = scipy.sparse.csr_matrix(network)
    if spectral_radius != None:
        esn._n_rad = spectral_radius
        eigenvals = scipy.sparse.linalg.eigs(esn._network, k=1, v0=np.ones(esn._n_dim),
        maxiter= 1e3 * esn._n_dim)[0]
        esn._network = ((esn._n_rad / np.absolute(eigenvals).max()) * esn._network)

def _make_sparse_w_in(esn, w_in_scale, p_w_in_dense, simple: bool):
    esn._w_in_scale = w_in_scale
    esn._w_in_sparse = True
    esn._w_in_ordered = False
    esn._w_in = np.zeros((esn._n_dim, esn._x_dim))
    if simple:
        rnd = np.random.default_rng(12345678)
        for i in range(esn._n_dim):
            if p_w_in_dense == None or rnd.rand() < p_w_in_dense:
                    random_x_coord = rnd.choice(np.arange(esn._x_dim))
                    esn._w_in[i, random_x_coord] = w_in_scale
    else:
        for i in range(esn._n_dim):
            if p_w_in_dense == None or np.random.rand() < p_w_in_dense:
                random_x_coord = np.random.choice(np.arange(esn._x_dim))
                esn._w_in[i, random_x_coord] = np.random.uniform(
                    low= -esn._w_in_scale,
                    high= esn._w_in_scale)

def _project_to_3d(vec1, vec2, vec3):
    v1 = np.linalg.norm(vec1)
    v2 = np.linalg.norm(vec2)
    v3 = np.linalg.norm(vec3)
    v1v2 = np.dot(vec1, vec2)
    v1v3 = np.dot(vec1, vec3)
    v2v3 = np.dot(vec2, vec3)

    a = np.array([v1, 0, 0])

    b0 = v1v2 / v1
    b = np.array([b0, np.sqrt(v2**2 - b0**2), 0])

    c0 = v1v3 / v1
    c1 = (v2v3 - b0 * c0) / b[1]
    c = np.array([c0, c1, np.sqrt(v3**2 - c0**2 - c1**2)])

    return a, b, c, v1, v2, v3

def _create_orthogonal_matrix(n_dim, epsilon):
        deviations = np.random.rand(n_dim, n_dim)

        # Modified Gram-Schmidt to make deviations orthonormal
        for i in range(deviations.shape[0]):
            deviations[:, i] = deviations[:, i] / np.linalg.norm(deviations[:, i])
            for k in range(i+ 1, deviations.shape[1]):
                deviations[:,k] = deviations[:, k] - np.dot(deviations[:, k], deviations[:, i])
        
        # Rescale to length epsilon
        for col_index in range(deviations.shape[1]):
            deviations[col_index,:] = epsilon * deviations[col_index,:]
        
        return deviations

def _create_random_network(N: int, d: float, rho: float):
    Minit = sparse.random(N, N, density=d, data_rvs=lambda n: 2*np.random.random(n) - 1)
    for _ in range(10):
        try:
            eigs = sparse.linalg.eigs(Minit,k=1,which='LM',return_eigenvectors=False)
        except _ArpackNoConvergence:
            print("_create_random_matrix: Convergence failed")
            continue
        break
    else:
        raise Exception("_create_random_matrix: No matrix creation attempt was successful!")
    #returns the eigenvalue of largest magnitude
    M = (rho/abs(eigs[0]))*Minit
    M = sparse.csr_matrix(M)
    return M


class LyapunovComputationParameters:
    def __init__(self, minimum_time_distance, epsilon, tau_begin, tau_end):
        self.minimum_time_distance = minimum_time_distance
        self.epsilon = epsilon
        self.tau_begin = tau_begin
        self.tau_end = tau_end

class CorrelationDimensionComputationParameters:
    def __init__(self, r_min: float, r_max: float):
        self.r_min = r_min
        self.r_max = r_max

class DataConfig:
    def __init__(self, scale = None, pos = None, rot = None, dt = 0.02):
        self.scale = scale
        self.starting_point = None
        self.position = pos
        self.rotation = rot
        self.dt = dt

    def as_dict(self):
        dict = {
            "type": type(self).__name__,
            "position": tuple([self.position[i] for i in range(self.position.shape[0])]),
            "starting_point": tuple([self.starting_point[i] for i in range(self.starting_point.shape[0])]),
            "scale": self.scale,
            "dt": self.dt
        }
        if self.rotation is not None:
            dict["rotation"] =  (self.rotation[0], self.rotation[1], self.rotation[2])
        return dict
    
    @staticmethod
    def from_dict(dict):
        scale = dict["scale"]
        position = np.array([dict["position"][0], dict["position"][1], dict["position"][2]])
        rotation = None
        if "rotation" in dict:
            rotation = np.array([dict["rotation"][0], dict["rotation"][1], dict["rotation"][2]])
        dt = 0.02
        if "dt" in dict:
            dt = dict["dt"]
        return DataConfig(scale, position, rotation, dt)

class LorenzConfig(DataConfig):
    def __init__(self, scale = None, pos = None, rot = None):
        super().__init__(scale, pos, rot)
        self.sigma = 10.
        self.rho = 28.
        self.beta = 8./3.

    def set_lorenz(self, sigma, rho, beta):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def error_bounds(self):
        if self.sigma == 10 and self.rho == 28 and self.beta == 8/3:
            return np.array([5.8, 8.0, 6.9])
        raise "No error bounds"

    def dimension_parameters(self):
        if self.sigma == 10 and self.rho == 28 and self.beta == 8/3:
            return CorrelationDimensionComputationParameters(0.39161745, 4.16995761)
        raise "No dimension parameters"

    def lyapunov_parameters(self) -> LyapunovComputationParameters:
        if self.sigma == 10 and self.rho == 28 and self.beta == 8/3:
            return LyapunovComputationParameters(0.76, 0.1, 0.5, 3.5)
        raise "No lyapunov parameters"

    def generate_data(self, discard_steps: int, total_time_steps: int, starting_point_offset = None):
        start_point = copy.deepcopy(self.starting_point)
        if type(starting_point_offset) != type(None):
            start_point += starting_point_offset
        
        data = rescomp.simulations.simulate_trajectory('lorenz', self.dt, total_time_steps, start_point, 
                    sigma = self.sigma, rho = self.rho, beta = self.beta)
        data = data[discard_steps:,]

        err_bounds = self.error_bounds()
        dim_params = self.dimension_parameters()
        lyap_params = self.lyapunov_parameters()

        data, err_bounds, dim_params, lyap_params = spr(data, err_bounds, dim_params, lyap_params, self.scale, self.position, self.rotation)
        return data, err_bounds, dim_params, lyap_params

    def as_dict(self):
        return {**super().as_dict(),
            "sigma": self.sigma,
            "rho": self.rho,
            "beta": self.beta
        }

    #pylint: disable=no-self-argument
    def from_dict(dict):
        cc = DataConfig.from_dict(dict)
        lorenz = LorenzConfig(cc.scale, cc.position, cc.rotation)
        lorenz.starting_point = np.array([dict["starting_point"][0], dict["starting_point"][1], dict["starting_point"][2]])
        lorenz.sigma = dict["sigma"]
        lorenz.rho = dict["rho"]
        lorenz.beta = dict["beta"]
        return lorenz

class HalvorsenConfig(DataConfig):
    def __init__(self, scale = None, pos = None, rot = None):
        super().__init__(scale, pos, rot)    
        self.sigma = 1.3

    def set_halvorsen(self, sigma):
        self.sigma = sigma

    def error_bounds(self):
        if self.sigma == 1.3:
            return np.array([3., 3., 3.])
        raise "No error bounds"

    def lyapunov_parameters(self) -> LyapunovComputationParameters:
        if self.sigma == 1.3:
            return LyapunovComputationParameters(1.5, 0.1, 0.5, 3.5)
        raise "No lyapunov parameters"

    def dimension_parameters(self):
        if self.sigma == 1.3:
            return CorrelationDimensionComputationParameters(0.14857506, 1.4855419)
        raise "No dimension parameters bounds"

    def generate_data(self, discard_steps: int, total_time_steps: int, starting_point_offset = None):
        start_point = copy.deepcopy(self.starting_point)
        if type(starting_point_offset) != type(None):
            start_point += starting_point_offset

        data = halvorsen(start_point, total_time_steps, self.dt, self.sigma)
        data = data[discard_steps:,]

        err_bounds = self.error_bounds()
        dim_params = self.dimension_parameters()
        lyap_params = self.lyapunov_parameters()

        data, err_bounds, dim_params, lyap_params = spr(data, err_bounds, dim_params, lyap_params, self.scale, self.position, self.rotation)
        return data, err_bounds, dim_params, lyap_params

    def as_dict(self):
        return {**super().as_dict(), 
        "sigma": self.sigma
        }

    #pylint: disable=no-self-argument
    def from_dict(dict):
        cc = DataConfig.from_dict(dict)
        halvorsen = HalvorsenConfig(cc.scale, cc.position, cc.rotation)
        halvorsen.starting_point = np.array([dict["starting_point"][0], dict["starting_point"][1], dict["starting_point"][2]])
        halvorsen.sigma = dict["sigma"]
        return halvorsen

class GuanConfig(DataConfig):
    def __init__(self, scale = None, pos = None, rot = None):
        super().__init__(scale, pos, rot)
        self.a = 5.
        self.b = 15.
        self.c = 3.
        self.d = 12.
    
    def set_guan(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def error_bounds(self):
        if self.a == 5 and self.b == 15 and self.c == 3 and self.d == 12:
            return np.array([4.95, 3.72, 3.39])
        raise "No error bounds"

    def dimension_parameters(self):
        if self.a == 5. and self.b == 15. and self.c == 3. and self.d == 12.:
            return CorrelationDimensionComputationParameters(0.23293405, 1.15832899)
        else:
            raise "No dimension parameters"

    def lyapunov_parameters(self) -> LyapunovComputationParameters:
        if self.a == 5. and self.b == 15. and self.c == 3. and self.d == 12.:
            return LyapunovComputationParameters(0.92, 0.1, 0.5, 3.5)
        raise "No lyapunov parameters"

    def generate_data(self, discard_steps: int, total_time_steps: int, starting_point_offset = None):
        start_point = copy.deepcopy(self.starting_point)
        if type(starting_point_offset) != type(None):
            start_point += starting_point_offset
        
        data = guan(start_point, total_time_steps, self.dt, self.a, self.b, self.c, self.d)
        data = data[discard_steps:,]

        err_bounds = self.error_bounds()
        dim_params = self.dimension_parameters()
        lyap_params = self.lyapunov_parameters()

        data, err_bounds, dim_params, lyap_params = spr(data, err_bounds, dim_params, lyap_params, self.scale, self.position, self.rotation)
        return data, err_bounds, dim_params, lyap_params

    def as_dict(self):
        return {**super().as_dict(),
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "d": self.d
        }

    #pylint: disable=no-self-argument
    def from_dict(dict):
        cc = DataConfig.from_dict(dict)
        guan = GuanConfig(cc.scale, cc.position, cc.rotation)
        guan.starting_point = np.array([dict["starting_point"][0], dict["starting_point"][1], dict["starting_point"][2]])
        guan.a = dict["a"]
        guan.a = dict["b"]
        guan.a = dict["c"]
        guan.a = dict["d"]
        return guan

class CircleConfig(DataConfig):
    def __init__(self, scale = None, pos = None):
        super().__init__(scale, pos, None)
        self.starting_point = 0 # For the circle this is a initial time offset
        self.omega = 1.
        self.s = 1.

    def set_circle(self, omega, s):
        self.omega = omega
        self.s = s

    def error_bounds(self):
        x = 2 * abs(self.s) * 0.15
        return np.array([x, x])

    def dimension_parameters(self):
        return CorrelationDimensionComputationParameters(abs(self.s) * 0.00430602, abs(self.s) * 0.40158051)

    def lyapunov_parameters(self):
        return LyapunovComputationParameters(2 * math.pi / self.omega, 0.1, 0.5, 3.5)
    
    def generate_data(self, discard_steps: int, total_time_steps: int, starting_time_offset = None):
        start_time = copy.deepcopy(self.starting_point)
        if type(starting_time_offset) != type(None):
            start_time += starting_time_offset
        
        data = circle(total_time_steps, start_time, self.dt, self.omega, self.s)
        data = data[discard_steps:,]

        err_bounds = self.error_bounds()
        dim_params = self.dimension_parameters()
        lyap_params = self.lyapunov_parameters()

        data, err_bounds, dim_params, lyap_params = spr(data, err_bounds, dim_params, lyap_params, self.scale, self.position, self.rotation)
        return data, err_bounds, dim_params, lyap_params

    def as_dict(self):
        return {**super().as_dict(),
            "omega": self.omega,
            "s": self.s
        }

    #pylint: disable=no-self-argument
    def from_dict(dict):
        cc = DataConfig.from_dict(dict)
        circle = CircleConfig(cc.scale, cc.position)
        circle.starting_point = np.array([dict["starting_point"][0], dict["starting_point"][1], dict["starting_point"][2]])
        circle.omega = dict["omega"]
        circle.s = dict["s"]
        return circle


class AttractorData:
    def __init__(self, config):
        self.config = config
        self.train = None
        self.true_prediction = None
        self.esn_prediction = None
        self.divergence_bounds = None
        self.correlation_dimension_parameters = None
        self.lyapunov_parameters = None

    def divergence_time(self) -> int:
        assert type(self.esn_prediction) == np.ndarray
        return int(rescomp.measures.divergence_time(self.true_prediction, self.esn_prediction, self.divergence_bounds))

    def rmse(self, last_n: int = None) -> float:
        assert type(self.esn_prediction) == np.ndarray
        if last_n == None:
            return rescomp.measures.rmse(self.esn_prediction, self.true_prediction)
        else:
            return rescomp.measures.rmse(self.esn_prediction[-last_n,:], self.true_prediction[-last_n,:])

    def reset_data(self, randomize_start: bool, discard_steps: int, train_sync_steps: int, train_steps: int, pred_sync_steps: int, pred_steps: int):
        total_time_steps = discard_steps + train_sync_steps + train_steps + pred_sync_steps + pred_steps
        
        random_offset = None
        if randomize_start and type(self.config) == CircleConfig:
            random_offset = 2 * math.pi * np.random.rand()
        elif randomize_start:
            random_offset = np.random.rand((3))
        
        data, err_bounds, dim_params, lyap_params = self.config.generate_data(discard_steps, total_time_steps, random_offset)

        assert data.shape[0] == train_sync_steps + train_steps + pred_sync_steps + pred_steps
        train, pred = rescomp.utilities.train_and_predict_input_setup(data, 0, train_sync_steps, train_steps, pred_sync_steps, pred_steps)

        self.train = train
        self.true_prediction = pred
        self.esn_prediction = None

        self.divergence_bounds = err_bounds
        self.correlation_dimension_parameters = dim_params
        self.lyapunov_parameters = lyap_params

    @staticmethod
    def _correlation_lyapunov(data, correlation: bool, lyapunov: bool, 
                        correlation_dimension_params: CorrelationDimensionComputationParameters,
                        lyapunov_params: LyapunovComputationParameters, last_n: int):
        if (correlation and lyapunov):
            raise "AttractorData._correlation_lyapunov: Lyapunov with correlation measurement is currently disabled"
        elif lyapunov:
            raise "AttractorData._correlation_lyapunov: Lyapunov measurement is currently disabled"
        elif correlation:
            r_min = correlation_dimension_params.r_min
            r_max = correlation_dimension_params.r_max
            if last_n == None:
                corr = rescomp.measures.dimension(data, r_min, r_max)
            else:
                corr = rescomp.measures.dimension(data[-last_n:,], r_min, r_max)
            return corr, None
        else:
            raise "AttractorData:_correlation_lypunov: No measurement specified"

    def actual_correlation_lyapunov(self, correlation: bool, lyapunov: bool, last_n = None):
        # Only call this method after the esn did a prediction
        # since otherwise true_prediction contains also the prediction sync data
        assert type(self.esn_prediction) == np.ndarray
        return AttractorData._correlation_lyapunov(self.true_prediction, correlation, lyapunov, self.correlation_dimension_parameters, self.lyapunov_parameters,last_n)

    def prediction_correlation_lyapunov(self, correlation: bool, lyapunov: bool, last_n = None):
        assert type(self.esn_prediction) == np.ndarray 
        return AttractorData._correlation_lyapunov(self.esn_prediction, correlation, lyapunov, self.correlation_dimension_parameters, self.lyapunov_parameters, last_n)

    def plot_prediction(self):
        assert type(self.esn_prediction) == np.ndarray
        
        x_dim = self.true_prediction.shape[1]
        if x_dim == 3:
            ax = plt.axes(projection="3d")
            ax.plot(self.true_prediction[:,0], self.true_prediction[:,1], self.true_prediction[:,2], label="Data")
            ax.plot(self.esn_prediction[:,0], self.esn_prediction[:,1], self.esn_prediction[:,2], label="ESN Prediction")
        elif x_dim == 2:
            plt.plot(self.true_prediction[:, 0], self.true_prediction[:, 1], label="Data")
            plt.plot(self.esn_prediction[:, 0], self.esn_prediction[:, 1], label="ESN Prediction")
        else:
            raise ValueError(f"AttractorData.plot_prediction: Dimension {x_dim} not supported.")

class AttractorConfig:
    def __init__(self, discard_steps, train_sync_steps, train_steps, pred_sync_steps, pred_steps):
        self.discard_steps = discard_steps
        self.train_sync_steps = train_sync_steps
        self.train_steps = train_steps
        self.pred_sync_steps = pred_sync_steps
        self.pred_steps = pred_steps

        self.attractors = []

    def __getitem__(self, index):
        return self.attractors[index]

    def add(self, config, randomize_start: bool = False):
        self.attractors += [AttractorData(config)]

    def generate_new_data(self, randomize_starts: bool):
        for attractor in self.attractors:
            attractor.reset_data(randomize_starts, self.discard_steps, self.train_sync_steps, self.train_steps, self.pred_sync_steps, self.pred_steps)

class TrainStateCapture:
    def __init__(self):
        self.r_captures = []
        self.x_captures = []

    def capture(self, esnx):
        self.r_captures += [esnx.esn._r_train]
        self.x_captures += [esnx.esn._x_train]

class ESNX:
    def __init__(self, attractor_config: AttractorConfig):
        self.attractor_config = attractor_config
        self.esn = rescomp.ESN()
        self.input_strength = None #Used by MemcapResult

    def _train_without_fit(self, train_data, train_sync_steps):
        if train_sync_steps != 0:
            x_sync = train_data[:train_sync_steps]
            self.esn._x_train = train_data[train_sync_steps:]
            self.esn.synchronize(x_sync)
        else:
            self.esn._x_train = train_data

        self.esn._r_train = self.esn.synchronize(self.esn._x_train[:-1], save_r = True)
        self.esn._r_train_gen = self.esn._r_to_generalized_r(self.esn._r_train)

    def train(self, capture_train_states: bool =  False) -> TrainStateCapture:
        assert self.attractor_config != None
        assert self.esn._x_dim == self.attractor_config.attractors[0].train.shape[1]
        assert self.esn._w_out_fit_flag != None
        assert type(self.esn._w_in) == np.ndarray
        assert self.esn._act_fct_flag != None
        assert self.esn._act_fct != None

        tsc = TrainStateCapture()
        r = []
        y = []
        for attractor in self.attractor_config.attractors:
            self._train_without_fit(attractor.train, self.attractor_config.train_sync_steps)
            r.append(self.esn._r_train_gen)
            y.append(self.esn._x_train[1:])
            if capture_train_states:
                tsc.capture(self)
        
        r_combined = np.vstack(tuple(r)).T
        y_combined = np.vstack(tuple(y))
        w_out = np.linalg.solve(r_combined @ r_combined.T + self.esn._reg_param * np.identity(r_combined.shape[0]),
                r_combined @ y_combined).T
        self.esn._w_out = w_out
        return tsc

    def predict(self):
        for attractor in self.attractor_config.attractors:
            assert type(attractor.esn_prediction) != np.ndarray
            pred, actual = self.esn.predict(attractor.true_prediction, sync_steps=self.attractor_config.pred_sync_steps)
            attractor.true_prediction = actual
            attractor.esn_prediction = pred

    def dump_wout(self):
        print("Shape: ", self.esn._w_out.shape, "\n", self.esn._w_out)

    def dump_win(self):
        print("Shape: ", self.esn._w_in.shape, "\n", self.esn._w_in)


class AttractorResult:
    def __init__(self, attractor_id: int, last_n_rmse_size: int, last_n_correlation_lyapunov_size: int):
        self.attractor_id = attractor_id
        self.divergence_time = None
        self.rmse = None
        self.prediction_correlation_dimension = None
        self.actual_correlation_dimension = None
        self.prediction_lyapunov = None
        self.actual_lyapunov = None

        self.last_n_rmse_size = last_n_rmse_size
        self.last_n_rmse = None

        self.last_n_correlation_lyapunov_size = last_n_correlation_lyapunov_size
        self.last_n_prediction_correlation_dimension = None
        self.last_n_actual_correlation_dimension = None
        self.last_n_prediction_lyapunov = None
        self.last_n_actual_lyapunov = None

    def measure(self, esnx: ESNX, rmse: bool, correlation: bool, lyapunov: bool, last_n_correlation: bool, last_n_lyapunov: bool):
        attractor_data = esnx.attractor_config.attractors[self.attractor_id]
        self.divergence_time = attractor_data.divergence_time()

        if rmse:
            self.rmse = attractor_data.rmse()

        if correlation or lyapunov:
            corr, lyap = attractor_data.prediction_correlation_lyapunov(correlation, lyapunov)
            self.prediction_correlation_dimension = corr
            self.prediction_lyapunov = lyap

            corr, lyap = attractor_data.actual_correlation_lyapunov(correlation, lyapunov)
            self.actual_correlation_dimension = corr
            ar.prediction_lyapunov = lyap

        if self.last_n_rmse_size != None:
            self.last_n_rmse = attractor_data.rmse(self.last_n_rmse_size)

        if self.last_n_correlation_lyapunov_size != None:
            assert last_n_correlation or last_n_lyapunov

            corr, lyap = attractor_data.prediction_correlation_lyapunov(last_n_correlation, last_n_lyapunov, self.last_n_correlation_lyapunov_size)
            self.last_n_prediction_correlation_dimension = corr
            self.last_n_prediction_lyapunov = lyap

            corr, lyap = attractor_data.actual_correlation_lyapunov(last_n_correlation, last_n_lyapunov, self.last_n_correlation_lyapunov_size)
            self.last_n_actual_correlation_dimension = corr
            self.last_n_actual_lyapunov = lyap

    def as_dict(self):
        dict = {
            "type": "AttractorResult",
            "attractor_id": self.attractor_id,
            "divergence_time": self.divergence_time
        }

        if self.rmse != None:
            dict["rmse"] = self.rmse

        if self.prediction_correlation_dimension != None:
            dict["prediction_correlation_dimension"] = self.prediction_correlation_dimension
        
        if self.actual_correlation_dimension != None:
            assert self.prediction_correlation_dimension != None
            dict["actual_correlation_dimension"] = self.actual_correlation_dimension

        if self.prediction_lyapunov != None:
            dict["prediction_lyapunov"] = self.prediction_lyapunov

        if self.actual_lyapunov != None:
            assert self.prediction_lyapunov != None
            dict["actual_lyapunov"] = self.actual_lyapunov

        if self.last_n_rmse_size != None:
            assert self.last_n_rmse != None
            dict["last_n_rmse_size"] = self.last_n_rmse_size

        if self.last_n_rmse != None:
            assert self.last_n_rmse_size != None
            dict["last_n_rmse"] = self.last_n_rmse

        if self.last_n_correlation_lyapunov_size != None:
            assert (self.last_n_prediction_correlation_dimension != None or self.last_n_prediction_lyapunov != None)
            dict["last_n_correlation_lyapunov_size"] = self.last_n_correlation_lyapunov_size

        if self.last_n_prediction_correlation_dimension != None:
            assert self.last_n_actual_correlation_dimension != None
            dict["last_n_prediction_correlation_dimension"] = self.last_n_prediction_correlation_dimension

        if self.last_n_actual_correlation_dimension != None:
            assert self.last_n_prediction_correlation_dimension != None
            dict["last_n_actual_correlation_dimension"] = self.last_n_actual_correlation_dimension

        if self.last_n_prediction_lyapunov != None:
            assert self.last_n_actual_lyapunov != None
            dict["last_n_prediction_lyapunov"] = self.last_n_actual_lyapunov

        if self.last_n_actual_lyapunov != None:
            assert self.last_n_prediction_lyapunov != None
            dict["last_n_actual_lyapunov"] = self.last_n_actual_lyapunov

        return dict

    @staticmethod
    def from_dict(dict):
        last_n_rmse_size = None
        if "last_n_rmse_size" in dict:
            assert "last_n_rmse" in dict
            last_n_rmse_size = dict["last_n_rmse_size"]
        
        last_n_correlation_lyapunov_size = None
        if "last_n_correlation_lyapunov_size" in dict:
            assert "last_n_actual_correlation_dimension" in dict or "last_n_actual_lyapunov" in dict
            last_n_correlation_lyapunov_size = dict["last_n_correlation_lyapunov_size"]

        ar = AttractorResult(dict["attractor_id"], last_n_rmse_size, last_n_correlation_lyapunov_size)
        ar.divergence_time = dict["divergence_time"]
        
        if "rmse" in dict:
            ar.rmse = dict["rmse"]

        if "prediction_correlation_dimension" in dict:
            assert "actual_correlation_dimension" in dict
            ar.prediction_correlation_dimension = dict["prediction_correlation_dimension"]

        if "actual_correlation_dimension" in dict:
            assert ar.prediction_correlation_dimension != None
            ar.actual_correlation_dimension = dict["actual_correlation_dimension"]

        if "prediction_lyapunov" in dict:
            assert "actual_lyapunov" in dict
            ar.prediction_lyapunov = dict["prediction_lyapunov"]

        if "actual_lyapunov" in dict:
            assert ar.prediction_lyapunov != None
            ar.actual_lyapunov = dict["actual_lyapunov"]

        if "last_n_rmse" in dict:
            assert "last_n_rmse_size" in dict
            ar.last_n_rmse = dict["last_n_rmse"]
        
        if "last_n_prediction_correlation_dimension" in dict:
            ar.prediction_correlation_dimension = dict["last_n_prediction_correlation_dimension"]

        if "last_n_actual_correlation_dimension" in dict:
            ar.actual_correlation_dimension = dict["last_n_actual_correlation_dimension"]

        if "last_n_prediction_lyapunov" in dict:
            ar.last_n_prediction_lyapunov = dict["last_n_prediction_lyapunov"]

        if "last_n_actual_lyapunov" in dict:
            ar.last_n_actual_lyapunoy = dict["last_n_actual_lyapunov"]

        return ar

class MemcapResult:
    def __init__(self, max_delay: int, train_sync_steps: int, train_steps: int, pred_steps: int):
        assert max_delay >= 1
        self.train_sync_steps = train_sync_steps
        self.train_steps = train_steps
        self.pred_steps = pred_steps
        self.max_delay = max_delay
        self.total_memcap = None
        self.memcap = None

    def _memcap(self, esn: ESNX, data, delay: int):
        train_sync = data[:self.train_sync_steps + (delay - 1)]
        train = data[self.train_sync_steps + (delay - 1):self.train_sync_steps + (delay - 1) + self.train_steps]
        fit_data = data[self.train_sync_steps:self.train_sync_steps + self.train_steps]
        pred = data[self.train_sync_steps + (delay - 1) + self.train_steps:self.train_sync_steps + (delay - 1) + self.train_steps + self.pred_steps]
        pred_test = data[self.train_sync_steps + self.train_steps:self.train_sync_steps + self.train_steps + self.pred_steps]

        esn.esn.synchronize(train_sync)
        esn.esn._r_train = esn.esn.synchronize(train, save_r=True)
        esn.esn._r_train_gen = esn.esn._r_to_generalized_r(esn.esn._r_train)

        relevant_y = fit_data.T
        relevant_r = esn.esn._r_train_gen.T

        esn.esn._w_out = np.linalg.solve(relevant_r @ relevant_r.T + esn.esn._reg_param * np.identity(relevant_r.shape[0]),
                            relevant_r @ relevant_y.T).T

        r_pred = np.zeros((self.pred_steps, 1))
        for i in range(self.pred_steps):
            r_pred[i] = esn.esn._predict_step(pred[i])

        result_data = np.vstack((pred_test.T, r_pred.T))
        corr = (np.corrcoef(result_data)[0, 1])**2
        return corr

    def measure(self, esn: ESNX, w_in_sparse: float):
        assert esn.input_strength != None
        #save some of the data changed during the memcap measurement
        esn_save_x_dim = esn.esn._x_dim
        esn_save_w_in = copy.deepcopy(esn.esn._w_in)
        esn_save_w_out = copy.deepcopy(esn.esn._w_out)
        
        esn.esn._x_dim = 1
        _make_sparse_w_in(esn.esn, esn.input_strength, 1. if w_in_sparse == None else w_in_sparse, False)
        data_size = self.train_sync_steps + self.train_steps + self.pred_steps + (self.max_delay - 1)
        data = 2 * np.random.rand(data_size, 1) - 1

        self.memcap = np.zeros(self.max_delay - 1)
        for m in range(1, self.max_delay):
            self.memcap[m - 1] = self._memcap(esn, data, m)
        self.total_memcap = np.sum(self.memcap)

        esn.esn._x_dim = esn_save_x_dim
        esn.esn._w_in = esn_save_w_in
        esn.esn._w_out = esn_save_w_out


    def as_dict(self, save_stddev: bool = True):
        assert type(self.memcap) == np.ndarray
        dict = {
            "type": "MemcapResult",
            "max_delay": self.max_delay,
            "train_sync_steps": self.train_sync_steps,
            "train_steps": self.train_steps,
            "pred_steps": self.pred_steps,
            "total_memcap": self.total_memcap,
            "memcap": base64.b85encode(self.memcap.astype('float32').tobytes()).decode('ASCII')
        }

        return dict

    @staticmethod
    def from_dict(dict):
        mr = MemcapResult(dict["max_delay"], dict["train_sync_steps"], dict["train_steps"], dict["pred_steps"])
        mr.total_memcap = dict["total_memcap"]
        mr.memcap = np.frombuffer(base64.b85decode(dict["memcap"]), dtype='float32').reshape(mr.max_delay - 1)

        return mr

class InubushiResult:
    def __init__(self, attractor_id: int, discard_steps: int, sync_steps: int, forward_steps: int, measurement_count: int, epsilon: float):
        self.attractor_id: int = attractor_id
        self.discard_steps = discard_steps
        self.sync_steps = sync_steps
        self.forward_steps = forward_steps
        self.measurement_count = measurement_count
        self.epsilon = epsilon
        self.inubushi_total_memcap = None
        self.inubushi_memcap = None
        self.inubushi_mem_stddev = None

    def measure(self, esnx: ESNX):
        result = np.zeros((self.forward_steps, 0))

        for _ in range(self.measurement_count):
            attractor_config = esnx.attractor_config[self.attractor_id].config

            randomized_start_offset = None
            if type(attractor_config) == CircleConfig:
                randomized_start_offset = 2 * math.pi * np.random.rand()
            else:
                randomized_start_offset = np.random.rand((3))

            total_time_steps = self.discard_steps + self.sync_steps + self.forward_steps
            data, _, _, _ = attractor_config.generate_data(self.discard_steps, total_time_steps, randomized_start_offset)

            if esnx.esn._act_fct_flag == esnx.esn._act_fct_flag_synonyms.get_flag("tanh_simple"):
                current_result = self._measure_with_tanh_simple(esnx, data)
            elif esnx.esn._act_fct_flag == esnx.esn._act_fct_flag_synonyms.get_flag("leaky_integrator"):
                current_result = self._measure_with_leaky_integrator(esnx, data)
            else:
                raise ValueError(f"InubushiResult.measure: Activation function {esnx.esn._act_fct_flag} not supported")

            result = np.hstack((result, current_result.reshape(self.forward_steps, 1)))

        self.inubushi_memcap = np.average(result, axis = 1).T
        self.inubushi_total_memcap = np.sum(self.inubushi_memcap)
        self.inubushi_mem_stddev = np.std(result, axis = 1).T

    # TODO: Unify with _measure_with_leaky_integrator
    def _measure_with_tanh_simple(self, esnx: ESNX, data):
        # tanh' == 1 / cosh^2

        # Synchronize with signal to make sure signal is on attractor
        # set r[0] and s[0]
        # create deviation d
        # compute d[1] with r[0] and s[0]
        # compute r[1] with r[0] and s[0]
        # compute d[1] with r[1] and s[1]
        # compute r[2] with r[1] and s[1]
        # ....

        sync = data[:self.sync_steps]
        esnx.esn.synchronize(sync)
        data = data[self.sync_steps:]
        deviations = _create_orthogonal_matrix(esnx.esn._n_dim, self.epsilon)

        current_s = data[0]
        current_r = esnx.esn._last_r

        result = np.zeros((self.forward_steps, esnx.esn._n_dim))
        
        for k in range(result.shape[1]):
            result[0, k] = np.linalg.norm(deviations[:,k])

        dense_network = esnx.esn._network.todense()
        DF = np.zeros((current_r.shape[0], current_r.shape[0]))
        for t in range(1, result.shape[0]):

            dx_tanh = np.cosh(esnx.esn._w_in @ current_s + esnx.esn._network @ current_r)**2
            for i in range(DF.shape[0]):
                DF[i, :] = dense_network[i,:] / dx_tanh[i]
            deviations = DF @ deviations

            current_r = np.tanh(esnx.esn._w_in @ current_s + esnx.esn._network @ current_r)
            current_s = data[t]

            for k in range(result.shape[1]):
                result[t, k] = np.linalg.norm(deviations[:, k])

        return np.average(result, axis=1)

    def _measure_with_leaky_integrator(self, esnx: ESNX, data):
        sync = data[:self.sync_steps]
        esnx.esn.synchronize(sync)
        data = data[self.sync_steps:]
        deviations = _create_orthogonal_matrix(esnx.esn._n_dim, self.epsilon)

        current_s = data[0]
        current_r = esnx.esn._last_r

        result = np.zeros((self.forward_steps, esnx.esn._n_dim))
        
        for k in range(result.shape[1]):
            result[0, k] = np.linalg.norm(deviations[:,k])

        dense_network = esnx.esn._network.todense()
        DF = np.zeros((current_r.shape[0], current_r.shape[0]))
        for t in range(1, result.shape[0]):

            dx_tanh = np.cosh(esnx.esn._w_in @ current_s + esnx.esn._network @ current_r)**2
            for i in range(DF.shape[0]):
                DF[i, :] = dense_network[i,:] / dx_tanh[i]
            deviations = (1 - esnx.esn._alpha) * deviations + DF @ deviations

            current_r = (1 - esnx.esn._alpha) * current_r + esnx.esn._alpha * np.tanh(esnx.esn._w_in @ current_s + esnx.esn._network @ current_r)
            current_s = data[t]

            for k in range(result.shape[1]):
                result[t, k] = np.linalg.norm(deviations[:, k])

        return np.average(result, axis=1)

    def as_dict(self, save_stddev: bool = True):
        assert type(self.inubushi_memcap) == np.ndarray
        dict = {
            "type": "InubushiResult",
            "attractor_id": self.attractor_id,
            "discard_steps": self.discard_steps,
            "sync_steps": self.sync_steps,
            "forward_steps": self.forward_steps,
            "measurement_count": self.measurement_count,
            "epsilon": self.epsilon,
            "inubushi_total_memcap": self.inubushi_total_memcap,
            "inubushi_memcap": base64.b85encode(self.inubushi_memcap.astype('float32').tobytes()).decode('ASCII')
        }

        if save_stddev:
            dict["inubushi_mem_stddev"] = base64.b85encode(self.inubushi_mem_stddev.astype('float32').tobytes()).decode('ASCII')
        
        return dict

    @staticmethod
    def from_dict(dict):
        attractor_id = dict["attractor_id"]
        discard_steps = dict["discard_steps"]
        sync_steps = dict["sync_steps"]
        forward_steps = dict["forward_steps"]
        measurement_count = dict["measurement_count"]
        epsilon = dict["epsilon"]

        ir = InubushiResult(attractor_id, discard_steps, sync_steps, forward_steps, measurement_count, epsilon)

        ir.inubushi_total_memcap = dict["inubushi_total_memcap"]
        ir.inubushi_memcap = np.frombuffer(base64.b85decode(dict["inubushi_memcap"]), dtype='float32')

        if "inubushi_mem_stddev" in dict:
            ir.inubushi_mem_stddev = np.frombuffer(base64.b85decode(dict["inubushi_mem_stddev"]), dtype='float32')

        return ir

class TrainStateSpaceResult:
    def __init__(self, minimum_distance, maximum_distance):
        self.minimum_distance = minimum_distance
        self.maximum_distance = maximum_distance

    @staticmethod
    def measure(tsc: TrainStateCapture):
        assert len(tsc.r_captures) == 2

        r1 = tsc.r_captures[0]
        r2 = tsc.r_captures[1]

        min_distance = np.finfo('d').max
        max_distance = 0
        for state1 in r1:
            distance_matrix = np.linalg.norm(r2 - state1)
            current_min = np.min(distance_matrix, axis=1)
            current_max = np.max(distance_matrix, axis=1)

            min_distance = min(min_distance, current_min)
            max_distance = max(max_distance, current_max)

        return TrainStateSpaceResult(min_distance, max_distance)   

    def as_dict(self):
        return {
            "type": "TrainStateSpaceResult",
            "minimum_distance": self.minimum_distance,
            "maximum_distance": self.maximum_distance,
        }

    @staticmethod
    def from_dict(dict):
        return TrainStateSpaceResult(dict["minimum_distance"], dict["maximum_distance"])

class SvdResult:
    def __init__(self, low_rank_approx_error = None, singular_values: [float] = None):
        self.low_rank_approximation_error = low_rank_approx_error
        self.singular_values = singular_values

    @staticmethod
    def measure(esnx: ESNX):
        u, s, vh = np.linalg.svd(esnx.esn._w_out, full_matrices=False)
        singular_values = s.tolist()
        s[-1] = 0
        low_rank_approx_error = np.linalg.norm(esnx.esn._w_out - (u * s) @ vh, ord='fro')
        return SvdResult(low_rank_approx_error, singular_values)

    def as_dict(self):
        dict = {
            "type": "SvdResult",
            "low_rank_approximation_error": self.low_rank_approximation_error,
            "singular_values": self.singular_values
        }
        return dict

    @staticmethod
    def from_dict(dict):
        svd = SvdResult()
        svd.low_rank_approximation_error = dict["low_rank_approximation_error"]
        svd.singular_values = dict["singular_values"]
        return svd

class VolumeResult:
    def __init__(self):
        self.volume = None
        self.normalized_volume = None

    @staticmethod
    def measure(esnx: ESNX, volume: bool = True, normalized_volume:bool = True):
        vr = VolumeResult()

        x1 = esnx.esn._w_out[0,:]
        y1 = esnx.esn._w_out[1,:]
        z1 = esnx.esn._w_out[2,:]

        a, b, c, v1, v2, v3 = _project_to_3d(x1, y1, z1)
        if volume == True:
            vr.volume = np.abs(np.dot(a, np.cross(b, c)))
        if normalized_volume == True:
            a = a / v1
            b = b / v2
            c = c / v3
            vr.normalized_volume = np.abs(np.dot(a, np.cross(b, c)))
            
        return vr

    def as_dict(self):
        dict = {
            "type": "VolumeResult",
        }

        if self.volume != None:
            dict["volume"] = self.volume

        if self.normalized_volume != None:
            dict["normalized_volume"] = self.normalized_volume

        return dict        

    @staticmethod
    def from_dict(dict):
        volume = VolumeResult()

        if "volume" in dict:
            volume.volume = dict["volume"]

        if "normalized_volume" in dict:
            volume.normalized_volume = dict["normalized_volume"]

        return volume

class CircleResult:
    def __init__(self, attractor_id: int):
        self.attractor_id = attractor_id
        raise "CircleResult.init: Not yet implemented"

    @staticmethod
    def check_rotation_directiom():
        raise NotImplementedError("CircleResult.check_rotation_direction: Not yet implemented.")

    @staticmethod
    def relative_roundness():
        raise NotImplementedError("CircleResult.relative_roundness: Not yet implemented.")

    @staticmethod
    def measure(esnx: ESNX, attratcor_id: int):
        circle_data = esnx.attractor_config.attractors[attractor_id]
        assert type(circle.config) == CircleConfig
        assert type(circle.esn_prediction) == np.ndarray

        data_points_per_period = round((2 * math.pi) / (abs(circle.config.omega) * circle.config.dt))

        raise NotImplementedError("CircleResult.measure: Not yet implemented.")

    def as_dict(self):
        dict = {
            "type": "CircleResult"
        }
        raise "CircleResult.as_dict: Not yet implemented"

    @staticmethod
    def from_dict(dict):
        raise "CircleResult.from_dict: Not yet implemented"

class AdvancedNetworkAnalyzationResult:
    class NodesAnalysis:
        def __init__(self, attractor_id: int):
            self.attractor_id = attractor_id
            self.correlations = None
            self.averages = None
            self.stddev = None

        @staticmethod
        def measure(attractor_id: int, ana):
            nc = NodeCorrelation(attractor_id)
            nc.correlations = ana.correlation_input_state(nc.attractor_id)
            nc.averages = ana.average(nc.attractor_id)
            nc.stddev = ana.stddev(nc.attractor_id)
            return nc

        def as_dict(self):
            raise NotImplementedError("NodesAnalysis.from_dict: Not yet implemented")

        @staticmethod
        def from_dict(self):
            raise NotImplementedError("NodesAnalysis.as_dict: Not yet implemented")

    def __init__(self, nodes_degree, input_coupling, output_coupling):
        self.nodes_degree = nodes_degree
        self.nodes_input_coupling = input_coupling
        self.nodes_output_coupling = output_coupling

        self.nodes_analysis: list[NodesAnalysis] = []
        
    @staticmethod
    def measure(esnx, ana):
        assert ana.is_permutated_by_degree == True
        anar = AdvancedNetworkAnalyzationResult(ana.degree_vector, ana.w_in, ana.w_out)

        for i in range(len(ana.train_state_capture.r_captures)):
            anar.nodes_analysis += [AdvancedNetworkAnalyzationResult.NodesAnalysis.measure(i, ana)]
        
        return anar

    def as_dict(self):
        dict = {
            "type": "AdvancedNetworkAnalyzationResult",
            "nodes_degree": base64.b85encode(self.nodes_degree.astype('float32').tobytes()).decode('ASCII'),
            "nodes_input_coupling": base64.b85encode(self.nodes_input_coupling.astype('float32').tobytes()).decode('ASCII'),
            "nodes_output_coupling": base64.b85encode(self.nodes_output_coupling.astype('float32').tobytes()).decode('ASCII')
        }
        dict["nodes_analysis"] = [analysis.to_dict() for analysis in self.nodes_analysis]

    @staticmethod
    def from_dict(dict):
        nodes_degree = np.frombuffer(base64.b85decode(dict["nodes_degree"]), dtype='float32')
        node_count = nodes_degree.shape[0]

        input_coupling = np.frombuffer(base64.b85decode(dict["nodes_degree"]), dtype='float32')
        assert input_coupling.shape[0] % node_count == 0
        iinput_coupling = input_coupling.reshape((node_count, iinput_coupling.shape[0] / nodes_count))

        output_coupling = np.frombuffer(base64.b85decode(dict["nodes_output_coupling"]), dtype='float32')
        assert output_coupling.shape[0] % node_count == 0
        output_coupling = output_coupling.reshape((node_count, output_coupling.shape[0] / nodes_count))

        anar = AdvancedNetworkAnalyzationResult(nodes_degree, input_coupling, output_coupling)

        for node_analysis in dict["nodes_analysis"]:
            anar.nodes_analysis += [AdvancedNetworkAnalyzationResult.NodesAnalysis.from_dict(node_analysis)]

        return anar

class FlouqetAnalysisResult:
    def __init__(self, attractor_id: int, eigenvalues: np.ndarray):
        self.attractor_id = attractor_id
        self.eigenvalues = eigenvalues

    @staticmethod
    def _jacobian_leaky_integrator(esnx: ESNX, r_state) -> np.ndarray:
        raise NotImplementedError("FloquetAnalysisResult._jacobian_leaky_integrator: Not yet implemented.")

    def _jacobian_tanh_simple(esnx: ESNX, r_state) -> np.ndarray:
        raise NotImplementedError("FlouqetAnalysisResult._jacobian_tanh_simple: Not yet implemented.")

    # TODO: Do not hardcode period
    @staticmethod
    def measure(esnx: ESNX, attractor_id: int, train_state_capture: TrainStateCapture):
        circle_data = esnx.attractor_config.attractors[attractor_id]

        assert type(circle.config) == CircleConfig
        assert type(circle.esn_prediction) == np.ndarray

        dt = circle.config.dt
        data_points_per_period = round((2 * math.pi) / (abs(circle.config.omega) * circle.config.dt))

        captured_training_r = train_state_capture.r_captures[0:data_points_per_period]

        tanh_simple_flag = esnx.esn._act_fct_flag_synonyms.get_flag("tanh_simple")
        leaky_integrator_flag = esnx.esn._act_fct_flag_synonyms.get_flag("leaky_integrator")

        q = np.identity(2 * esnx.esn._n_dim)
        for r_state in captured_training_r:
            if esnx.esn._act_fct_flag == tanh_simple_flag:
                J = FlouqetAnalysisResult._jacobian_tanh_simple(esnx, r_state)
            elif esnx.esn._act_fct_flag == leaky_integrator_flag:
                J = FlouqetAnalysisResult._jacobian_leaky_integrator(esnx, r_state)
            else:
                raise ValueError("FlouqetAnalysisResult.measure: Only 'tanh_simple' and 'leaky_integrator' are supported activation functions.")

            q = q + (J @ q) * dt

        eigenvalues = scipy.sparse.linalg.eigs(q, k=10, which='LM', return_eigenvectors=False)
        return FlouqetAnalysisResult(attractor_id, eigenvalues)

    def as_dict(self):
        return {
            "type": "FlouqetAnalysisResult",
            "attractor_id": self.attractor_id,
            "eigenvalues": base64.b85encode(self.eigenvalues.astype('cfloat').tobytes()).decode('ASCII')
        }

    @staticmethod
    def from_dict(dict):
        fqr = FlouqetAnalysisResult(dict["attractor_id"])
        fqr.eigenvalues = np.frombuffer(base64.b85decode(dict["eigenvalues"]), dtype='cfloat')
        return fqr


class Run:
    def __init__(self, size: int, spectral_radius: float, average_degree: int, regression: float, input_strength: float, readout: str, topology: str,
                rewire_probability: float = None, activation_function: str = None, leaky_alpha: float = None, w_in_sparse: float = None,
                simplified_network: bool = False, andrews_matrix_creation: bool = False):
        self.size = size
        self.spectral_radius = spectral_radius
        self.average_degree = average_degree
        self.regression = regression
        self.input_strength = input_strength
        self.readout = readout

        self.topology = topology
        self.rewire_probability = rewire_probability
        if self.topology == "small_world":
            assert self.rewire_probability >= 0. and self.rewire_probability <= 1.

        self.activation_function = activation_function
        self.leaky_alpha = leaky_alpha
        if self.activation_function == "leaky_integrator":
            assert self.leaky_alpha >= 0.0 and self.leaky_alpha <= 1.0

        self.w_in_sparse = w_in_sparse

        self.andrews_matrix_creation = andrews_matrix_creation
        self.simplified_network = simplified_network

        self.measurements = []

    def get_esnx(self, attractor_config: AttractorConfig) -> ESNX:
        esnx = ESNX(attractor_config)
        esnx.input_strength = self.input_strength

        if self.topology == "random" and self.andrews_matrix_creation:
            density = run.average_degree / run.size
            esnx.esn._network = _create_random_network(run.size, density, run.spectral_radius)
        else:
            esnx.esn.create_network(self.size, self.spectral_radius, self.average_degree, self.topology, rewire_probability=self.rewire_probability)
        
        if self.simplified_network:
            _simplify_adjacency_matrix(esnx.esn, self.spectral_radius)

        if self.activation_function == None:
            esnx.esn._act_fct_flag = "tanh_simple"
        else:
            esnx.esn._act_fct_flag = self.activation_function
        esnx.esn._set_activation_function(esnx.esn._act_fct_flag)
        esnx.esn._alpha = self.leaky_alpha

        esnx.esn._reg_param = self.regression
        esnx.esn._w_out_fit_flag = esnx.esn._w_out_fit_flag_synonyms.get_flag(self.readout)
        
        if attractor_config != None:
            esnx.esn._x_dim = attractor_config.attractors[0].train.shape[1]
            _make_sparse_w_in(esnx.esn, self.input_strength, self.w_in_sparse, self.simplified_network)

        return esnx

    def add_esnx_measurements(self, *measurements):
        self.measurements += [measurements]

    def as_dict(self):
        dict = {
            "size": self.size,
            "spectral_radius": self.spectral_radius,
            "average_degree": self.average_degree,
            "regression": self.regression,
            "input_strength": self.input_strength,
            "readout": self.readout,
            "topology": self.topology,
        }

        if self.rewire_probability != None:
            dict["rewire_probability"] = self.rewire_probability

        if self.activation_function != None:
            dict["activation_function"] = self.activation_function

        if self.leaky_alpha != None:
            assert self.activation_function == "leaky_integrator"
            dict["leaky_alpha"] = self.leaky_alpha

        if self.w_in_sparse != None:
            dict["w_in_sparse"] = self.w_in_sparse

        if self.simplified_network:
            dict["simplified_network"] = self.simplified_network

        if self.andrews_matrix_creation:
            dict["andrews_matrix_creation"] = self.andrews_matrix_creation

        if len(self.measurements) > 0:
            dict["measurements"] = [[x.as_dict() for x in m] for m in self.measurements]

        return dict

    @staticmethod
    def from_dict(dict):
        size = dict["size"]
        spectral_radius = dict["spectral_radius"]
        average_degree = dict["average_degree"]
        regression = dict["regression"]
        input_strength = dict["input_strength"]
        readout = dict["readout"]
        topology = dict["topology"]

        run = Run(size, spectral_radius, average_degree, regression, input_strength, readout, topology)

        if "rewire_probability" in dict:
            run.rewire_probability = dict["rewire_probability"]
        
        if "activation_function" in dict:
            run.activation_function = dict["activation_function"]

        if "leaky_alpha" in dict:
            run.leaky_alpha = dict["leaky_alpha"]

        if "w_in_sparse" in dict:
            run.w_in_sparse = dict["w_in_sparse"]

        run.simplified_network = ("simplified_network" in dict)
        run.andrews_matrix_creation = ("andrews_matrix_creation" in dict)

        if "measurements" in dict:
            for m in dict["measurements"]:
                measurements = []
                for x in m:
                    assert "type" in x
                    if x["type"] == "AttractorResult":
                        measurements += [AttractorResult.from_dict(x)]
                    elif x["type"] == "MemcapResult":
                        measurements += [MemcapResult.from_dict(x)]
                    elif x["type"] == "InubushiResult":
                        measurements += [InubushiResult.from_dict(x)]
                    elif x["type"] == "SvdResult":
                        measurements += [SvdResult.from_dict(x)]
                    elif x["type"] == "VolumeResult":
                        measurements += [VolumeResult.from_dict(x)]
                    elif x["type"] == "CircleResult":
                        measurements += [CircleResult.from_dict(x)]
                    elif x["type"] == "AdvancedNetworkAnalyzationResult":
                        measurements += [AdvancedNetworkAnalyzationResult.from_dict(x)]
                    elif x["type"] == "TrainStateSpaceResult":
                        measurements += [TrainStateSpaceResult.from_dict(x)]
                    else:
                        raise f"Failed to deserialize type: {m['type']}"
                run.add_esnx_measurements(*tuple(measurements))

        return run

class SimulationResult:
    def __init__(self, config: AttractorConfig):
        self.attractor_config = config
        self.comments = []
        self.runs = []

    def add_comment(self, comment: str):
        self.comments += [comment]

    def add_run(self, run: Run):
        self.runs += [run]

    def as_dict(self):
        result = {}
        if self.attractor_config != None:
            attractor_dict = {}
            attractor_dict["discard_steps"] = self.attractor_config.discard_steps
            attractor_dict["train_sync_steps"] = self.attractor_config.train_sync_steps
            attractor_dict["train_steps"] = self.attractor_config.train_steps
            attractor_dict["pred_sync_steps"] = self.attractor_config.pred_sync_steps
            attractor_dict["pred_steps"] = self.attractor_config.pred_steps
            attractor_dict["attractors"] = [attractor.config.as_dict() for attractor in self.attractor_config.attractors]

            result = {
                "attractor_config": attractor_dict
            }

        if self.comments != []:
            result["comments"] = self.comments

        result["runs"] = [r.as_dict() for r in self.runs]
        return result

    @staticmethod
    def from_dict(dict):
        sr = SimulationResult(None)
        if "attractor_config" in dict:
            attractor_dict = dict["attractor_config"]
            discard_steps = attractor_dict["discard_steps"]
            train_sync_steps = attractor_dict["train_sync_steps"]
            train_steps = attractor_dict["train_steps"]
            pred_sync_steps = attractor_dict["pred_sync_steps"]
            pred_steps = attractor_dict["pred_steps"]

            sr.attractor_config = AttractorConfig(discard_steps, train_sync_steps, train_steps, pred_sync_steps, pred_steps)
            
            attractor_data = []
            for attractor_specification in attractor_dict["attractors"]:
                assert "type" in attractor_specification
                if attractor_specification["type"] == "LorenzConfig":
                    lorenz = LorenzConfig.from_dict(attractor_specification)
                    attractor_data += [AttractorData(lorenz)]
                elif attractor_specification["type"] == "HalvorsenConfig":
                    halvorsen = HalvorsenConfig(attractor_specification)
                    attractor_data += [AttractorData(halvorsen)]
                elif attractor_specification["type"] == "GuanConfig":
                    guan = GuanConfig.from_dict(attractor_specification)
                    attractor_data += [AttractorData(guan)]
                elif attractor_specification["type"] == "CircleConfig":
                    circle = CircleConfig.from_dict(attractor_specification)
                    attractor_data += [AttractorData(circle)]
                else:
                    raise ValueError(f"Failed to find valid attractor configuration")
            sr.attractor_config.attractors = attractor_data

        if "comments" in dict:
            sr.comments = [c for c in dict["comments"]]

        for run in dict["runs"]:
            sr.add_run(Run.from_dict(run))

        return sr

    @staticmethod
    def merge(*sr):
        print("Warning: SimulationResult.merge fails to catch common mismatching attractor configs or same runs.")
        
        # TODO: Assert the attractor config is the same
        merge = SimulationResult(sr[0].attractor_config)
        
        for simulation_result in sr:
            for run in simulation_result.runs:
                merge.add_run(run)
        return merge

class AdvancedNetworkAnalyzation:
    def __init__(self, esnx: ESNX, train_state_capture: TrainStateCapture):
        self.n_dim = esnx.esn._n_dim
        self.x_dim = esnx.esn._x_dim
        self.train_steps = train_states.x_captures[0].shape[0]
        self.degree_vector = self._get_in_degrees(esnx.esn)
        self.w_in = copy.deepcopy(esn._w_in)
        self.w_out = np.hstack((esn._w_out.T[:self.n_dim,], esn._w_out.T[self.n_dim:,]))
        self.train_state_capture = train_state_capture
        
        self.is_permutated_by_degree = False

    def _get_in_degrees(self, esn):
        self.degree_vector = np.zeros((esn._n_dim, 1))
        for row in range(esn._n_dim):
            self.degree_vector[row] = esn._network.indptr[row + 1] - esn._network.indptr[row]
        return self.degree_vector

    def permutate_by_degree(self):
        permutation_degree_vector = []
        for row in range(self.n_dim):
            permutation_degree_vector += [(self.degree_vector[row], row)]
        permutation_degree_vector.sort(key=lambda e: e[0], reverse=True)

        new_degree_vector = np.zeros(self.degree_vector.shape)
        new_w_in = np.zeros(self.w_in.shape)
        new_w_out = np.zeros(self.w_out.shape)
        new_train_states = [np.zeros(x.shape) for x in self.train_state_capture.r_captures]

        for row in range(self.n_dim):
            permutation_index = permutation_degree_vector[row][1]
            new_degree_vector[row] = self.degree_vector[permutation_index]
            new_w_in[row,:] = self.w_in[permutation_index,:]
            new_w_out[:,row] = self.w_out[:,permutation_index]
            for train_state_index in range(len(new_train_states)):
                new_train_states[train_state_index][:, row] = self.train_states[train_state_index][:, permutation_index]

        self.degree_vector = new_degree_vector
        self.w_in = new_w_in
        self.w_out = new_w_out
        self.train_state_capture.r_states = tuple(new_train_states)

        self.is_permutated_by_degree = True

    def correlation_input_state(self, nth: int):
        correlation = np.zeros((self.n_dim, 1))
        for row in range(correlation.shape[0]):
            if self.w_in[row, 0] != 0 and self.w_in[row, 1] == 0 and self.w_in[row, 2] == 0:
                correlation[row] = np.corrcoef(self.train_state_capture.r_captures[nth][:, row], self.train_state_capture.x_captures[nth][:, 0])[0,1]
            elif self.w_in[row, 1] != 0 and self.w_in[row, 0] == 0 and self.w_in[row, 2] == 0:
                correlation[row] = np.corrcoef(self.train_state_capture.r_captures[nth][:, row], self.train_state_capture.x_captures[nth][:, 1])[0,1]
            elif self.w_in[row, 2] != 0 and self.w_in[row, 0] == 0 and self.w_in[row, 1] == 0:
                correlation[row] = np.corrcoef(self.train_state_capture.r_captures[nth][:, row], self.train_state_capture.x_captures[nth][:, 2])[0,1]
            else:
                raise "Failed to compute corrleation"
        return correlation

    def average(self, nth: int):
        return np.average(self.train_state_capture.r_captures[nth], axis = 0)

    def stddev(self, nth: int):
        print("Capture state shape: ", self.train_state_capture[nth].shape) # TODO: remove!
        return np.average(self.train_state_capture.r_captures[nth], axis = 0)


    def as_dict(self):
        result = {
            "version": "2.0.0",
            "n_dim": self.ndim,
            "x_dim": self.x_dim,
            "train_steps": self.train_steps,
            "degree_vector": base64.b85encode(self.degree_vector.tobytes()),
            "w_out": base64.b85encode(self.w_out.tobytes()),
        }
        for i, state in enumerate(self.train_states):
            result[f"train_state_{i}"] = base64.b85decode(state.tobytes())
        return result

    @staticmethod
    def from_dict(dict):
        ana = AdvancedNetworkAnalyzation()
        ana.n_dim = dict["n_dim"]
        ana.x_dim = dict["x_dim"]
        ana.train_steps = dict["train_steps"]
        ana.degree_vector = np.frombuffer(base64.b85decode(dict["degree_vector"])).reshape((ana.n_dim, 1))
        ana.w_out = np.frombuffer(base64.b85decode(dict["w_out"])).reshape((ana.x_dim, ana.n_dim))

        train_states = []
        for i in range(100):
            if f"train_state_{i}" in dict:
                train_states.append(np.frombuffer(base64.b85decode(dict[f"train_state_{i}"])).reshape((ana.train_steps, ana.n_dim)))
            else:
                break
        ana.train_states = tuple(train_states)
        return ana

    def draw_nth(self, nth):
        fig = plt.figure()
        ax0 = fig.add_subplot(10, 25, (4, 20))
        ax1 = fig.add_subplot(10, 25, (26, 226))
        ax2 = fig.add_subplot(10, 25, (27, 247), sharex=ax0, sharey=ax1)
        ax3 = fig.add_subplot(10, 25, (48, 248), sharey=ax1)
        ax4 = fig.add_subplot(10, 25, (49, 250), sharey=ax1)

        corr = self.correlation_input_state(nth)

        img0 = ax0.imshow(self.train_state_capture.x_captures[nth].T, interpolation='nearest')
        ax1.imshow(self.w_in, interpolation='nearest', vmin=-1., vmax=1.)
        img = ax2.imshow(self.train_state_capture.r_capture[nth].T, interpolation='nearest', vmin=-1., vmax=1.)
        ax3.imshow(corr, interpolation='nearest', vmin=-1., vmax=1.)

        ax4.imshow(self.w_out, interpolation='nearest', vmin=-1., vmax=1.)

        ax0.set_title('Input')
        ax0.set_yticks([0, 1, 2])
        ax0.set_yticklabels(['X', 'Y', 'Z'])

        ax1.set_title('W_in')
        ax1.set_xticks([0, 1, 2])
        ax1.set_xticklabels(['X', 'Y', 'Z'])

        ax2.set_title('Node values')
        ax2.set_yticks(range(0, self.n_dim, 3))
        ax2.set_yticklabels([str(self.degree_vector[i]) for i in range(0, self.n_dim, 3)])
        ax2.set_ylabel('In degree')

        ax4.set_title('W_out')
        ax4.set_xticks([0, 1, 2, 3, 4, 5])
        ax4.set_xticklabels(['X', 'Y', 'Z', 'X²', 'Y²', 'Z²'])

        fig.colorbar(img0, location='right', ax=ax0)
        fig.colorbar(img, ax=ax4, location='right')

        plt.show()