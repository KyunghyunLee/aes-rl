import numpy as np
from collections import deque
import copy


class Population:
    def __init__(self, population_size, individual_dim, mean, var):
        assert population_size > 0
        assert mean.size == individual_dim
        if var is not None:
            assert var.size == individual_dim

        self.population_size = population_size
        self.individual_dim = individual_dim
        self.mean = mean
        self.var = var
        self.mean_result = -10000

    def update_mean_result(self, result):
        self.mean_result = result

    def ask(self, n):
        raise NotImplementedError

    def tell(self, individuals, results):
        raise NotImplementedError

    def get_mean(self):
        return self.mean, self.var


class RLPopulation(Population):
    def __init__(self, individual_dim, mean, args):
        var = np.zeros_like(mean)
        super(RLPopulation, self).__init__(1, individual_dim, mean, var)

    def ask(self, n):
        assert n == 1
        return self.mean

    def tell(self, individuals: np.ndarray, results):
        assert individuals.size == self.individual_dim
        self.mean = individuals


class CemrlPopulation(Population):
    def __init__(self, population_size, individual_dim, mean, args):
        super(CemrlPopulation, self).__init__(population_size, individual_dim, mean, None)
        self.sigma_init = args.sigma_init
        self.damp = args.cemrl_damp_init
        self.damp_limit = args.cemrl_damp_limit
        self.damp_tau = args.cemrl_damp_tau
        self.tau = args.cemrl_tau
        self.antithetic = not args.cemrl_no_antithetic

        self.var = self.sigma_init * np.ones(individual_dim)

        if self.antithetic:
            assert self.population_size % 2 == 0

        self.parents = population_size // 2
        self.weights = np.array([np.log((self.parents + 1) / i)
                                 for i in range(1, self.parents + 1)])
        self.weights /= self.weights.sum()

    def ask(self, n):
        if self.antithetic and not n % 2:
            epsilon_half = np.random.randn(n // 2, self.individual_dim)
            epsilon = np.concatenate([epsilon_half, - epsilon_half])
        else:
            epsilon = np.random.randn(n, self.individual_dim)
        individuals = self.mean + epsilon * np.sqrt(self.var)
        return individuals

    def tell(self, individuals, results):
        results = np.array(results)
        results *= -1
        idx_sorted = np.argsort(results)
        old_mean = self.mean
        self.damp = self.damp * self.tau + (1 - self.tau) * self.damp_limit
        mean_diff = self.weights @ individuals[idx_sorted[:self.parents]] - old_mean
        self.mean = old_mean + mean_diff
        z = (individuals[idx_sorted[:self.parents]] - old_mean)
        self.var = 1 / self.parents * self.weights @ (z * z) + self.damp * np.ones(self.individual_dim)


class OnePlusOnePoluation(Population):
    def __init__(self, individual_dim, mean, args):
        super(OnePlusOnePoluation, self).__init__(1, individual_dim, mean, None)
        self.success_rate = args.aesrl_one_plus_one_success_rate
        self.success_rate_tau = args.aesrl_one_plus_one_success_rate_tau
        # self.no_success_rate_tau = float(np.power(self.success_rate_tau, -0.25))

        self.no_success_rate_tau = 1 / args.aesrl_one_plus_one_success_rate_tau

        self.sigma_init = args.sigma_init
        self.var = self.sigma_init * np.ones(individual_dim)
        self.sigma_limit = args.aesrl_sigma_limit
        self.success_history_len = 10
        self.success_history = deque(maxlen=self.success_history_len)

    def ask(self, n):
        assert n == 1
        epsilon = np.random.randn(self.individual_dim)
        individuals = self.mean + epsilon * np.sqrt(self.var)
        return individuals

    def tell(self, individuals, results):
        assert len(individuals.shape) == 1
        assert individuals.shape[0] == self.individual_dim

        if results > self.mean_result:
            self.mean = individuals
            success = 1
        else:
            success = 0

        if len(self.success_history) < self.success_history_len:
            self.success_history.append(success)
        else:
            success_rate = sum(self.success_history) / self.success_history_len
            # todo: confirm
            if success_rate > self.success_rate:
                self.var /= self.success_rate_tau
            else:
                self.var /= self.no_success_rate_tau
            self.var[self.var < self.sigma_limit] = self.sigma_limit

        return float(success)


class ACemrlPopulation(Population):
    def __init__(self, population_size, individual_dim, mean, args):
        super(ACemrlPopulation, self).__init__(population_size, individual_dim, mean, None)
        self.sigma_init = args.sigma_init
        self.damp = args.cemrl_damp_init
        self.damp_limit = args.cemrl_damp_limit
        self.damp_tau = args.cemrl_damp_tau
        self.tau = float(np.power(args.cemrl_tau, 1/population_size))

        self.var = self.sigma_init * np.ones(individual_dim)

        self.parents = population_size // 2
        self.weights = np.array([np.log((self.parents + 1) / i)
                                 for i in range(1, self.parents + 1)])
        self.weights /= self.weights.sum()
        self.previous_individuals = np.zeros((population_size, individual_dim), dtype=np.float32)
        self.previous_results = np.zeros(population_size, dtype=np.float32)
        self.individual_index = 0
        self.buffer_fill = False

    def ask(self, n):
        assert n == 1
        epsilon = np.random.randn(self.individual_dim)
        individuals = self.mean + epsilon * np.sqrt(self.var)
        return individuals

    def tell(self, individuals, results):
        assert len(individuals.shape) == 1
        assert individuals.shape[0] == self.individual_dim

        self.previous_individuals[self.individual_index] = individuals
        self.previous_results[self.individual_index] = results
        self.individual_index += 1
        if self.individual_index >= self.population_size:
            self.buffer_fill = True
            self.individual_index = 0

        if self.buffer_fill:
            results = copy.deepcopy(np.array(self.previous_results))
            individuals = copy.deepcopy(self.previous_individuals)

            results *= -1
            idx_sorted = np.argsort(results)
            old_mean = self.mean
            self.damp = self.damp * self.tau + (1 - self.tau) * self.damp_limit
            mean_diff = self.weights @ individuals[idx_sorted[:self.parents]] - old_mean
            self.mean = old_mean + mean_diff / self.population_size
            z = (individuals[idx_sorted[:self.parents]] - old_mean)
            self.var = 1 / self.parents * self.weights @ (z * z) + self.damp * np.ones(self.individual_dim)

        return 0


class AesrlPopulation(Population):
    def __init__(self, individual_dim, mean, args):
        super(AesrlPopulation, self).__init__(1, individual_dim, mean, None)

        self.sigma_init = args.sigma_init

        self.var = self.sigma_init * np.ones(individual_dim)

        self.algorithm = args.algorithm
        self.mean_update_rule = args.aesrl_mean_update
        self.mean_param = args.aesrl_mean_update_param
        self.fixed_var_n = args.aesrl_fixed_var_n
        self.sigma_limit = args.aesrl_sigma_limit
        self.var_update_rule = args.aesrl_var_update
        self.negative_ratio = args.aesrl_negative_ratio

        if self.mean_update_rule in ['fixed-linear', 'fixed-sigmoid', 'baseline-relative']:
            assert self.mean_param != 0
        if self.var_update_rule == 'fixed':
            assert self.fixed_var_n != 0

    def ask(self, n):
        assert n == 1
        epsilon = np.random.randn(self.individual_dim)
        individuals = self.mean + epsilon * np.sqrt(self.var)
        return individuals

    def tell(self, individuals, results):
        assert len(individuals.shape) == 1
        assert individuals.shape[0] == self.individual_dim

        if self.mean_update_rule == 'fixed-linear':
            result_diff = results - self.mean_result
            result_diff /= self.mean_param
            p = min(max(result_diff, -1.0), 1.0)

            if result_diff < 0:
                p *= self.negative_ratio

        elif self.mean_update_rule == 'fixed-sigmoid':
            x = (results - self.mean_result) / self.mean_param
            p = 1 / (1 + np.exp(-x))

        elif self.mean_update_rule == 'baseline-absolute':
            result_moved = results - self.mean_param
            mean_moved = self.mean_result - self.mean_param
            if result_moved < 0:
                p = 0.0
            else:
                p = 1 - mean_moved / (result_moved + mean_moved)

            p = min(max(p, 0), 1.0)

        elif self.mean_update_rule == 'baseline-relative':
            moving_baseline = self.mean_result - self.mean_param
            result_moved = results - moving_baseline
            mean_moved = self.mean_result - moving_baseline

            if result_moved < 0:
                p = 0.0
            else:
                p = 1 - mean_moved / (result_moved + mean_moved)

            p = min(max(p, 0), 1.0)

        else:
            raise NotImplementedError

        if self.var_update_rule == 'fixed':
            if p < 1e-3:
                n = 0
            else:
                n = self.fixed_var_n
        elif self.var_update_rule == 'adaptive':
            if p < 1e-3:
                n = 0
            else:
                n = max(1 / p - 1, 1)
        else:
            raise NotImplementedError

        old_mean = copy.deepcopy(self.mean)
        self.mean = self.mean * (1 - p) + p * individuals

        if n != 0:
            self.var += ((individuals - old_mean) * (individuals - self.mean) - self.var) / n
            self.var[self.var < self.sigma_limit] = self.sigma_limit

        return p
