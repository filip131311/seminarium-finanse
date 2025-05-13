import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import time

def simulate_batch(args):
    theta, mu, sigma, r0, T, N, batch_size = args
    results = []
    for _ in range(batch_size):
        dt = T / N
        rates = np.zeros(N + 1)
        rates[0] = r0
        for i in range(1, N + 1):
            dW = np.random.normal(0, np.sqrt(dt))
            rates[i] = rates[i-1] + theta * (mu - rates[i-1]) * dt + sigma * dW
        results.append(rates[-1])
    return np.mean(results)

class IRM:
    def __init__(self, theta: float, mu: float, sigma: float, r0: float):
        """
        Initialize the interest rate model parameters.
        
        :param theta: Speed of reversion
        :param mu: Long-term mean level
        :param sigma: Volatility
        :param r0: Initial interest rate
        """
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.r0 = r0
    
    def vasicek_euler_maruyama_path(self, T: float, N: int):
        """
        Simulate vasicek paths using the Euler-Maruyama scheme.
        
        :param T: Time horizon
        :param N: Number of time steps
        :return: Simulated paths
        """
        dt = T / N
        rates = np.zeros(N + 1)
        rates[0] = self.r0
        for i in range(1, N + 1):
            dW = np.random.normal(0, np.sqrt(dt))  # Generate random increments
            rates[i] = rates[i-1] + self.theta * (self.mu - rates[i-1]) * dt + self.sigma * dW
        return rates
    

    def calibrate(self, market_data: np.ndarray, path_method = 'vasicek_euler_maruyama_path'):
        """
        Calibrate the model parameters to market data.
        
        :param market_data: Array of market data for calibration
        :param path_method: A name of path method to be used for calibration, if none chosen, vasicek_euler_maruyama_path is the default
            Options: vasicek_euler_maruyama_path
        """
        def objective(params):
            theta, mu, sigma = params
            model = IRM(theta, mu, sigma, self.r0)
            method = getattr(model, path_method)
            simulated_data = method(T=len(market_data)-1, N=len(market_data)-1)
            simulated_data = simulated_data.flatten()
            return np.sum((market_data - simulated_data)**2)
        
        initial_guess = [self.theta, np.mean(market_data), np.std(market_data)]
        result = minimize(objective, initial_guess, bounds=[(0, None), (0, None), (0, None)])
        self.theta, self.mu, self.sigma = result.x
        print(f"Calibrated parameters: theta={self.theta}, mu={self.mu}, sigma={self.sigma}")

    def simulate_euler_maruyama_average_parallel(self, T: float, N: int, n_simulations: int, n_workers: int):
        """
        Simulate Euler-Maruyama paths multiple times in parallel and return the average value at T.
        
        :param T: Time horizon
        :param N: Number of time steps
        :param n_simulations: Number of simulations to run
        :param n_workers: Number of parallel workers
        :return: Average value of the interest rate at T
        """

        batch_size = max(1, n_simulations // (n_workers))

        # Use ProcessPoolExecutor to parallelize the simulations
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(simulate_batch, (self.theta, self.mu, self.sigma, self.r0, T, N, batch_size)) for _ in range(n_workers)]
            results = [future.result() for future in futures]
        return np.mean(results)
    
    def compare_workers_performance(self, T: float, N: int, n_simulations: int, n_workers: int):
        results = []
        for num_workers in range(1, n_workers + 1):
            start_time = time.time()
            ir_estimate = self.simulate_euler_maruyama_average_parallel(T, N, n_simulations, num_workers)
            end_time = time.time()
            elapsed_time = end_time - start_time
            results.append((num_workers, ir_estimate, elapsed_time))
            print(f"Workers: {num_workers}, Estimated interest rate: {ir_estimate}, Time: {elapsed_time:.2f} seconds")
        
        return results
    
def plot_performance(results):
    num_workers = [result[0] for result in results]
    times = [result[2] for result in results]

    # Scale 1/x to match the first execution time
    scaled_1_over_x = [times[0] / workers for workers in num_workers]

    plt.figure(figsize=(10, 6))
    plt.plot(num_workers, times, marker='o', linestyle='-', color='b', label='Execution Time')
    plt.plot(num_workers, scaled_1_over_x, marker='x', linestyle='--', color='r', label='Scaled 1/x')
    plt.title('Performance of π Estimation with Different Numbers of Workers')
    plt.xlabel('Number of Workers')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    TOTAL_SIMULATIONS = 10_000
    MAX_WORKERS = 11  

    file_path = './data/market_data_PL.csv'

    # Read the CSV file into a DataFrame
    data = pd.read_csv(file_path, header=None, names=['Date', 'Period', 'Value'])

    # Convert the 'Value' column to float
    data['Value'] = data['Value'].astype(float)

    market_data = list(map(lambda value: value[2], data.values))

    model = IRM(r0=market_data[0], theta=0.5, mu=0.03, sigma=0.02)
    model.calibrate(market_data)
    results = model.compare_workers_performance(T=10, N=1_000, n_simulations=TOTAL_SIMULATIONS, n_workers=MAX_WORKERS)
    plot_performance(results)