import numpy as np
import math
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import norm
import matplotlib.pyplot as plt 
import time

def simulate_option_path(S0, T, r, sigma, Z):
    """
    Simulate the final stock price using the Black-Scholes model and random normal variable Z.
    """
    return S0 * math.exp((r - 0.5 * sigma ** 2) * T + sigma * math.sqrt(T) * Z)

def call_option_payoff(S_T, K):
    """
    Calculate the payoff of a call option at maturity.
    """
    return max(S_T - K, 0)

def monte_carlo_call_option(S0, K, T, r, sigma, num_simulations):
    """
    Estimate the price of a European call option using the Monte Carlo simulation.
    """
    np.random.seed(235) 
    Z = np.random.normal(size=num_simulations)
    sim_end_prices = [simulate_option_path(S0, T, r, sigma, z) for z in Z]
    payoffs = [call_option_payoff(s, K) for s in sim_end_prices]
    discounted_payoffs = [math.exp(-r * T) * payoff for payoff in payoffs]
    return np.mean(discounted_payoffs)


def monte_carlo_call_option_parallel(S0, K, T, r, sigma, num_simulations, num_workers):
    """
    Estimate the price of a European call option using Monte Carlo simulation in parallel.
    
    :param S0: Initial stock price
    :param K: Strike price
    :param T: Time to expiration in years
    :param r: Risk-free rate
    :param sigma: Volatility
    :param num_simulations: Total number of paths simulated
    :param num_workers: Number of parallel workers
    :return: Estimated call option price
    """
    # Divide the simulations among workers
    chunk_size = num_simulations // num_workers

    print(chunk_size, num_simulations, num_workers)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(monte_carlo_call_option, S0, K, T, r, sigma, chunk_size)
            for _ in range(num_workers)
        ]
        results = [f.result() for f in futures]

    return np.mean(results)

def compare_black_scholes_workers_performance(S0, K, T, r, sigma, num_simulations, max_workers):
    """
    Compare the performance of the Black-Scholes Monte Carlo simulation with different numbers of workers.
    
    :param S0: Initial stock price
    :param K: Strike price
    :param T: Time to expiration in years
    :param r: Risk-free rate
    :param sigma: Volatility
    :param num_simulations: Total number of paths simulated
    :param max_workers: Maximum number of parallel workers to test
    :return: List of tuples containing (num_workers, estimated option price, elapsed time)
    """
    results = []
    for num_workers in range(1, max_workers + 1):
        start_time = time.time()
        option_price = monte_carlo_call_option_parallel(S0, K, T, r, sigma, num_simulations, num_workers)
        end_time = time.time()
        elapsed_time = end_time - start_time
        results.append((num_workers, option_price, elapsed_time))
        print(f"Workers: {num_workers}, Estimated Option Price: {option_price:.2f}, Time: {elapsed_time:.2f} seconds")
    
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
       # Parameters
    S0 = 100       # Initial stock price
    K = 100        # Strike price
    T = 1          # Time to expiration in years
    r = 0.05       # Risk-free rate
    sigma = 0.2    # Volatility
    num_simulations = 100_000_000  # Number of paths simulated
    max_workers = 11         # Maximum number of workers to test

    print("Comparing performance with different numbers of workers:")
    results = compare_black_scholes_workers_performance(S0, K, T, r, sigma, num_simulations, max_workers)

    # Optionally, plot the performance
    plot_performance(results)