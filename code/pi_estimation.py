import random
import time
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt  

def estimate_pi(num_samples):
    inside_circle = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random()
        if x**2 + y**2 <= 1.0:
            inside_circle += 1
    return (4.0 * inside_circle) / num_samples

# def parallel_pi_estimate_wrong(total_samples, num_workers):
#     samples_per_worker = total_samples // num_workers
#     with ProcessPoolExecutor(max_workers=num_workers) as executor:
#         futures = [executor.submit(estimate_pi, samples_per_worker) for _ in range(num_workers)]
#         estimates = [future.result() for future in futures]

#     mean_estimate = sum(estimates) / num_workers
#     return mean_estimate

def count_points_in_circe(num_samples):
    inside_circle = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random()
        if x**2 + y**2 <= 1.0:
            inside_circle += 1
    return inside_circle

def parallel_pi_estimate_correct(total_samples, num_workers):
    samples_per_worker = total_samples // num_workers

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(count_points_in_circe, samples_per_worker) for _ in range(num_workers)]
        num_of_points = [future.result() for future in futures]

    estimate = 4 * sum(num_of_points) / (num_workers * samples_per_worker)
    return estimate

def compare_workers_performance(total_samples, max_workers):
    results = []
    for num_workers in range(1, max_workers + 1):
        start_time = time.time()
        pi_estimate = parallel_pi_estimate_correct(total_samples, num_workers)
        end_time = time.time()
        elapsed_time = end_time - start_time
        results.append((num_workers, pi_estimate, elapsed_time))
        print(f"Workers: {num_workers}, Estimated π: {pi_estimate}, Time: {elapsed_time:.2f} seconds")
    
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
    TOTAL_SAMPLES = 10_000_000
    MAX_WORKERS = 11  
    
    print("Comparing performance with different numbers of workers:")
    results = compare_workers_performance(TOTAL_SAMPLES, MAX_WORKERS)
    
    plot_performance(results)