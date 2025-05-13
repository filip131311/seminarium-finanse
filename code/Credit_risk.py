import numpy as np
from concurrent.futures import ProcessPoolExecutor

def simulate_default(prob_default, economic_factor):
    """ Simulate default occurrence based on default probability and economic factor. """
    random_threshold = np.random.random()
    adjusted_prob = prob_default * (1 + economic_factor)
    return 1 if random_threshold < adjusted_prob else 0

def simulate_loan_defaults(n, prob_default, economic_factor):
    """ Simulate defaults for n loans. """
    defaults = [simulate_default(prob_default, economic_factor) for _ in range(n)]
    return sum(defaults)

def main():
    n_loans = 1000
    base_prob_default = 0.05  # Base probability of default
    economic_scenarios = [-0.2, -0.1, 0.0, 0.1, 0.2]  # Economic factors

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(simulate_loan_defaults, n_loans, base_prob_default, factor) for factor in economic_scenarios]
        default_counts = [future.result() for future in futures]

    for factor, count in zip(economic_scenarios, default_counts):
        print(f"Economic Factor {factor}: Total defaults = {count}")

if __name__ == "__main__":
    main()