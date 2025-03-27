import numpy as np
import random
import matplotlib.pyplot as plt

def QuantileRK(A, b, q, t, N, correct_labels, labels, numMislabelled, numDataPoints):
    A = np.array(A)
    m, n = A.shape
    x = np.random.rand(n)  # Initial guess for the solution
    residuals = []  # To store the residuals for plotting
    cumulative_times = []  # List to store cumulative times

    # Create a boolean mask for the labels that are in correct_labels but not in labels
    difference_mask = np.where(correct_labels == labels)  # Only retain the spots where they are equal

    # Apply the mask to filter the rows
    A_uncorrupted = A[difference_mask]
    b_uncorrupted = b[difference_mask]

    random.seed(0)  # For reproducibility

    cumulative_time = 0  # Initialize cumulative time

    # Iterate for N steps
    for j in range(N):
        start_time = time.time()  # Start time for this iteration

        condition = np.dot(A_uncorrupted, x) > b_uncorrupted  # Compute the condition

        # Count the number of elements that satisfy the condition as a percent of correct inequalities
        not_set_count = np.count_nonzero(condition) / (numDataPoints - numMislabelled) * 100

        residuals.append(not_set_count)

        # ALGORITHM IS FROM HERE DOWN
        # Randomly sample t indices from the set of m indices
        sampled_indices = np.random.choice(m, t, replace=True)

        # Pick a random index k from the m rows
        k = random.choice(np.arange(m))
        a_k = A[k]
        b_k = b[k]

        # Compute the expression for the selected index
        e = np.maximum((np.dot(x, a_k) - b_k), 0)

        # Residuals for the sampled indices
        nyet_list_of_distances = A[sampled_indices] @ x - b[sampled_indices]
        nonnegative_residuals = np.maximum(nyet_list_of_distances, 0)  # Non-negative residuals

        if e <= np.quantile(nonnegative_residuals, q):
            x = x - e * a_k
        else:
            x = x  # No update if the condition is not satisfied

        end_time = time.time() # End time for this iteration
        iteration_time = end_time - start_time
        cumulative_time += iteration_time
        cumulative_times.append(cumulative_time)  # Store the cumulative time

    return x, residuals, cumulative_times

def perceptronModified(data, labels, q, t, N, correct_labels, numMislabelled, numDataPoints):
    n, m = data.shape
    X = data.T
    Y = labels
    Y = np.diag(Y)
    X_tilda = np.matmul(Y, X)
    X_tilda = np.negative(X_tilda)
    x, residuals, cumulative_times = QuantileRK(X_tilda, np.zeros((m,)), q, t, N, correct_labels, labels, numMislabelled, numDataPoints)

    return x, residuals, cumulative_times
