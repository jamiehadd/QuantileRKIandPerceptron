import numpy as np
import random
import matplotlib.pyplot as plt

def QuantileRK(A, b, q, t, N, correct_labels, labels, numMislabelled, numDataPoints):
    A = np.array(A)
    m, n = A.shape
    x = np.random.rand(n)  # Initial guess for the solution
    residuals = []  # To store the residuals for plotting

    # Create a boolean mask for the labels that are in correct_labels but not in labels
    
    difference_mask = np.where(correct_labels == labels)  # Only retain the spots where they are equal

    # Apply the mask to filter the rows
    A_uncorrupted = A[difference_mask]
    b_uncorrupted = b[difference_mask]

    random.seed(0)  # For reproducibility

    # Iterate for N steps
    for j in range(N):
        condition = np.dot(A_uncorrupted, x) > b_uncorrupted  # Compute the condition

        # Count the number of elements that satisfy the condition as a percent of correct inequalities
        not_set_count = np.count_nonzero(condition) / (numDataPoints - numMislabelled) * 100

        # Store the count in the residuals list
        residuals.append(not_set_count)

        # ALGORITHM IS FROM HERE DOWN
        # Randomly sample t indices from the set of m indices
        sampled_indices = np.random.choice(m, t, replace=True)

        # Pick a random index k from the m rows
        k = random.choice(np.arange(m))

        # Compute the corresponding row of A and b
        a_k = A[k]
        b_k = b[k]

        # Compute the expression for the selected index
        e = np.maximum((np.dot(x, a_k) - b_k), 0)  # Part of the update rule

        # Compute residuals for the sampled indices
        nyet_list_of_distances = A[sampled_indices] @ x - b[sampled_indices]
        nonnegative_residuals = np.maximum(nyet_list_of_distances, 0)  # Non-negative residuals

        # Check if the expression is less than or equal to the q-th quantile
        if e <= np.quantile(nonnegative_residuals, q):
            x = x - e * a_k  # Update the solution vector
        else:
            x = x  # No update if the condition is not satisfied

    # Plot residuals
    plt.plot(range(1, N + 1), residuals)
    
    # Increase font size for axes labels and ticks
    plt.xlabel("Iteration Number", fontsize=16)  # Increase x-axis label font size
    plt.ylabel("Percent of Misclassified Points", fontsize=16)  # Increase y-axis label font size
    plt.xticks(fontsize=14)  # Increase font size for x-axis ticks
    plt.yticks(fontsize=14)  # Increase font size for y-axis ticks

    # Remove the title
    plt.title("")  # Title is removed (empty string)

    # Show the plot
    plt.show()

    # Return the final solution and residuals
    return x, residuals


"""Modified Perceptron algorithm"""

def perceptronModified(data,labels,q, t, N,correct_labels,numMislabelled,numDataPoints):
    n, m = data.shape
    X = data.T
    Y=labels
    Y = np.diag(Y)
    X_tilda = np.matmul(Y, X)
    X_tilda = np.negative(X_tilda)
    x,residuals= QuantileRK(X_tilda,np.zeros((m,)), q, t, N,correct_labels,labels,numMislabelled,numDataPoints) # this inherits necessary files
    print("Decision boundary is", x)
    return x, residuals

