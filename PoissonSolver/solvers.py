import numpy as np

def gauss_seidel(A,b,iter_limit):
    """Solves the equation Ax=b via the Gauss-Seidel iterative method."""
    x = np.zeros_like(b)
    
    for it_count in range(1, iter_limit):
        x_new = np.zeros_like(x)
        print("Iteration {0}: {1}".format(it_count, x))
        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        if np.allclose(x, x_new, rtol=1e-8):
            break
        x = x_new
    return x

def jacobi(A,b,iter_limit):
    """Solves the equation Ax=b via the Jacobi iterative method."""
    # Create an initial guess if needed
    n = np.size(b)
    x = np.zeros_like(b)

    # Create a vector of the diagonal elements of A                                                                                                                                                
    # and subtract them from A                                                                                                                                                                     
    D = np.diag(A)
    R = A - np.diagflat(D)

    # Iterate for N times                                                                                                                                                                          
    for i in range(iter_limit):
        x_new = np.zeros_like(x)
        x_new = (b - np.dot(R,x)) / D
        print("Iteration {0}: {1}".format(i, x))
        
        if np.allclose(x, x_new, rtol=1e-9):
            break
        x = x_new
    return x

def sor_solver(A, b, omega, initial_guess, convergence_criteria):
  """
  This is an implementation of the pseduo-code provided in the Wikipedia article.
  Inputs:
    A: nxn numpy matrix
    b: n dimensional numpy vector
    omega: relaxation factor
    initial_guess: An initial solution guess for the solver to start with
  Returns:
    phi: solution vector of dimension n
  """
  count = 0
  phi = initial_guess[:]
  residual = np.linalg.norm(np.matmul(A, phi) - b) #Initial residual
  while residual > convergence_criteria:
    count = count + 1
    
    for i in range(A.shape[0]):
      sigma = 0
      for j in range(A.shape[1]):
        if j != i:
          sigma += A[i][j] * phi[j]
      phi[i] = (1 - omega) * phi[i] + (omega / A[i][i]) * (b[i] - sigma)
    residual = np.linalg.norm(np.matmul(A, phi) - b)
    
  return phi
