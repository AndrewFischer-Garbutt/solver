import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter



class Solver:
    def __init__(self, tolerance, debug=False, logData=False):
        self.debug = debug
        self.logData = logData

        self.SOR_residuals = []
        self.gauss_seidel_redisuals = []
        self.jocabi_residuals = []
        self.conjugate_gradient_residuals = []
        self.steepest_descent_residuals = []

        self.max_iterations = 1000
        self.exit_on_tolerance = True

        self.tolerance = tolerance

    def plot_reduals(self):
        iterations = np.arange(40) + 1
        plt.plot(iterations, self.SOR_residuals[0:40], label='SOR')
        plt.plot(iterations, self.gauss_seidel_redisuals[0:40], label='Gauss Seidel')
        plt.plot(iterations, self.jocabi_residuals[0:40], label='Jocabi method')

        plt.legend()

        plt.show()

    def print_divider(self):
        print("-" * 50)

    def print_iterations_needed(self):
        if self.SOR_residuals:
            num_iterations = len(self.SOR_residuals)
            self.print_divider()
            print("Iterations needed for SOR method: {}".format(num_iterations))

            SOR_ratio = self.SOR_residuals[num_iterations - 1] / self.SOR_residuals[num_iterations - 2]

            print("The ratio of the last two residuals is: {0:.2f}".format(SOR_ratio))
            self.print_divider()

        if self.gauss_seidel_redisuals:
            num_iterations = len(self.gauss_seidel_redisuals)
            print("Iterations needed for Gauss Seidel: {}".format(num_iterations))

            Gauss_Seidel_ratio = self.gauss_seidel_redisuals[num_iterations - 1] / self.gauss_seidel_redisuals[num_iterations - 2]

            print("The ratio of the last two residuals is: {0:.2f}".format(Gauss_Seidel_ratio))
            self.print_divider()
        
        if self.jocabi_residuals:
            num_iterations = len(self.jocabi_residuals)
            print("Iterations needed for Jocabi Iteration: {0:.2f}".format(num_iterations))

            Jocabi_ratio = self.jocabi_residuals[num_iterations - 1] / self.jocabi_residuals[num_iterations - 2]

            print("The ratio of the last two residuals is: {0:.2f}".format(Jocabi_ratio))
            self.print_divider()

        if self.conjugate_gradient_residuals:
            num_iterations = len(self.conjugate_gradient_residuals)
            print("Iterations needed for Conjugate Gradient: {}".format(num_iterations))

            conj_grandient_ratio = self.conjugate_gradient_residuals[num_iterations - 1] / self.conjugate_gradient_residuals[num_iterations - 2]

            print("The ratio of the last two residuals is: {0:.2f}".format(conj_grandient_ratio))
            self.print_divider()
        if self.steepest_descent_residuals:
            num_iterations = len(self.steepest_descent_residuals)

            print("Iterations needed for Steepest Descent: {}".format(num_iterations))

            steepest_descent_ratio = self.steepest_descent_residuals[num_iterations - 1] / self.steepest_descent_residuals[num_iterations - 2]

            print("The ratio of the last two residuals is: {0:.2f}".format(steepest_descent_ratio))
            self.print_divider()

    def sum_without(self, M, x, i):   
        return np.inner(M[i], x) - M[i][i] * x[i]

    def SOR_Solver(self, M, b, omega, x):
        for _ in range(self.max_iterations):
            for i in range(x.shape[0]):
                x[i] = (1 - omega) * x[i] + omega / M[i][i] * ( b[i] - self.sum_without(M, x, i) )

            residual = LA.norm(np.matmul(M, x) - b)

            if self.debug:
                print("SOR residual: ", residual)
                print("x: ", x)

            if self.logData:
                self.SOR_residuals.append(residual)

            if residual < self.tolerance and self.exit_on_tolerance:
                return x

        print("Took to long!")
        return x

    def gauss_seidel(self, M, b, x):
        for _ in range(self.max_iterations):
            for i in range(x.shape[0]):
                x[i] = ( b[i] - self.sum_without(M, x, i) ) / M[i][i]

            residual = LA.norm(np.matmul(M, x) - b)

            if self.debug:
                print("Gauss-Seidel residual: ", residual)
                print("x: ", x)

            if self.logData:
                self.gauss_seidel_redisuals.append(residual)

            if residual < self.tolerance and self.exit_on_tolerance:
                return x

        print("Took to long!")
        return x



    def jocabi_solver(self, M, b, x):
        x1 = np.copy(x)
        for _ in range(self.max_iterations):
            for i in range(x.shape[0]):
                x1[i] = ( b[i] - self.sum_without(M, x, i) ) / M[i][i]

            x = np.copy(x1)

            residual = LA.norm(np.matmul(M, x) - b)
            if residual < self.tolerance and self.exit_on_tolerance: 
               return x

            if self.debug: 
                print("Jocabi residual: ", residual)
                print("x1: ", x1)

            if self.logData:
                self.jocabi_residuals.append(residual)


        print("Took to long!")
        return x

    def conjugate_gradient(self, A, x, b):
        r = b - np.matmul(A, x)

        if LA.norm(r) < self.tolerance and self.exit_on_tolerance:
            return x

        p = r

        for _ in range(self.max_iterations):
            Ap = np.matmul(A, p)
            alpha = np.inner(r, r) / np.inner(p, Ap)
            x = x + alpha * p
            r_cpy = np.copy(r)
            r = r - alpha * Ap

            residual = LA.norm(r)

            if residual < self.tolerance and self.exit_on_tolerance:
                return x

            if self.logData:
                self.conjugate_gradient_residuals.append(residual)

            beta = np.inner(r, r) / np.inner(r_cpy, r_cpy)
            p = r + beta * p



        print("Took more than {} iterations".format(self.max_iterations))
        return x

    def steepest_descent(self, A, x, b):
        for _ in range(self.max_iterations):
            r = b - np.matmul(A, x)
            alpha = np.inner(r, r) / np.inner(r, np.matmul(A, r))
            x = x + alpha * r

            residual = LA.norm(b - np.matmul(A, x))

            if residual < self.tolerance and self.exit_on_tolerance:
                return x
            
            if self.logData:
                self.steepest_descent_residuals.append(residual)

            if self.debug: 
                print("Steepest descent: ", residual)
                print("x: ", x)
            
        print(f"Took more than {self.max_iterations} iterations")
        return x
    
    def calculate_optimal_relaxation(self, M):
        D_inv = LA.inv(np.diag(np.diag(M)))
        R_J = np.identity(M.shape[0]) - np.matmul(D_inv, M)
        vals = LA.eigvals(R_J)

        if self.debug:
            print("Eigenvalues: ", LA.eigvals(M))

        mu = max([abs(x) for x in vals])

        # return (1 + np.sqrt(1 - mu * mu)) / 2
        return 1 + (mu / (1 + np.sqrt(1 - mu * mu))) ** 2


class LaplaciansEqnSolver:
    def __init__(self, mesh_size):
        self.debug = False
        self.mesh_size = mesh_size


    def generate_laplacian(self):
        n = int(1 / self.mesh_size)
    
        Adim = (n - 1) ** 2
        A = np.zeros((Adim, Adim), dtype=float)
   
        if self.debug:
            print(A.shape)
    
            print(Adim - 1)

        h_sqr = self.mesh_size * self.mesh_size
    
        for i in range(1, n):
            for j in range(1, n):
                # print(i, j)
                A[(j - 1) + (n - 1) * (i - 1)][(j - 1) + (n - 1) * (i - 1)] = 4 / h_sqr

                if j + 1 != n:
                    A[j + (n - 1) * (i - 1)][(j - 1) + (n - 1) * (i - 1)] = -1 / h_sqr
                
                if j - 1 != 0:
                    A[(j - 2) + (n - 1) * (i - 1)][(j - 1) + (n - 1) * (i - 1)] = -1 / h_sqr
                
                if i + 1 != n:
                    A[(j - 1) + (n - 1) * i][(j - 1) + (n - 1) * (i - 1)] = -1 / h_sqr
                
                if i - 1 != 0:
                    A[(j - 1) + (n - 1) * (i - 2)][(j - 1) + (n - 1) * (i - 1)] = -1 / h_sqr

        
        if self.debug: 
            Str = np.array2string(A, max_line_width=np.inf)
            print(Str)
    
        return A

    def generate_boundary_values(self):
       n = int(1 / self.mesh_size)
       h_sqr = self.mesh_size * self.mesh_size
       dimension = (n - 1) ** 2

       b = np.zeros(dimension)

       for j in range(1, n):
            b[j - 1] = j * self.mesh_size * (1 - j * self.mesh_size) / h_sqr

       return b

def surface_plot (matrix, **kwargs):
    # acquire the cartesian coordinate matrices from the matrix
    # x is cols, y is rows
    (x, y) = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, matrix, **kwargs)
    return (fig, ax, surf)

def show_image(M):
    # M = x5.reshape((n - 1, n - 1))

    (fig, ax, surf) = surface_plot(M, cmap=plt.cm.coolwarm)

    fig.colorbar(surf)

    ax.set_xlabel('X (cols)')
    ax.set_ylabel('Y (rows)')
    ax.set_zlabel('Z (values)')

    plt.show()

def solve_laplace(h):
    
    n = int(1 / h)

    print("Running with meshsize: {}".format(h))

    solDim = (n - 1) ** 2

    LapSolver = LaplaciansEqnSolver(h)
    L = LapSolver.generate_laplacian()
    b = LapSolver.generate_boundary_values()

    # print(L)

    # str1 = np.array2string(L)
    # str2 = np.array2string(b)

    # print(str1)
    # print(str2)

    x1 = np.zeros(solDim, dtype=float)
    x2 = np.zeros(solDim, dtype=float)
    x3 = np.zeros(solDim, dtype=float)
    x4 = np.zeros(solDim, dtype=float)
    x5 = np.zeros(solDim, dtype=float)
    
    solver = Solver(0.001 * LA.norm(b), debug=False, logData=True)

    omega = solver.calculate_optimal_relaxation(L)

    print_times = False
    
    import time

    start = time.perf_counter()
    solver.SOR_Solver(L, b, omega, x1)
    end = time.perf_counter()

    if print_times:
        print("Finished SOR in {} seconds".format(end - start))
    start = time.perf_counter()
    solver.gauss_seidel(L, b, x2)
    end = time.perf_counter()

    if print_times:
        print("Finished Gauss Seidel in {} seconds".format(end - start))
    start = time.perf_counter()
    x3 = solver.jocabi_solver(L, b, x3)
    end = time.perf_counter()
    if print_times:
        print("Finished Jocabi Solver in {} seconds".format(end - start))
    start = time.perf_counter()
    x4 = solver.conjugate_gradient(L, x4, b)
    end = time.perf_counter()
    if print_times:
        print("Finished Conjugate gradient in {} seconds".format(end - start))
    start = time.perf_counter()
    x5 = solver.steepest_descent(L, x5, b)
    end = time.perf_counter()
    if print_times:
        print("Finished Steepest descent in {} seconds".format(end - start))

    solver.print_iterations_needed()

    plot_image = True

    # Plot of solution
    if plot_image:
        M = np.resize(x5,(n - 1, n -1))
        show_image(M)
    




def run_solvers():
    coff = np.array(
                [[ 4.0, 3.0, 0.0],
                [ 3.0, 4.0, -1.0],
                [ 0.0, -1.0, 4.0]]
            )
    
    solver = Solver(1e-7, debug=False, logData=True)
    
    values = np.array( [ 24.0, 30.0, -24.0 ] )
    
    x1 = np.array( [ 0.0, 0.0, 0.0 ] )
    x2 = np.array( [ 0.0, 0.0, 0.0 ] )
    x3 = np.array( [ 0.0, 0.0, 0.0 ] ) 
    # x4 = np.array( [ 0.0, 0.0, 0.0 ] )
    # x5 = np.array( [ 0.0, 0.0, 0.0 ] )
    
    
        
    omega = solver.calculate_optimal_relaxation(coff)
    print("Optimal Relaxation: ", omega)
    
    
    
    solver.SOR_Solver(coff, values, omega, x1)
    solver.gauss_seidel(coff, values, x2)
    solver.jocabi_solver(coff, values, x3)

    
    # solver.plot_reduals()
    
    # nonlin = NonSolver(1e-7)
    # print("Output of conjugate gradient: ")
    # print(nonlin.conjugate_gradient(coff, x4, values))
    
    # print("Output of steepest descent: ")
    # print(nonlin.steepest_descent(coff, x5, values))
    
    solver.print_iterations_needed()

def test_steepest_descent():
    solver = Solver(0.01, debug=True, logData=True)

    A = np.array(
        [[3.0, -1.0, 1.0],
         [-1.0, 3.0, -1.0],
         [1.0, -1.0, 3.0]]
    )

    b = np.array([-1.0, 7.0, -7.0])
    x = np.array([0.0, 0.0, 0.0])

    solver.steepest_descent(A, x, b)

if __name__ == '__main__':
    # for i in range(2, 6):
    #     solve_laplace(1.0 / (2.0 ** i))
    solve_laplace(1.0 / (2.0 ** 5))
