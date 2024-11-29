import jax_cgd as jcgd
import jax
import jax.numpy as jnp

if __name__ == "__main__":
    # target function
    def f(x):
        """
        f(x) = (x1 - 3)^2 + (x2 - 5)^2
        """
        return jnp.sum((x - jnp.array([3.0, 5.0])) ** 2)
    # constraints
    def g(x):
        """
        g(x) = Ax - b
        A = [[1, -1], [-1, -1]], b = [1, -3]
        """
        A = jnp.array([[1.0, -1.0], [-1.0, -1.0]])
        b = jnp.array([1.0, -3.0])
        return A @ x - b

    def lagrangian(x_dict, y_dict):
        """
        L(x, y) = f(x) + y^2^T * g(x) 
        """
        x = x_dict["x"]
        y = y_dict["y"]
        return jnp.sum(f(x) - jnp.dot(y ** 2, g(x)))

    x_params_dict = {"x": jnp.array([3.0, 5.0])}  
    y_params_dict = {"y": jnp.array([5., 5.])}

    lr = 0.5
    beta = 0.9
    eps = 1e-3

    solver = jcgd.solvers.GMRES()
    
    # optimizer = ACGD(x_params_dict, y_params_dict, lagrangian, lr, beta, eps, solver)
    optimizer = jcgd.BCGD(x_params_dict, y_params_dict, lagrangian, lr, solver)
    
    num_steps = 10000
    for step in range(num_steps):

        optimizer.step()
        x_params, y_params, x_params_dict, y_params_dict = optimizer.get_infos()

        print(f"Step {step + 1}:")
        print(f"  x_params: {x_params_dict}")
        print(f"  y_params: {y_params_dict}")
        print(f"  Lagrangian value: {lagrangian(x_params_dict, y_params_dict):.4f}")
        print(f"Constraint value: {g(x_params_dict['x'])}\n")

    print("\nOptimization completed.")
    print(f"Final x_params: {x_params_dict}")
    print(f"Final y_params: {y_params_dict}")
    print(f"Final Lagrangian value: {lagrangian(x_params_dict, y_params_dict):.4f}")

   
