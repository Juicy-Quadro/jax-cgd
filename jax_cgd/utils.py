import jax
import jax.numpy as jnp
from jax import grad, jacrev, jacfwd, jvp
import jax.lax as lax
import jax.numpy as jnp
import numpy as np

def params_to_array(params_dict):
    """
    Convert all jnp.array in the dictionary to a single 1D jnp.array in a certain order.

    Args:
        params_dict (dict): Dictionary with keys as strings and values as jnp.array.

    Returns:
        params_arr (jnp.ndarray): A 1D array concatenating all parameters.
        metadata (list): List of metadata recording each parameter's key and shape for reconstruction.
    """
    params_list = []
    metadata = []

    for key, value in params_dict.items():
        params_list.append(value.ravel())  # Flatten the array to 1D
        metadata.append((key, value.shape))  # Save the key and its original shape

    params_arr = jnp.concatenate(params_list)  # Concatenate into a single 1D array
    return params_arr, tuple(metadata)

# def params_to_dict(params_arr, metadata):
#     """
#     Convert a 1D jnp.array back to a dictionary.

#     Args:
#         params_arr (jnp.ndarray): 1D array containing all parameter values.
#         metadata (list): List of metadata recording each parameter's key and shape.

#     Returns:
#         params_dict (dict): Dictionary with keys as strings and values as jnp.array, restored to original structure.
#     """
#     params_dict = {}
#     offset = 0

#     for key, shape in metadata:
#         size = jnp.prod(jnp.array(shape))  # Calculate the total number of elements for this parameter
#         param_values = params_arr[offset:offset + size]  # Slice out the corresponding parameter values
#         params_dict[key] = param_values.reshape(shape)  # Restore the original shape
#         offset += size

#     return params_dict

def params_to_dict(params_arr, metadata):
    """
    Reconstruct a dictionary of parameters from a flattened array and metadata.

    Args:
        params_arr: Flattened 1D array of all parameters.
        metadata: Metadata for reconstructing the dictionary.

    Returns:
        params_dict: Reconstructed dictionary of parameters.
    """
    params_dict = {}
    offset = 0
    for key, shape in metadata:
        size = int(np.prod(np.array(shape)))  # Ensure size is a concrete integer
        # Use lax.dynamic_slice for slicing
        param_values = lax.dynamic_slice(params_arr, (offset,), (size,))
        params_dict[key] = param_values.reshape(shape)
        offset += size
    return params_dict




def H_xy_vy(h, x, y, vy):
    """
    Compute the Hessian-Vector Product: H_xy * vy.

    Args:
        h: Callable, scalar function h(x, y).
        x: 1D array, parameters with respect to x.
        y: 1D array, parameters with respect to y.
        vy: 1D array, vector to multiply with the Hessian H_xy (same shape as y).

    Returns:
        1D array, result of H_xy * vy (same shape as x).
    """
    # Define the function g(x, y) = ∂h / ∂x
    grad_x = lambda x_, y_: jax.grad(h, argnums=0)(x_, y_)

    # Compute the Hessian-vector product: (∂²h / ∂x ∂y^T) * vy
    _, hvp = jax.jvp(lambda y_: grad_x(x, y_), (y,), (vy,))
    return hvp

def H_yx_vx(h, x, y, vx):
    """
    Compute the Hessian-Vector Product: H_yx * vx.

    Args:
        h: Callable, scalar function h(x, y).
        x: 1D array, parameters with respect to x.
        y: 1D array, parameters with respect to y.
        vx: 1D array, vector to multiply with the Hessian H_yx (same shape as x).

    Returns:
        1D array, result of H_yx * vx (same shape as y).
    """
    # Define the function g(y, x) = ∂h / ∂y
    grad_y = lambda x_, y_: jax.grad(h, argnums=1)(x_, y_)

    # Compute the Hessian-vector product: (∂²h / ∂y ∂x^T) * vx
    _, hvp = jax.jvp(lambda x_: grad_y(x_, y), (x,), (vx,))
    return hvp

class Operater:
    def __init__(self, f, y_params_arr, x_params_arr, eta_y, eta_x):


        self.f = f
        self.y_params_arr = y_params_arr
        self.x_params_arr = x_params_arr
        self.eta_y = eta_y
        self.eta_x = eta_x
        self.shape = self.shape = (len(x_params_arr), len(x_params_arr))

    def __matmul__(self, v: jnp.array) -> jnp.array:
        """
        Performs a matrix-free matrix-vector product in JAX.

        Args:
            v: Input vector for the matrix-vector product (1D array). The same shape as x_params_arr.

        Returns:
            result: The result of the matrix-vector product (1D array).
        """
        # Scale the input vector by sqrt(eta_x) A^{1/2}_{x,t} v
        v0 = self.eta_x ** 0.5 * v

        # Compute the first Hessian-vector product: A_{y,t} * (D^2_{yx}f * v0)
        grad_fn1 = H_yx_vx(self.f, self.x_params_arr, self.y_params_arr, v0)
        v1 = self.eta_y * grad_fn1

        # Compute the second Hessian-vector product: A^{1/2}_{x,t} * (D^2_{xy}f * v1)
        grad_fn2 = H_xy_vy(self.f, self.x_params_arr, self.y_params_arr, v1)
        v2 = self.eta_x ** 0.5 * grad_fn2

        # Add the identity contribution: I * v + v2
        result = v + v2

        return result
    
    def as_function(self):
        return lambda v: self @ v


        


# def Hvp_vec(grad_fn, params, vec):
#     """
#     Hessian-vector product using JAX.
    
#     :param grad_fn: Gradient function w.r.t which the Hessian will be computed
#     :param params: Parameters for which the Hessian will be computed
#     :param vec: The vector in Hessian-vector product
#     :return: Hessian vector product
#     """
#     assert params.shape == vec.shape, "Shape mismatch: params and vec must have the same shape."
    
#     product = 
    
#     return hvp_fun(params)

# if __name__ =="__main__":
#     # 示例字典
#     params_dict = {
#         "w1": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
#         "b1": jnp.array([0.5, 0.6]),
#         "w2": jnp.array([1.5, -2.3, 3.7]),
#     }

#     # 转换为数组
#     params_arr, metadata = params_to_array(params_dict)
#     print("Params Array:", params_arr)
#     print("Metadata:", metadata)

#     # 从数组还原为字典
#     reconstructed_dict = params_to_dict(params_arr, metadata)
#     print("Reconstructed Dict:", reconstructed_dict)

#     # 验证原始字典和还原字典是否一致
#     assert all(jnp.array_equal(params_dict[key], reconstructed_dict[key]) for key in params_dict)
