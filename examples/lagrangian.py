import jax_cgd as jcgd
import jax
import jax.numpy as jnp

if __name__ == "__main__":
    # # 定义目标函数和约束函数
    # def f(x):
    #     return (x - 2)**2  # 原始目标函数

    # def g(x):
    #     return x - 1  # 约束函数

    # def lagrangian(x_dict, y_dict):
    #     """
    #     Compute the Lagrangian: L(x, y) = f(x) + y * g(x).

    #     Args:
    #         x_dict: Dictionary containing x parameters.
    #         y_dict: Dictionary containing y parameters (Lagrange multipliers).

    #     Returns:
    #         Scalar value of the Lagrangian.
    #     """
    #     x = x_dict["x"]
    #     y = y_dict["y"]
    #     return jnp.sum(f(x) + y * g(x))

    # # 初始化参数
    # x_params_dict = {"x": jnp.array([2.0])}  # x 的初始值
    # y_params_dict = {"y": jnp.array([1.0])}  # y 的初始值（Lagrange multiplier）

    # # 转换为数组和元数据
    # x_params, x_metadata = params_to_array(x_params_dict)
    # y_params, y_metadata = params_to_array(y_params_dict)

    # x_metadata = jax.device_get(x_metadata)
    # y_metadata = jax.device_get(y_metadata)

    # @jax.jit
    # def lg_f(x, y):
    #     return lagrangian(params_to_dict(x, x_metadata), params_to_dict(y, y_metadata))

    # 定义目标函数和约束函数
    def f(x):
        """
        目标函数: 处理多维输入 x.
        f(x) = (x1 - 3)^2 + (x2 - 5)^2
        """
        return jnp.sum((x - jnp.array([3.0, 5.0])) ** 2)

    def g(x):
        """
        约束函数: 多维线性约束.
        g(x) = Ax - b
        A = [[1, -1], [-1, -1]], b = [1, -3]
        """
        A = jnp.array([[1.0, -1.0], [-1.0, -1.0]])
        b = jnp.array([1.0, -3.0])
        return A @ x - b

    def lagrangian(x_dict, y_dict):
        """
        L(x, y) = f(x) + y^T * g(x)

        Args:
            x_dict: 包含 x 参数的字典.
            y_dict: 包含 y 参数 (Lagrange multipliers) 的字典.

        Returns:
            标量值 L(x, y).
        """
        x = x_dict["x"]
        y = y_dict["y"]
        return jnp.sum(f(x) - jnp.dot(y ** 2, g(x)))

    # 初始化多维参数
    x_params_dict = {"x": jnp.array([3.0, 5.0])}  # x 的初始值
    y_params_dict = {"y": jnp.array([5., 5.])}  # y 的初始值 (Lagrange multipliers)

    # 转换参数为数组和元数据
    x_params, x_metadata = jcgd.utils.params_to_array(x_params_dict)
    y_params, y_metadata = jcgd.utils.params_to_array(y_params_dict)

    # 确保元数据在 JIT 中静态化
    x_metadata = jax.device_get(x_metadata)
    y_metadata = jax.device_get(y_metadata)

    # 定义 Lagrangian 的 JIT 加速函数
    @jax.jit
    def lg_f(x, y):
        return lagrangian(jcgd.utils.params_to_dict(x, x_metadata), jcgd.utils.params_to_dict(y, y_metadata))


    # 初始化优化器状态
    vx = jnp.zeros_like(x_params)
    vy = jnp.zeros_like(y_params)
    timestep = 0
    lr = 0.5
    beta = 0.9
    eps = 1e-3
    prev_sol = None

    solver = jcgd.solvers.GMRES()
    # optimizer = ACGD(x_params_dict, y_params_dict, lagrangian, lr, beta, eps, solver)
    optimizer = jcgd.BCGD(x_params_dict, y_params_dict, lagrangian, lr, solver)
    num_steps = 10000
    for step in range(num_steps):
        # 调用 step_func
        # x_params, y_params, vx, vy, prev_sol, x_params_dict, y_params_dict = step_func(
        #     x_params, y_params, x_metadata, y_metadata,
        #     vx, vy, timestep, lr, beta, eps, solver, prev_sol, lg_f
        # )
        optimizer.step()
        x_params, y_params, x_params_dict, y_params_dict = optimizer.get_infos()
        # 投影到非负空间（确保 y >= 0）
        # y_params = jnp.maximum(y_params, 0.0)

        # 打印进度
        print(f"Step {step + 1}:")
        print(f"  x_params: {x_params_dict}")
        print(f"  y_params: {y_params_dict}")
        print(f"  Lagrangian value: {lagrangian(x_params_dict, y_params_dict):.4f}")
        print(f"Constraint value: {g(x_params_dict['x'])}\n")

    # 打印最终结果
    print("\nOptimization completed.")
    print(f"Final x_params: {x_params_dict}")
    print(f"Final y_params: {y_params_dict}")
    print(f"Final Lagrangian value: {lagrangian(x_params_dict, y_params_dict):.4f}")

   
