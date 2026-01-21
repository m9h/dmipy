
import jax
import jax.numpy as jnp
import optimistix as optx

def main():
    # Simple Least Squares problem
    # y = a*x + b
    # data: y_true
    
    x = jnp.linspace(0, 1, 10)
    a_true, b_true = 2.0, 1.0
    y_true = a_true * x + b_true
    
    def residual(params, args):
        a, b = params
        pred = a * x + b
        return pred - y_true
        
    solver = optx.LevenbergMarquardt(rtol=1e-5, atol=1e-5)
    init_params = jnp.array([0.0, 0.0])
    
    sol = optx.least_squares(
        fn=residual,
        solver=solver,
        y0=init_params,
        args=(),
        max_steps=100,
        throw=False
    )
    
    print("Solution keys:", sol.__dict__.keys())
    print("Stats:", sol.stats)
    print("Result:", sol.result)
    print("Value:", sol.value)

if __name__ == "__main__":
    main()
