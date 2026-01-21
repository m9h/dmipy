
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float
from typing import Optional, Callable

class RationalQuadraticSpline(eqx.Module):
    """
    Rational Quadratic Spline (RQS) Bijector.
    
    Implements the element-wise transformation described in Durkan et al. (2019).
    """
    

def rational_quadratic_spline(inputs, unconstrained_widths, unconstrained_heights, unconstrained_derivatives,
                            inverse=False, left=-3.0, right=3.0, bottom=-3.0, top=3.0,
                            min_bin_width=1e-3, min_bin_height=1e-3, min_derivative=1e-3):
    
    # inputs: (..., D)
    # unconstrained_*: (..., D, K) or (..., D, K+1)
    
    num_bins = unconstrained_widths.shape[-1]
    
    # 1. Parameter Constraints
    widths = jax.nn.softmax(unconstrained_widths, axis=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    
    heights = jax.nn.softmax(unconstrained_heights, axis=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    
    derivatives = min_derivative + jax.nn.softplus(unconstrained_derivatives)
    
    # 2. Knots (Cumulative sum to get boundaries)
    # Pad 0 at the start
    pad_width = ((0,0),) * (widths.ndim - 1) + ((1,0),)
    cum_widths = jnp.pad(jnp.cumsum(widths, axis=-1), pad_width, constant_values=0)
    cum_heights = jnp.pad(jnp.cumsum(heights, axis=-1), pad_width, constant_values=0)
    
    # Force last element to be exactly 1.0 to avoid numerical drift
    cum_widths = cum_widths.at[..., -1].set(1.0)
    cum_heights = cum_heights.at[..., -1].set(1.0)
    
    # Scale to domain
    widths = widths * (right - left)
    heights = heights * (top - bottom)
    cum_widths = left + cum_widths * (right - left)
    cum_heights = bottom + cum_heights * (top - bottom)
    
    # 3. Identify Input Range
    # Inputs outside the range are treated as identity transform
    inside_interval_mask = (inputs >= left) & (inputs <= right)
    outside_interval_mask = ~inside_interval_mask
    
    # Clamp inputs to range for calculation (values outside will be masked out later)
    inputs_clamped = jnp.clip(inputs, left, right)
    
    # 4. Bin Selection
    if inverse:
        # Inverse: Find bin for y (inputs) in y_knots (cum_heights)
        knots = cum_heights
    else:
        # Forward: Find bin for x (inputs) in x_knots (cum_widths)
        knots = cum_widths

    # SearchSorted: Find indices k such that knots[..., k] <= inputs < knots[..., k+1]
    # We vmap over the last dimension (D) and any batch dimensions preceding it?
    # Actually, simpler: flatten batch/D, vmap over that, then reshape?
    # Or just vmap over the last dimension of arrays if they align.
    
    # Let's assume broadcasting works if we define a helper for single feature (D=1 scalar)
    # and vmap it over the D dimension.
    
    # Function to transform a single scalar input given its parameters vector (K,)
    def transform_scalar(x, w, h, cw, ch, d, inv, k_idx):
        # x: scalar
        # w, h: (K,)
        # cw, ch: (K+1,)
        # d: (K+1,)
        # k_idx: index of the bin
        
        # Get bin parameters
        input_w = w[k_idx]
        input_h = h[k_idx]
        input_cw = cw[k_idx]
        input_ch = ch[k_idx]
        
        # Slopes
        delta_k = input_h / input_w
        
        d0 = d[k_idx]
        d1 = d[k_idx+1]
        
        # Forward Transform
        if not inv:
            xi = (x - input_cw) / input_w
            
            s_k = input_h / input_w
            
            numerator = (input_h * (s_k * xi**2 + d0 * xi * (1 - xi)))
            denominator = s_k + (d1 + d0 - 2 * s_k) * xi * (1 - xi)
            
            y = input_ch + numerator / denominator
            
            # Derivative (Jacobian)
            # dy/dx = (s_k^2 (d_{k+1} xi^2 + 2 s_k xi (1-xi) + d_k (1-xi)^2)) / denominator^2
            
            numerator_grad = s_k**2 * (d1 * xi**2 + 2 * s_k * xi * (1 - xi) + d0 * (1 - xi)**2)
            log_det = jnp.log(numerator_grad) - 2 * jnp.log(denominator)
            
            return y, log_det
        else:
            # Inverse Transform
            # y is input (x argument in function signature, but semantically y)
            y = x 
            
            # y corresponds to equation:
            # y = input_ch + numerator / denominator
            # let y_hat = y - input_ch
            # y_hat (s_k + (d1 + d0 - 2 s_k) xi (1 - xi)) = input_h (s_k xi^2 + d0 xi (1 - xi))
            
            # This is a quadratic in xi: A xi^2 + B xi + C = 0
            
            y_hat = y - input_ch
            s_k = input_h / input_w
            
            # Term coeffs
            # Left side: LHS = y_hat * (s_k + (d1+d0-2sk)(xi - xi^2))
            # LHS = y_hat * s_k + y_hat*(d1+d0-2sk)*xi - y_hat*(d1+d0-2sk)*xi^2
            
            # Right Side: RHS = h * (sk xi^2 + d0 xi - d0 xi^2)
            # RHS = h * (sk - d0) xi^2 + h * d0 * xi
            
            # A xi^2 + B xi + C = 0
            # Grouping xi^2: RHS - LHS
            # Coeff xi^2: h(sk - d0) - (-y_hat(d1+d0-2sk)) = h(sk-d0) + y_hat(d1+d0-2sk)
            # Coeff xi: h*d0 - y_hat(d1+d0-2sk)
            # Constant: - y_hat * s_k
            
            term_common = d1 + d0 - 2 * s_k
            
            a = input_h * (s_k - d0) + y_hat * term_common
            b = input_h * d0 - y_hat * term_common
            c = -s_k * y_hat
            
            # Solve quadratic (-b + sqrt(b^2 - 4ac)) / 2a
            # Since 0 <= xi <= 1, we generally take the stable root.
            # Using textbook formula
            
            discriminant = b**2 - 4 * a * c
            root = (-b + jnp.sqrt(discriminant)) / (2 * a)
            
            # Handle a=0 case (linear) separately or rely on stability?
            # RQS usually guarantees curvature so a!=0 unless d0=d1=sk (linear bin)
            # If linear, y_hat / h * w. 
            
            xi_sol = root
            
            # Reconstruct x
            x_rec = input_cw + xi_sol * input_w
            
            # Calculate log_det (inverse derivative)
            # dy/dx at x_rec
            xi = xi_sol
            numerator_grad = s_k**2 * (d1 * xi**2 + 2 * s_k * xi * (1 - xi) + d0 * (1 - xi)**2)
            denominator = s_k + term_common * xi * (1 - xi)
            
            dy_dx = numerator_grad / denominator**2
            
            # log|dx/dy| = - log|dy/dx|
            log_det = -jnp.log(dy_dx)
            
            return x_rec, log_det
    # Vectorized bin application
    # We vmap over the last dimension D
    def apply_per_dim(inputs_d, widths_d, heights_d, cum_widths_d, cum_heights_d, derivatives_d):
        # inputs_d: scalar
        # others: (K,) or (K+1,)
        
        if not inverse:
            # Find bin
            idx = jnp.searchsorted(cum_widths_d, inputs_d, side='right') - 1
            idx = jnp.clip(idx, 0, num_bins - 1)
            # Transform
            return transform_scalar(inputs_d, widths_d, heights_d, cum_widths_d, cum_heights_d, derivatives_d, inverse, idx)
        else:
            # Find bin (inverse)
            idx = jnp.searchsorted(cum_heights_d, inputs_d, side='right') - 1
            idx = jnp.clip(idx, 0, num_bins - 1)
            return transform_scalar(inputs_d, widths_d, heights_d, cum_widths_d, cum_heights_d, derivatives_d, inverse, idx)

    # vmap over dimension D
    final_outputs, final_log_det = jax.vmap(apply_per_dim)(
        inputs_clamped, widths, heights, cum_widths, cum_heights, derivatives
    )
    
    # Log determinant for identity transform is 0
    outside_log_det = jnp.asarray(0.0)
    
    # Apply mask (identity outside range)
    final_outputs = jnp.where(inside_interval_mask, final_outputs, inputs)
    final_log_det = jnp.where(inside_interval_mask, final_log_det, outside_log_det)
    
    return final_outputs, final_log_det


class RationalQuadraticSpline(eqx.Module):
    num_bins: int
    range_min: float
    range_max: float
    
    def __init__(self, num_bins: int = 8, range_min: float = -3.0, range_max: float = 3.0):
        self.num_bins = num_bins
        self.range_min = range_min
        self.range_max = range_max
        
    def __call__(self, x: Array, params: Array, inverse: bool = False):
        """
        x: (D,)
        params: (D, 3*K + 1)
        """
        K = self.num_bins
        # params layout: [widths(K), heights(K), derivatives(K+1)]
        # Split params
        c_w = params[..., :K]
        c_h = params[..., K:2*K]
        c_d = params[..., 2*K:]
        
        return rational_quadratic_spline(
            x, c_w, c_h, c_d, 
            inverse=inverse, 
            left=self.range_min, right=self.range_max, 
            bottom=self.range_min, top=self.range_max
        )



class CouplingLayer(eqx.Module):
    dimension: int
    conditioner: eqx.nn.MLP
    bijector_fn: Callable = eqx.field(static=True)
    
    def __init__(self, key, dimension, n_context, hidden_size=64, num_bins=8):
        self.dimension = dimension
        # Conditioner: Inputs [split_x, context] -> Output [params for RQS]
        # Split dimension: we transform second half based on first half.
        d_in = (dimension // 2) + n_context
        
        # RQS params: 3*K + 1 per dimension.
        # We transform (dimension - dimension//2) dims.
        n_transformed = dimension - (dimension // 2)
        n_params = n_transformed * (3 * num_bins + 1)
        
        self.conditioner = eqx.nn.MLP(
            in_size=d_in,
            out_size=n_params,
            width_size=hidden_size,
            depth=2,
            key=key
        )
        
        # We store the bijector class/config
        self.bijector_fn = lambda x, p, inv: RationalQuadraticSpline(num_bins=num_bins)(x, p, inverse=inv)

    def __call__(self, x: Array, context: Array, inverse: bool = False):
        # x: (D,)
        # context: (C,)
        
        d_split = self.dimension // 2
        
        x_id, x_change = x[:d_split], x[d_split:]
        
        # Conditioner input
        cond_in = jnp.concatenate([x_id, context])
        params = self.conditioner(cond_in)
        
        # Reshape params for RQS: (D_change, 3*K+1)
        params = params.reshape(x_change.shape[0], -1)
        
        # Transform
        y_change, log_det = self.bijector_fn(x_change, params, inverse)
        
        y = jnp.concatenate([x_id, y_change])
        
        return y, log_det

class FlowNetwork(eqx.Module):
    layers: list
    base_dist: Callable = eqx.field(static=True)
    
    def __init__(self, key, n_layers=3, n_dim=2, n_context=10):
        keys = jax.random.split(key, n_layers)
        self.layers = []
        for i in range(n_layers):
            # Alternate masks or just swap indices?
            # Simple coupling: always split. We need permutation layers to mix.
            # Simplified: Random Permutation fixed?
            # For this prototype, we just implement the Coupling. 
            # In real flows, we'd add eqx.nn.Permutation(indices).
            
            layer = CouplingLayer(keys[i], n_dim, n_context)
            self.layers.append(layer)
            # Add permutation (flip)
            # We can bake it into Coupling or loop logic
            
        self.base_dist = lambda x: jax.scipy.stats.norm.logpdf(x).sum()
            
    def log_prob(self, theta, context):
        # theta -> z
        # log_p(theta) = log_p(z) + sum(log_det)
        log_det_sum = 0
        x = theta
        
        # Normalize/Transform (Forward flow x->z)
        # Note: CouplingLayer checks 'inverse' flag.
        # If 'inverse=False' maps x -> z (Normalization)
        
        for layer in self.layers:
            # We also need to permute. 
            # Placeholder: Reverse elements between layers
            x = x[::-1]
            x, ld = layer(x, context, inverse=False)
            log_det_sum += jnp.sum(ld)
            
        # Base distribution N(0, I)
        log_prob_z = jax.scipy.stats.norm.logpdf(x).sum()
        
        return log_prob_z + log_det_sum

    def sample(self, key, context, n_samples=1):
        # Sample z ~ N(0, I)
        # Transform z -> x (Inverse flow)
        
        z = jax.random.normal(key, (n_samples, self.layers[0].dimension))
        
        # Inverse pass
        # Must apply layers in reverse order
        
        def sample_single(z_i):
            x = z_i
            for layer in reversed(self.layers):
                x, _ = layer(x, context, inverse=True)
                # Undo permutation (reverse of reverse is reverse)
                x = x[::-1]
            return x

        return jax.vmap(sample_single)(z)


