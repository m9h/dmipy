
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import jax.numpy as jnp
from dmipy_jax.cylinder import C1Stick
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.design import oed

def main():
    # --- Setup ---
    print("Setting up simulation...")
    # Simple Stick model
    model = C1Stick()

    # Target parameters: D=1.7e-9 m^2/s (typical axonal), mu aligned with x-axis
    target_params = {
        'lambda_par': 1.7e-9,
        'mu': jnp.array([np.pi/2, 0.0]) # theta=90 (pi/2), phi=0 -> x-axis
    }

    # Initial Acquisition: 30 random gradients
    N_meas = 30
    np.random.seed(42)
    bvals = np.random.uniform(0, 3000e6, N_meas)
    # Random directions
    vecs = np.random.randn(N_meas, 3)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    initial_acq = JaxAcquisition(
        bvalues=jnp.array(bvals),
        gradient_directions=jnp.array(vecs)
    )

    # Wrappers for model func
    def model_func_wrapper(bvals, gradient_directions, **kwargs):
        # C1Stick call
        return model(bvals, gradient_directions, **kwargs)

    # --- Optimization ---
    print("Running OED optimization...")
    # Run for 100 steps
    n_steps = 100
    final_acq, history = oed.optimize_protocol(
        initial_acq=initial_acq,
        model_func=model_func_wrapper,
        target_params=target_params,
        n_steps=n_steps,
        b_max=3000e6, # 3000 s/mm^2 max
        return_history=True
    )

    loss_history = np.array(history['loss'])
    # Convert JAX arrays to numpy and scaling to s/mm^2 immediately
    bvals_history_list = [np.array(b) / 1e6 for b in history['bvalues']]
    # Stack: (n_steps, N_meas)
    bvals_history = np.stack(bvals_history_list)

    # --- Animation ---
    print("Creating animation...")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Axis 1: B-values
    ax1.set_xlim(0, n_steps)
    ax1.set_ylim(0, 3200) 
    ax1.set_xlabel('Iteration Number')
    ax1.set_ylabel('B-value ($s/mm^2$)')
    ax1.set_title('OED Optimization Process')

    # Lines for trajectories
    lines = [ax1.plot([], [], 'b-', alpha=0.3, lw=1)[0] for _ in range(N_meas)]
    
    # Scanner Head (Dots at current iteration)
    scat = ax1.scatter([], [], c='blue', alpha=0.8, s=30, label='Current B-values', zorder=5)

    # Axis 2: Criterion
    ax2 = ax1.twinx()
    ax2.set_ylabel('Design Criterion (Log-Det FIM)')
    
    # Criterion is Negative Loss
    criterion_vals = -loss_history
    c_min, c_max = criterion_vals.min(), criterion_vals.max()
    c_pad = (c_max - c_min) * 0.1 if c_max != c_min else 1.0
    ax2.set_ylim(c_min - c_pad, c_max + c_pad)
    
    criterion_line, = ax2.plot([], [], 'r-', lw=2, label='Design Criterion')

    # Legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    # Manual legend entry for trajectories? No, scatter handles it roughly.
    # Actually 'Current B-values' is in scat label.
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    def init():
        scat.set_offsets(np.empty((0, 2)))
        criterion_line.set_data([], [])
        for line in lines:
            line.set_data([], [])
        return [scat, criterion_line] + lines

    def update(frame):
        # frame goes from 0 to n_steps-1
        # iterations: 0 to frame
        iterations = np.arange(frame + 1)
        
        # Update trails
        current_b_history = bvals_history[:frame+1, :]
        for k in range(N_meas):
            lines[k].set_data(iterations, current_b_history[:, k])
            
        # Update Scanner Head
        # x is just [frame, frame, ...]
        # y is the last row of current_b_history
        x_head = np.full(N_meas, frame)
        y_head = current_b_history[-1, :]
        scat.set_offsets(np.column_stack([x_head, y_head]))
        
        # Update Criterion
        criterion_line.set_data(iterations, criterion_vals[:frame+1])
        
        return [scat, criterion_line] + lines

    ani = animation.FuncAnimation(fig, update, frames=n_steps, init_func=init, blit=True)

    # Save
    save_path = os.path.join(os.path.dirname(__file__), 'oed_convergence.gif')
    print(f"Saving animation to {save_path}...")
    try:
        ani.save(save_path, writer='pillow', fps=15)
        print("Animation saved successfully.")
    except Exception as e:
        print(f"Error saving animation: {e}")
        # Try mp4 if pillow fails (though pillow is standard)
        try:
            mp4_path = save_path.replace('.gif', '.mp4')
            print(f"Trying MP4: {mp4_path}")
            ani.save(mp4_path, writer='ffmpeg', fps=15)
            print("Animation saved as MP4.")
        except Exception as e2:
            print(f"Error saving MP4: {e2}")

if __name__ == "__main__":
    main()
