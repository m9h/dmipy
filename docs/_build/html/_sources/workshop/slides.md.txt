---
marp: true
theme: default
class: lead
paginate: true
backgroundColor: #fff
backgroundImage: url('https://marp.app/assets/hero-background.svg')
---

# scanner-loop: Designing MRI Protocols with JAX

## Active Learning for Microstructure Imaging

<!-- 
Speaker Notes:
Welcome everyone. Today we're talking about "scanner-loop", a project to make MRI scanners "smart" using JAX.
Instead of just taking pictures, we want the scanner to *decide* what pictures to take.
-->

---

# The Problem: Why are protocols static?

*   **One Size Fits None**: We use the same protocol for a stroke patient as we do for a healthy volunteer.
*   **Inefficient**: We waste time acquiring redundant data (e.g., too many low b-values).
*   **Hardware Ignorant**: Gradients have limits ($G_{max}$, Slew), but generic protocols don't fully exploit them.

> **Goal**: A protocol that adapts *live* to the patient in the bore.

<!--
Speaker Notes:
Currently, you sit down at the console and pick "Protocol A". 
It runs for 10 minutes.
But if the patient has a large lesion, maybe we need more high-b-value data to characterize it.
If they are moving, maybe we need faster acquisition.
Static protocols leave information on the table.
-->

---

# The Math: Fisher Information

We want to maximize the information content of our measurements.

$$
\mathcal{I}(\theta) = J(\theta)^T \Sigma^{-1} J(\theta)
$$

Where:
*   $J(\theta) = \frac{\partial S}{\partial \theta}$ is the **Sensitivity** (Jacobian).
*   $\Sigma$ is the noise covariance.

**Objective**: Maximize the determinant (D-optimality).
$$
\theta^* = \text{argmax} \det(\mathcal{I}(\theta))
$$

<!--
Speaker Notes:
How do we define "best"? We use the Fisher Information Matrix (FIM).
Intuitively, FIM tells us how "curved" the signal landscape is. High curvature = high sensitivity to parameter changes.
We want to pick acquisition parameters (b-values, directions) that maximize this matrix.
Specifically, we maximize the determinant (D-Optimality), which minimizes the volume of the confidence ellipsoid.
-->

---

# The Code: Auto-Diff with JAX

Calculating $J = \frac{\partial S}{\partial \theta}$ by hand is painful. JAX makes it trivial.

```python
import jax
import jax.numpy as jnp

def get_sensitivity(model_params, acquisition_params):
    # Forward pass: predict signal
    # Backward pass: compute gradients
    J = jax.jacfwd(model.predict, argnums=0)(
        model_params, acquisition_params
    )
    return J

def fisher_info(theta, bvecs):
    J = get_sensitivity(theta, bvecs)
    return J.T @ J  # Simple matrix multiplication!
```

<!--
Speaker Notes:
This is the magic. 
In the past, you'd spend weeks deriving derivatives for a biophysical model.
With JAX, `jacfwd` gives us the exact Jacobian automatically.
This allows us to optimize *any* model—NODDI, spherical mean, etc.—without rewriting the math.
-->

---

![bg right 90%](../visuals/oed_convergence.gif)

# The Result

**Visualizing Convergence**

*   **Dots**: b-shells being selected.
*   **Curve**: The design criterion (Determinant of FIM) increasing over time.
*   **Outcome**: The algorithm automatically "discovers" that high b-values are better for measuring axon diameter.

<!--
Speaker Notes:
Here is the optimizer in action.
Watch how the b-values (dots) migrate.
It starts random, but quickly snaps to specific "shells" that are optimal for the specific tissue parameters we simulated.
The line shows the information content rising.
-->

---

# Live Learning: The "Scout Scan"

```mermaid
graph LR
    A[Scout Scan\n(1 min)] --> B[Fit Model\n(JAX GPU)]
    B --> C[Optimize Next Shell\n(OED)]
    C --> D[Acquire Data]
    D --> B
```

1.  **Scout**: Quick, low-res scan to guess parameters.
2.  **Fit**: Estimate $\theta$ (e.g., axon density) on GPU.
3.  **Optimize**: Calculate $J(\theta)$ and find best next $b, \vec{v}$.
4.  **Acquire**: Send instructions to scanner.

<!--
Speaker Notes:
This is the workflow we are building.
It's a feedback loop.
1. Fast scout scan.
2. Fit the tissue parameters in seconds on a GPU.
3. Use those parameters to calculate the local FIM.
4. Tell the scanner: "The best place to look next is b=3000 in this direction."
-->
