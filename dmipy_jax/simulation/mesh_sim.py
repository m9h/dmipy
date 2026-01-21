
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array, Int
from typing import Tuple, Optional

class Mesh(eqx.Module):
    """
    Differentiable Mesh container.
    """
    vertices: Float[Array, "N_vert 3"]
    faces: Int[Array, "N_face 3"]
    
    def __init__(self, vertices: Float[Array, "N_vert 3"], faces: Int[Array, "N_face 3"]):
        self.vertices = jnp.asarray(vertices)
        if faces.shape[1] != 3:
            raise ValueError("Only triangular meshes are supported.")
        self.faces = jnp.asarray(faces, dtype=jnp.int32)

def compute_triangle_areas(vertices: Float[Array, "N 3"], faces: Int[Array, "M 3"]) -> Float[Array, "M"]:
    """
    Compute area of each triangle face.
    Area = 0.5 * | cross(v1-v0, v2-v0) |
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    cross_prod = jnp.cross(v1 - v0, v2 - v0)
    areas = 0.5 * jnp.linalg.norm(cross_prod, axis=1)
    return areas

def compute_cotangents(vertices: Float[Array, "N 3"], faces: Int[Array, "M 3"]) -> Float[Array, "M 3"]:
    """
    Compute cotangents of angles for each triangle.
    Used for Stiffness matrix construction.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # Edge vectors
    e0 = v2 - v1 # Opposite to v0
    e1 = v0 - v2 # Opposite to v1
    e2 = v1 - v0 # Opposite to v2
    
    # Cotangent formula: cot(A) = (b.c) / |bx c|
    # Here we need cot(angle at v0), cot(angle at v1), cot(angle at v2)
    
    # Angle at v0 is between -e2 and e1 ? No, vectors out from v0 are (v1-v0) and (v2-v0).
    # Let u = v1-v0, v = v2-v0.
    # cot(theta) = (u . v) / |u x v|
    
    # For v0: u = v1-v0, v = v2-v0
    u0 = v1 - v0
    v0_vec = v2 - v0
    cot0 = jnp.sum(u0 * v0_vec, axis=1) / jnp.linalg.norm(jnp.cross(u0, v0_vec), axis=1)
    
    # For v1: u = v2-v1, v = v0-v1
    u1 = v2 - v1
    v1_vec = v0 - v1
    cot1 = jnp.sum(u1 * v1_vec, axis=1) / jnp.linalg.norm(jnp.cross(u1, v1_vec), axis=1)
    
    # For v2: u = v0-v2, v = v1-v2
    u2 = v0 - v2
    v2_vec = v1 - v2
    cot2 = jnp.sum(u2 * v2_vec, axis=1) / jnp.linalg.norm(jnp.cross(u2, v2_vec), axis=1)
    
    return jnp.stack([cot0, cot1, cot2], axis=1)

def construct_fem_matrices(mesh: Mesh) -> Tuple[Float[Array, "N N"], Float[Array, "N N"]]:
    """
    Constructs the Stiffness (S) and Mass (M) matrices for the Laplace-Beltrami operator.
    
    Returns dense matrices for now (N is small ~1000). 
    For larger N, we need sparse support (jax.experimental.sparse or BCOO).
    ReMiDi uses spectral decomposition, so dense is fine if we truncate or if N is manageable.
    
    Args:
        mesh: Mesh object
        
    Returns:
        S: Stiffness matrix (N_vert, N_vert)
        M: Mass matrix (N_vert, N_vert)
    """
    N = mesh.vertices.shape[0]
    
    # --- Stiffness Matrix S ---
    # S_ij = integral (grad phi_i . grad phi_j) dA
    # For linear elements, S_local is computed from cotangents.
    # Diagonal entries: Sum of cotangents for edges connected to i.
    # Off-diagonal: -0.5 * (cot alpha + cot beta) where alpha, beta are angles opposite to edge ij.
    
    # 1. Compute cotangents per face
    # shape: (N_faces, 3) -> cot0, cot1, cot2 correspond to angles at v0, v1, v2
    cots = compute_cotangents(mesh.vertices, mesh.faces)
    
    # 2. Scatter into global matrix
    # S_local for a triangle (indices i, j, k):
    # s_ii = 0.5 * (cot_j + cot_k)  <-- wait, standard formula implies edge weights.
    # Let's use the edge-based assembly.
    
    #   Nodes in face: v0, v1, v2
    #   Edge e0 (v1-v2) is opposite v0. Contribution 0.5*cot0 to S_12 and S_21?
    #   Yes. S_ij for edge connecting i,j is -0.5 * sum(cot(opp_angle)).
    #   Standard FEM assembly:
    #   Local S:
    #   [ cot1+cot2  -cot2      -cot1 ]
    #   [ -cot2       cot0+cot2 -cot0 ] * 0.5
    #   [ -cot1      -cot0       cot0+cot1 ]
    
    S_dense = jnp.zeros((N, N))
    M_dense = jnp.zeros((N, N))
    
    # Vectorized assembly
    # Indices for scatter
    # faces is (F, 3)
    ii = mesh.faces[:, [0, 1, 2, 0, 1, 2, 0, 1, 2]] # rows
    jj = mesh.faces[:, [0, 1, 2, 1, 2, 0, 2, 0, 1]] # cols? No.
    
    # Let's explicitly define standard local matrix mapping
    # 0,0 ; 1,1 ; 2,2 (Diagonals)
    # 0,1 ; 1,2 ; 2,0 (Off-diagonals)
    # 1,0 ; 2,1 ; 0,2 (Symmetric off-diagonals)
    
    # Local S values (F, 3, 3)
    # diag 0: 0.5*(cot1 + cot2)
    # off 0,1 (-cot2?): Edge 0-1 is e2 (opposite v2). So angle is at v2. -0.5*cot2.
    # Correct.
    
    c0 = cots[:, 0]
    c1 = cots[:, 1]
    c2 = cots[:, 2]
    
    # Construct local 3x3 matrices
    # S_local[0,0] = 0.5*(c1+c2)
    # S_local[0,1] = -0.5*c2
    # S_local[0,2] = -0.5*c1
    
    # S_local[1,0] = -0.5*c2
    # S_local[1,1] = 0.5*(c0+c2)
    # S_local[1,2] = -0.5*c0
    
    # S_local[2,0] = -0.5*c1
    # S_local[2,1] = -0.5*c0
    # S_local[2,2] = 0.5*(c0+c1)
    
    # Flatten everything to scatter
    
    rows = []
    cols = []
    vals_S = []
    vals_M = []
    
    # --- Stiffness ---
    # Diagonals
    rows.append(mesh.faces[:, 0]); cols.append(mesh.faces[:, 0]); vals_S.append(0.5*(c1+c2))
    rows.append(mesh.faces[:, 1]); cols.append(mesh.faces[:, 1]); vals_S.append(0.5*(c0+c2))
    rows.append(mesh.faces[:, 2]); cols.append(mesh.faces[:, 2]); vals_S.append(0.5*(c0+c1))
    
    # Off-diags
    rows.append(mesh.faces[:, 0]); cols.append(mesh.faces[:, 1]); vals_S.append(-0.5*c2)
    rows.append(mesh.faces[:, 1]); cols.append(mesh.faces[:, 0]); vals_S.append(-0.5*c2)
    
    rows.append(mesh.faces[:, 1]); cols.append(mesh.faces[:, 2]); vals_S.append(-0.5*c0)
    rows.append(mesh.faces[:, 2]); cols.append(mesh.faces[:, 1]); vals_S.append(-0.5*c0)
    
    rows.append(mesh.faces[:, 2]); cols.append(mesh.faces[:, 0]); vals_S.append(-0.5*c1)
    rows.append(mesh.faces[:, 0]); cols.append(mesh.faces[:, 2]); vals_S.append(-0.5*c1)
    
    # --- Mass Matrix M ---
    # Lumped Mass? Or Consistent?
    # ReMiDi paper probably uses consistent mass matrix for FEM.
    # M_ij = integral (phi_i * phi_j) dA
    # Area/12 for off-diag, Area/6 for diag.
    
    areas = compute_triangle_areas(mesh.vertices, mesh.faces) # (F,)
    
    # Diagonals: Area / 6
    vals_M.append(areas / 6.0); # 0,0 appended to rows/cols lists... wait, need to sync
    
    # This separation is messy. Let's do S and M separately.
    
    # Scatter S
    all_rows_S = jnp.concatenate(rows)
    all_cols_S = jnp.concatenate(cols)
    all_vals_S = jnp.concatenate(vals_S)
    
    S_dense = S_dense.at[all_rows_S, all_cols_S].add(all_vals_S)
    
    # Scatter M
    rows_M = []
    cols_M = []
    vals_M_list = []
    
    # Diag: Area/6
    rows_M.append(mesh.faces[:, 0]); cols_M.append(mesh.faces[:, 0]); vals_M_list.append(areas/6.0)
    rows_M.append(mesh.faces[:, 1]); cols_M.append(mesh.faces[:, 1]); vals_M_list.append(areas/6.0)
    rows_M.append(mesh.faces[:, 2]); cols_M.append(mesh.faces[:, 2]); vals_M_list.append(areas/6.0)
    
    # Off-diag: Area/12
    off_val = areas / 12.0
    rows_M.append(mesh.faces[:, 0]); cols_M.append(mesh.faces[:, 1]); vals_M_list.append(off_val)
    rows_M.append(mesh.faces[:, 1]); cols_M.append(mesh.faces[:, 0]); vals_M_list.append(off_val)
    
    rows_M.append(mesh.faces[:, 1]); cols_M.append(mesh.faces[:, 2]); vals_M_list.append(off_val)
    rows_M.append(mesh.faces[:, 2]); cols_M.append(mesh.faces[:, 1]); vals_M_list.append(off_val)
    
    rows_M.append(mesh.faces[:, 2]); cols_M.append(mesh.faces[:, 0]); vals_M_list.append(off_val)
    rows_M.append(mesh.faces[:, 0]); cols_M.append(mesh.faces[:, 2]); vals_M_list.append(off_val)
    
    all_rows_M = jnp.concatenate(rows_M)
    all_cols_M = jnp.concatenate(cols_M)
    all_vals_M = jnp.concatenate(vals_M_list)
    
    M_dense = M_dense.at[all_rows_M, all_cols_M].add(all_vals_M)
    
    return S_dense, M_dense

def construct_position_matrices(mesh: Mesh) -> Float[Array, "3 N N"]:
    """
    Constructs the weighted Mass matrices for position (Mx, My, Mz).
    (Mk)_ij = integral (phi_i * x_k * phi_j) dA
    
    Used for the gradient term in Bloch-Torrey: i * gamma * G.x
    """
    N = mesh.vertices.shape[0]
    Mx = jnp.zeros((N, N))
    My = jnp.zeros((N, N))
    Mz = jnp.zeros((N, N))
    
    # We essentially need the "Triple Product" integral.
    # On a triangle T, x is linear: x(xi, eta) = sum N_m(xi, eta) * x_m
    # Int (N_i N_j x) = sum_m x_m * Int (N_i N_j N_m)
    # The integral I_abc = Int (N_a N_b N_c) over a triangle of area A is:
    # (a! b! c!) / (a+b+c+2)! * 2A
    # Here indices differ.
    
    # Case 1: i=j=m (All same) -> Int N_i^3 = 6/120 * 2A = A/10 ??
    # Formula: Int lambda1^a lambda2^b lambda3^c dA = 2A * a!b!c! / (a+b+c+2)!
    # 
    # For cubic terms N_i N_j N_k:
    # 1. i=j=k: Int L_i^3 = 2A * 6 / 120 = A/10
    # 2. i=j!=k: Int L_i^2 L_k = 2A * 2 / 120 = A/30
    # 3. i!=j!=k: Int L_i L_j L_k = 2A * 1 / 120 = A/60
    
    # This is getting complex to vectorize purely with indices.
    # Alternative: Use quadrature? Or just hardcode the 10 coefficients for local matrix.
    # Local matrix M_x_local (3x3) for triangle T with nodes v0, v1, v2 having x-coords x0, x1, x2.
    # (M_x)_ab = x0 * Int(N_a N_b N_0) + x1 * Int(N_a N_b N_1) + x2 * Int(N_a N_b N_2)
    
    # Let's define the local interaction tensors.
    # Coeffs times Area.
    # C_aaa = 1/10
    # C_aab = 1/30
    # C_abc = 1/60 (all distinct)
    
    areas = compute_triangle_areas(mesh.vertices, mesh.faces)
    
    # Get coordinates of vertices for each face
    # (F, 3, 3) -> [face_idx, node_in_face, xyz]
    face_coords = mesh.vertices[mesh.faces] 
    
    # We need to assemble 3 matrices: Mx, My, Mz
    # Let's do it per-coordinate dim
    
    matrices = []
    
    for dim in range(3):
        M_dim = jnp.zeros((N, N))
        coords = face_coords[:, :, dim] # (F, 3) -> x0, x1, x2 for each face
        
        # Local 3x3 matrix for each face
        # Entry (0,0): x0*C_000 + x1*C_001 + x2*C_002
        # = x0(A/10) + x1(A/30) + x2(A/30)
        # = A/30 * (3*x0 + x1 + x2)
        
        # Entry (0,1): x0*C_010 + x1*C_011 + x2*C_012
        # = x0(A/30) + x1(A/30) + x2(A/60)
        # = A/60 * (2*x0 + 2*x1 + x2)
        
        # Symmetry applies.
        
        A_30 = areas / 30.0
        A_60 = areas / 60.0
        
        # Diagonals
        # M_00
        val_00 = A_30 * (3*coords[:, 0] + coords[:, 1] + coords[:, 2])
        val_11 = A_30 * (coords[:, 0] + 3*coords[:, 1] + coords[:, 2])
        val_22 = A_30 * (coords[:, 0] + coords[:, 1] + 3*coords[:, 2])
        
        # Off-diagonals
        # M_01 = M_10
        val_01 = A_60 * (2*coords[:, 0] + 2*coords[:, 1] + coords[:, 2])
        # M_12 = M_21
        val_12 = A_60 * (coords[:, 0] + 2*coords[:, 1] + 2*coords[:, 2])
        # M_02 = M_20
        val_02 = A_60 * (2*coords[:, 0] + coords[:, 1] + 2*coords[:, 2])
        
        rows = []
        cols = []
        vals = []
        
        # 0,0
        rows.append(mesh.faces[:, 0]); cols.append(mesh.faces[:, 0]); vals.append(val_00)
        # 1,1
        rows.append(mesh.faces[:, 1]); cols.append(mesh.faces[:, 1]); vals.append(val_11)
        # 2,2
        rows.append(mesh.faces[:, 2]); cols.append(mesh.faces[:, 2]); vals.append(val_22)
        
        # 0,1 & 1,0
        rows.append(mesh.faces[:, 0]); cols.append(mesh.faces[:, 1]); vals.append(val_01)
        rows.append(mesh.faces[:, 1]); cols.append(mesh.faces[:, 0]); vals.append(val_01)
        
        # 1,2 & 2,1
        rows.append(mesh.faces[:, 1]); cols.append(mesh.faces[:, 2]); vals.append(val_12)
        rows.append(mesh.faces[:, 2]); cols.append(mesh.faces[:, 1]); vals.append(val_12)
        
        # 0,2 & 2,0
        rows.append(mesh.faces[:, 0]); cols.append(mesh.faces[:, 2]); vals.append(val_02)
        rows.append(mesh.faces[:, 2]); cols.append(mesh.faces[:, 0]); vals.append(val_02)
        
        all_rows = jnp.concatenate(rows)
        all_cols = jnp.concatenate(cols)
        all_vals = jnp.concatenate(vals)
        
        M_dim = M_dim.at[all_rows, all_cols].add(all_vals)
        matrices.append(M_dim)
        
    return jnp.stack(matrices, axis=0) # (3, N, N)

class MatrixFormalismSimulator(eqx.Module):
    """
    Simulates diffusion signal using FEM Matrix Formalism (ReMiDi / SpinDoctor flavor).
    Uses Eigendecomposition (ROM) for fast simulation.
    """
    mesh: Mesh
    diffusivity: Float[Array, ""]
    eig_vals: Float[Array, "K"]
    eig_vecs: Float[Array, "N K"]
    Mx_proj: Float[Array, "3 K K"]
    
    def __init__(self, mesh: Mesh, diffusivity: float, K: int = 50):
        self.mesh = mesh
        self.diffusivity = jnp.asarray(diffusivity)
        
        # Precompute Matrices
        S, M = construct_fem_matrices(mesh)
        
        # Generalized Eigendecomposition: S v = lambda M v
        # Since M is positive definite (Mass matrix), we can use eigh on transformed problem.
        # But jax.scipy.linalg.eigh supports a 'b' argument for generalized!
        # Top K smallest eigenvalues (lowest frequency modes) are needed.
        # 'subset_by_index' is not supported in JIT? 
        # For small meshes (<2000 nodes), full eigh is fast enough (~0.1s).
        
        # Generalized Eigendecomposition: S v = lambda M v
        # JAX eigh(a, b) isn't implemented. Transform to standard:
        # M = L L^T
        # L^-1 S L^-T y = lambda y, with v = L^-T y
        
        # 1. Cholesky of M (symmetric, pos-def)
        # Add jitter for stability
        M_stable = M + jnp.eye(M.shape[0]) * 1e-12
        L = jax.scipy.linalg.cholesky(M_stable, lower=True)
        
        # 2. Compute S_hat = L^-1 S L^-T
        # L^-1 S
        # Solve L X = S -> X = L^-1 S
        X = jax.scipy.linalg.solve_triangular(L, S, lower=True)
        # S_hat = X L^-T = X (L^T)^-1
        # Solve L y = X^T -> y = L^-1 X^T = (X L^-T)^T = S_hat^T = S_hat
        S_hat = jax.scipy.linalg.solve_triangular(L, X.T, lower=True).T
        
        # Force symmetry just in case
        S_hat = 0.5 * (S_hat + S_hat.T)
        
        # 3. Standard Eigendecomposition
        vals, y_vecs = jax.scipy.linalg.eigh(S_hat)
        
        # 4. Recover v = L^-T y
        # L^T v = y
        vecs = jax.scipy.linalg.solve_triangular(L.T, y_vecs, lower=False)
        
        # Take first K
        self.eig_vals = vals[:K]
        self.eig_vecs = vecs[:, :K]
        
        # Project Position Matrices
        # M_coords = construct_position_matrices(mesh) # (3, N, N)
        # X_proj_kl = v_k^T M_coords v_l
        # But wait, orthogonality is v_i^T M v_j = delta_ij
        # The equation for coefficients c(t):
        # M \dot{c} = -D S c + i g M_x c
        # Project with V^T:
        # V^T M V \dot{a} = -D V^T S V a + i g V^T M_x V a
        # I \dot{a} = -D Lambda a + i g X_proj a
        
        M_pos = construct_position_matrices(mesh)
        
        def project_matrix(Min):
            # V^T @ Min @ V
            return self.eig_vecs.T @ Min @ self.eig_vecs
            
        self.Mx_proj = jax.vmap(project_matrix)(M_pos)
        
    def __call__(self, G_amp: float, delta: float, Delta: float, dt: float = 1e-3) -> Float[Array, ""]:
        """
        Simulate PGSE signal using ROM Matrix Exponential.
        """
        K = self.eig_vals.shape[0]
        gamma = 267.513e6 # rad/s/T (Proton)
        
        # Damping matrix (Diagonal)
        # R = -D * Lambda
        R = -self.diffusivity * jnp.diag(self.eig_vals)
        
        # Excitation matrix (Coupling)
        # B = i * gamma * G * X_proj
        # G is vector (Gx, Gy, Gz). Assume gradient along x for now or input direction.
        # Let's assume passed G is magnitude, along X-axis. 
        # TODO: Support arbitrary directions.
        G_vec = jnp.array([1.0, 0.0, 0.0]) * G_amp
        
        # Interaction term: sum (G_k * X_proj_k)
        Interaction = jnp.einsum('k,kij->ij', G_vec, self.Mx_proj)
        
        # Total System Matrix A = R + i * gamma * Interaction
        A_grad = R + 1j * gamma * Interaction
        A_nograd = R # When G=0
        
        # Propagators
        # 1. Pulse 1 (duration delta)
        E1 = jax.scipy.linalg.expm(A_grad * delta)
        
        # 2. Gap (duration Delta - delta)
        E2 = jax.scipy.linalg.expm(A_nograd * (Delta - delta))
        
        # 3. Pulse 2 (duration delta, gradient -G)
        # Refocusing: Effective gradient is -G.
        A_grad_neg = R - 1j * gamma * Interaction
        E3 = jax.scipy.linalg.expm(A_grad_neg * delta)
        
        # Initial state: Uniform distribution?
        # Initial coefficients a(0).
        # rho(0) = 1/Volume (Uniform density).
        # c(0) such that Psi(r) = 1/V.
        # c = V^T M rho_vec?
        # No, c = V^T M * (1_vec / Volume)? 
        # Actually, the 0-th eigenmode of Neumann Laplacian is constant 1/sqrt(V).
        # So if we project uniform density, it should align with 0-th mode.
        # a_0 = [1, 0, 0, ...] if phi_0 is normalized const.
        # Let's assume 0-th is constant.
        
        a0 = jnp.zeros(K, dtype=jnp.complex64)
        a0 = a0.at[0].set(1.0) # Assuming mode 0 is the constant mode
        
        # If eigenmodes are sorted, val[0] ~ 0.
        
        # Evolve
        # a_final = E3 @ E2 @ E1 @ a0
        a_post_p1 = E1 @ a0
        a_post_gap = E2 @ a_post_p1
        a_final = E3 @ a_post_gap
        
        # Signal is Integral of magnetization.
        # S = Integral sum c_k phi_k dV
        #   = sum c_k (Integral phi_k dV)
        #   = sum c_k (phi_k, 1)_L2
        #   = c_0 * sqrt(V) ? (If phi_0 = 1/sqrt(V))
        #   If a0=[1,0..], then S(0) should be 1.
        #   We just need to project back to "Mean".
        #   Since 1_constant is V_0 * sqrt(Vol), 
        #   Projection of 1 is just picking the first component (scaled).
        #   Or simpler: Signal = a_final[0] * S(0)
        
        # Let's return magnitude relative to initial
        signal = jnp.abs(a_final[0])
        
        return signal
