import jax
import jax.numpy as jnp

def compute_face_areas(vertices, faces):
    """
    Compute area of each triangle face.
    
    Args:
        vertices: (N, 3) float array
        faces: (M, 3) int array
        
    Returns:
        areas: (M,) float array
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # Cross product of two edges
    cross_prod = jnp.cross(v1 - v0, v2 - v0)
    areas = 0.5 * jnp.linalg.norm(cross_prod, axis=1)
    return areas

def compute_vertex_areas(vertices, faces, face_areas=None):
    """
    Compute barycentric vertex areas (1/3 of sum of adjacent face areas).
    
    Args:
        vertices: (N, 3) float array
        faces: (M, 3) int array
        face_areas: (M,) float array (optional)
        
    Returns:
        vertex_areas: (N,) float array
    """
    if face_areas is None:
        face_areas = compute_face_areas(vertices, faces)
        
    N = vertices.shape[0]
    vertex_areas = jnp.zeros(N)
    
    # Scatter add 1/3 of face area to each vertex
    val = face_areas / 3.0
    vertex_areas = vertex_areas.at[faces[:, 0]].add(val)
    vertex_areas = vertex_areas.at[faces[:, 1]].add(val)
    vertex_areas = vertex_areas.at[faces[:, 2]].add(val)
    
    return vertex_areas

def safe_arccos(x):
    """Numerically safe arccos."""
    # Clip to avoid NaNs at -1, 1 due to floating point error
    # Also use a smoother form if needed, but clipping usually suffices
    # However, gradients at -1, 1 are infinite.
    # We clip to 1 - epsilon
    eps = 1e-6
    x_clipped = jnp.clip(x, -1.0 + eps, 1.0 - eps)
    return jnp.arccos(x_clipped)

def compute_face_angles(vertices, faces):
    """
    Compute the 3 angles of each triangle face.
    
    Returns:
        angles: (M, 3) float array, corresponding to vertices (0, 1, 2) of the face.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # Vectorized:
    # Edges from vertices
    # v1-v0, v2-v0
    vec10 = v1 - v0
    vec20 = v2 - v0
    
    vec01 = v0 - v1
    vec21 = v2 - v1
    
    vec02 = v0 - v2
    vec12 = v1 - v2
    
    # Normalize with safe division
    def safe_normalize(v):
        norm = jnp.linalg.norm(v, axis=1, keepdims=True)
        return v / (norm + 1e-12)

    n_vec10 = safe_normalize(vec10)
    n_vec20 = safe_normalize(vec20)
    
    n_vec01 = safe_normalize(vec01)
    n_vec21 = safe_normalize(vec21)
    
    n_vec02 = safe_normalize(vec02)
    n_vec12 = safe_normalize(vec12)
    
    cos0 = jnp.sum(n_vec10 * n_vec20, axis=1)
    cos1 = jnp.sum(n_vec01 * n_vec21, axis=1)
    cos2 = jnp.sum(n_vec02 * n_vec12, axis=1)
    
    angle0 = safe_arccos(cos0)
    angle1 = safe_arccos(cos1)
    angle2 = safe_arccos(cos2)
    
    return jnp.stack([angle0, angle1, angle2], axis=1)

def discrete_mean_curvature_cotangent(vertices, faces, vertex_areas=None):
    """
    Compute discrete mean curvature H using the cotangent Laplacian formula.
    
    H = 0.5 * || Laplace-Beltrami(p) ||
    
    Returns:
        H: (N,) float array, Mean Curvature scalar.
    """
    N = vertices.shape[0]
    
    if vertex_areas is None:
        vertex_areas = compute_vertex_areas(vertices, faces)
    
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # Compute cotangents using robust dot product formula:
    # cot A = (AB . AC) / ||AB x AC||
    # ||AB x AC|| is 2 * Area
    
    # Edges
    e0 = v2 - v1 # Edge opposite v0
    e1 = v0 - v2 # Edge opposite v1
    e2 = v1 - v0 # Edge opposite v2
    
    # Double Area
    cross = jnp.cross(e2, -e1) # (v1-v0) x (v2-v0)
    double_area = jnp.linalg.norm(cross, axis=1)
    denom = double_area + 1e-12
    
    # Cotangents
    # Angle at v0 (between v1-v0 and v2-v0 -> e2 and -e1)
    # dot(e2, -e1)
    cot0 = jnp.sum(e2 * (-e1), axis=1) / denom
    
    # Angle at v1 (between v0-v1 and v2-v1 -> -e2 and e0)
    cot1 = jnp.sum((-e2) * e0, axis=1) / denom
    
    # Angle at v2 (between v0-v2 and v1-v2 -> e1 and -e0)
    cot2 = jnp.sum(e1 * (-e0), axis=1) / denom
    
    # Accumulate forces
    # Edge v1-v2 (opposite v0) gets weight 0.5 * cot0
    f12 = 0.5 * cot0[:, None] * (v2 - v1)
    f21 = -f12
    
    # Edge v2-v0 (opposite v1) gets weight 0.5 * cot1
    f20 = 0.5 * cot1[:, None] * (v0 - v2)
    f02 = -f20
    
    # Edge v0-v1 (opposite v2) gets weight 0.5 * cot2
    f01 = 0.5 * cot2[:, None] * (v1 - v0)
    f10 = -f01
    
    mean_curvature_normal_vector = jnp.zeros_like(vertices)
    
    # Update v0
    mean_curvature_normal_vector = mean_curvature_normal_vector.at[faces[:, 0]].add(f02 + f01)
    # Update v1
    mean_curvature_normal_vector = mean_curvature_normal_vector.at[faces[:, 1]].add(f12 + f10)
    # Update v2
    mean_curvature_normal_vector = mean_curvature_normal_vector.at[faces[:, 2]].add(f20 + f21)
    
    # Laplace-Beltrami
    laplace_beltrami = mean_curvature_normal_vector / (vertex_areas[:, None] + 1e-12)
    
    # H magnitude
    H_magnitude = 0.5 * jnp.linalg.norm(laplace_beltrami, axis=1)
    
    # Sign handling
    # Compute simple vertex normals
    face_normals = jnp.cross(v1 - v0, v2 - v0)
    vertex_normal_vec = jnp.zeros_like(vertices)
    vertex_normal_vec = vertex_normal_vec.at[faces[:, 0]].add(face_normals)
    vertex_normal_vec = vertex_normal_vec.at[faces[:, 1]].add(face_normals)
    vertex_normal_vec = vertex_normal_vec.at[faces[:, 2]].add(face_normals)
    
    vertex_normal_vec = vertex_normal_vec / (jnp.linalg.norm(vertex_normal_vec, axis=1, keepdims=True) + 1e-12)
    
    alignment = jnp.sum(laplace_beltrami * vertex_normal_vec, axis=1)
    H = -0.5 * alignment
    
    return H, H_magnitude

def discrete_gaussian_curvature_angle_deficit(vertices, faces, vertex_areas=None):
    """
    Compute discrete Gaussian curvature K using angle deficit.
    
    K = (2*pi - sum(angles)) / Area
    
    Returns:
        K: (N,) float array
    """
    N = vertices.shape[0]
    
    if vertex_areas is None:
        vertex_areas = compute_vertex_areas(vertices, faces)
    
    angles = compute_face_angles(vertices, faces) # (M, 3)
    
    angle_sum = jnp.zeros(N)
    angle_sum = angle_sum.at[faces[:, 0]].add(angles[:, 0])
    angle_sum = angle_sum.at[faces[:, 1]].add(angles[:, 1])
    angle_sum = angle_sum.at[faces[:, 2]].add(angles[:, 2])
    
    K = (2 * jnp.pi - angle_sum) / (vertex_areas + 1e-12)
    return K

def compute_shape_index(vertices, faces):
    """
    Compute Shape Index (SI).
    
    SI = (2/pi) * arctan( (k1 + k2) / (k2 - k1) )
       = (2/pi) * arctan( H / sqrt(H^2 - K) )   [Koenderink & van Doorn style]
       
    Args:
        vertices: (N, 3) float array
        faces: (M, 3) int array
        
    Returns:
        SI: (N,) float array, range [0, 1] for convex? 
        Range is [-1, 1]. Sphere (cap) is 1.0. 
        Saddle is 0.0. Cylinder is 0.5.
    """
    vertex_areas = compute_vertex_areas(vertices, faces)
    
    H, _ = discrete_mean_curvature_cotangent(vertices, faces, vertex_areas)
    K = discrete_gaussian_curvature_angle_deficit(vertices, faces, vertex_areas)
    
    # Numerical guard for sqrt
    discriminant = H**2 - K
    # Add epsilon to avoid sqrt(0) gradient singularity at umbilic points
    discriminant = jnp.maximum(discriminant, 0.0) + 1e-12
    
    denominator = jnp.sqrt(discriminant)
    
    SI = (2.0 / jnp.pi) * jnp.arctan2(H, denominator)
    
    return SI
