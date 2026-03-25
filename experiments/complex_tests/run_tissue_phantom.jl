
using Pkg
Pkg.add("HDF5")
using HDF5
using KomaMRI

println("--- Multi-Compartment Tissue Simulation ---")

# 1. Load Data
h5_path = "experiments/complex_tests/tissue_phantom.h5"
if !isfile(h5_path)
    error("File $h5_path not found. Run generate_tissue_phantom.py first.")
end

println("Loading $h5_path...")
fid = h5open(h5_path, "r")
traj = read(fid["trajectories"])
init_pos = read(fid["initial_positions"])
ids = read(fid["compartment_ids"])
dt = read_attribute(fid, "dt")
close(fid)

println("Loaded logic. Compartments: $(unique(ids))")

# 2. Physics Assignment
# IDs: 0 = Intra (Restricted), 1 = Extra (Hindered)
N = length(ids)
T1 = zeros(Float64, N)
T2 = zeros(Float64, N)
rho = ones(Float64, N)

# Assign T1/T2 [s]
# Intra-axonal (Myelinated/Restricted): Lower T2
mask_in = ids .== 0
T1[mask_in] .= 1.0
T2[mask_in] .= 0.080 # 80ms

# Extra-axonal (Hindered): Higher T2
mask_ex = ids .== 1
T1[mask_ex] .= 1.5
T2[mask_ex] .= 0.200 # 200ms

# 3. Motion
x_dims = traj[1, :, :]
y_dims = traj[2, :, :]
z_dims = traj[3, :, :]

x0 = Float64.(init_pos[1, :])
y0 = Float64.(init_pos[2, :])
z0 = Float64.(init_pos[3, :])

sim_dx = Float64.(x_dims .- x0)
sim_dy = Float64.(y_dims .- y0)
sim_dz = Float64.(z_dims .- z0)

path = Path(dx=sim_dx, dy=sim_dy, dz=sim_dz)

N_steps = size(traj, 3) 
t_steps = collect(1:N_steps) * dt
tc = TimeCurve(t_steps, t_steps)

motion = Motion(path, tc)

# 4. Phantom
println("Creating Phantom with Heterogeneous T1/T2...")
obj = Phantom(x=x0, y=y0, z=z0, T1=T1, T2=T2, motion=motion)

# 5. Simulation
println("Defining Sequence (EPI)...")
# We want to vary TE if possible, but for first pass, just run one.
seq = PulseDesigner.EPI_example()
sys = Scanner()

println("Simulating...")
raw = simulate(obj, seq, sys)

println("Simulation finished!")
println("Raw Signal Acquired: $(length(raw.profiles)) profiles.")
