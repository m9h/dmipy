using KomaMRI
using HDF5
using ArgParse

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--seq_path"
            help = "Path to .seq file"
            required = true
        "--output_path"
            help = "Path to output .h5 file"
            required = true
        "--diffusivity"
            help = "Diffusivity (m^2/s)"
            arg_type = Float64
            default = 1e-9
    end
    return parse_args(s)
end

function main()
    args = parse_commandline()
    
    seq_path = args["seq_path"]
    out_path = args["output_path"]
    D_val = args["diffusivity"]

    println("--> KomaWrapper: Reading sequence from $seq_path")
    seq = read_seq(seq_path)

    # Define Phantom
    # Single isochromat at origin
    # T1/T2 infinite to isolate diffusion
    # M0 = 1
    println("--> KomaWrapper: Creating Phantom (D = $D_val m^2/s)")
    
    # KomaMRI Phantom Position: x, y, z
    # KomaMRI uses SI units? 
    # Usually inputs are x,y,z, rho, T1, T2, T2s, Delta_w
    # SimplePhantom(x, y, z, rho, T1, T2, T2s, Delta_w)
    
    # We need to check if SimplePhantom supports diffusivity.
    # Looking at KomaMRI docs (or recall):
    # Phantom struct has D field?
    # Usually: obj = Phantom(...)
    # obj.D = ...
    
    obj = Phantom{Float64}(x=[0.0], y=[0.0], z=[0.0])
    obj.T1 = [1e4] # large
    obj.T2 = [1e4]
    obj.ρ = [1.0]
    
    # Set Diffusivity
    # KomaMRI's Phantom has Dλ1, Dλ2
    println("--> KomaWrapper: Setting D=$D_val")
    obj.Dλ1 = [D_val]
    obj.Dλ2 = [D_val]
    # Dz? 
    # Usually tensor is (D1, D2, D3) or Cylindrical?
    # Wait, Koma Phantom fields list in error: Dλ1, Dλ2, Dθ.
    # It might be 2D/Cylindrical? Or maybe Dλ1, Dλ2 are sufficient for what?
    # KomaMRI v0.7 docs say: "Diffusion properties: Dλ1, Dλ2"
    # Actually, Koma usually assumes 1D or something?
    # Let's check source or assume Isotropic D -> Dlam1=D, Dlam2=D.
    
    println("--> KomaWrapper: Simulating...")
    sys = Scanner()
    
    # Simulate
    # raw = simulate(obj, seq, sys)
    
    # To be safe for v0.9:
    # sim_method = Bloch()
    # Use return_type="mat" to get magnetization trace
    sim_params = Dict{String,Any}("return_type"=>"mat")
    
    raw = simulate(obj, seq, sys; sim_params=sim_params)
    
    # Extract
    if raw isa Array
        println("--> KomaWrapper: Raw output is Array of size $(size(raw))")
        # Usually (N_spins, N_time) Complex? Or (N_time, N_spins)?
        # Or (3, N_time) if matrix of vectors?
        # If it is ComplexF64, it is likely Mxy.
        # If Float64, maybe Mx,My,Mz stacked?
        # Let's assume Mxy if complex.
        sig = raw
    else
        println("--> KomaWrapper: Raw output type: $(typeof(raw))")
        # Fallback
        try
            sig = complex.(raw.x, raw.y)
        catch
            println("Unknown output format")
            rethrow()
        end
    end
    if length(sig) == 1
        t = [sum(seq.DUR)]
    else
        t = range(0, sum(seq.DUR), length=length(sig)) |> collect
    end
    
    println("--> KomaWrapper: Saving to $out_path")
    h5open(out_path, "w") do file
        write(file, "t", t)
        # write(file, "grads", grads) # Gradients are objects, skip. Use JAX gradients.
        # Magnitude
        write(file, "signal_magnitude", abs.(vec(sig)))
        # Complex too if we want
        write(file, "signal_complex", reinterpret(Float64, vec(sig))) 
    end
    
    println("--> KomaWrapper: Done.")
end

main()
