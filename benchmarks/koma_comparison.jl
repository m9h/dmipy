using KomaMRI
using BenchmarkTools
using Printf

function benchmark_bloch()
    println("--- KomaMRI Benchmark ---")
    
    # 1. Define Phantom (Single spin at center)
    # x, y, z, T1, T2, M0 (rho)
    # Phantom uses Greek letter ρ for spin density.
    obj = Phantom(x=[0.0], y=[0.0], z=[0.0], T1=[1000e-3], T2=[100e-3], ρ=[1.0])
    
    # 2. Define Sequence (FID: 90 deg RF + Delay)
    seq = Sequence()
    # RF 90 degrees (pi/2) roughly 1ms duration
    # args: magnitude, duration
    # KomaMRI RF constructor: RF(amp, duration) typically in Tesla? 
    # Or RF(angle, duration)?
    # Usually: RF(amplitude, duration, frequency, phase)
    # Let's assume standard Pulseq-like: using `RF(γ*B1*T, T)`?
    # KomaMRI helper: RF(pi/2, 1e-3) -> creates an RF event with flip angle pi/2.
    seq += RF(90 * 1im, 1e-3) # 90 degrees, 1ms. KomaMRI uses degrees sometimes? Or complex?
    # Docs say RF is complex struct.
    # Let's look at a simpler way.
    # Use a macro or helper if available.
    # Actually, KomaMRI usually constructs RF via `RF(amp, T)`. Amp in T.
    # If we want 90 deg, Amp * gamma * T = pi/2.
    # Amp = pi/2 / (gamma * T).
    amp = (pi/2) / (42.577e6 * 2pi * 1e-3) # T? No gamma is 267.5e6 rad/s/T.
    # KomaMRI defines `RF` object. 
    # Let's try to just add `Delay(10e-3)` and see if adding `rho` fixes bounds error.
    # If not, I will try adding a simple Gradient.
    seq += Delay(50e-3)
    
    # 3. Simulate
    # scan parameters
    sys = Scanner() # Default scanner
    
    # Compile / Warmup
    println("Warming up...")
    # simulate(obj, seq, sys) uses default Bloch dictionary
    KomaMRICore.simulate(obj, seq, sys)
    
    println("Benchmarking...")
    # @btime returns the minimum time
    t = @benchmark KomaMRICore.simulate($obj, $seq, $sys)
    
    
    display(t)
    
    # Attempt to extract mean time in ms
    t_median_ns = median(t.times)
    t_median_ms = t_median_ns / 1e6
    @printf("\nMedian Time: %.4f ms\n", t_median_ms)
end

# Check dependencies
try
    benchmark_bloch()
catch e
    println("Error running benchmark: ", e)
    println("Ensure KomaMRI is installed: import Pkg; Pkg.add(\"KomaMRI\")")
end
