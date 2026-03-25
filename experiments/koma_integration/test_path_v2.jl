
using KomaMRI

t = [0.0, 0.5, 1.0]
x_d = [0.0 0.5e-3 1.0e-3]
y_d = [0.0 0.0 0.0]
z_d = [0.0 0.0 0.0]

try
    println("Creating Path...")
    path = Path(dx=x_d, dy=y_d, dz=z_d)
    println("Path created.")

    println("Creating TimeCurve...")
    # TimeCurve(t, t_unit)
    # t: time points (0 to 1 usually?) or actual time?
    # t_unit: The value of the curve at time t? 
    # If Path defines the shape, maybe TimeCurve defines how fast we traverse it?
    # Or TimeCurve defines the time points corresponding to the columns of dx?
    # Let's assume TimeCurve maps "simulation time" to "path parameter".
    # Note: KomaMRI motion usually interpolates.
    
    # Let's try TimeCurve(t, t) assuming linear time mapping.
    tc = TimeCurve(t, t)
    println("TimeCurve created.")

    println("Creating Motion...")
    motion = Motion(path, tc)
    println("Motion created.")

    obj = Phantom(x=[0.0], y=[0.0], z=[0.0], motion=motion)
    println("Phantom created successfully!")
catch e
    println("Error: $e")
    # precise info
    # println(stacktrace(catch_backtrace()))
end
