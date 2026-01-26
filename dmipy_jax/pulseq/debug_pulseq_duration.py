import pypulseq as pp
import numpy as np

system = pp.Opts(max_grad=30, grad_unit='mT/m', max_slew=100, slew_unit='T/m/s')
seq = pp.Sequence(system)
gx = pp.make_trapezoid(channel='x', flat_area=100, flat_time=10e-3, system=system)
seq.add_block(gx)

try:
    dur = seq.duration()
    print(f"Duration type: {type(dur)}")
    print(f"Duration content: {dur}")
except Exception as e:
    print(e)
