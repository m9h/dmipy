import pypulseq as pp
import numpy as np

system = pp.Opts(max_grad=30, grad_unit='mT/m', max_slew=100, slew_unit='T/m/s')
seq = pp.Sequence(system)
gx = pp.make_trapezoid(channel='x', flat_area=100, flat_time=10e-3, system=system)
seq.add_block(gx)

try:
    ret = seq.waveforms_and_times()
    print(f"Return type: {type(ret)}")
    print(f"Length: {len(ret)}")
    print(f"Content: {ret}")
except Exception as e:
    print(e)
