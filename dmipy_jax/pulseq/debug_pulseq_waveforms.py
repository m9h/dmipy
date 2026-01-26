import pypulseq as pp
import numpy as np

system = pp.Opts(max_grad=30, grad_unit='mT/m', max_slew=100, slew_unit='T/m/s')
seq = pp.Sequence(system)
gx = pp.make_trapezoid(channel='x', flat_area=100, flat_time=10e-3, system=system)
seq.add_block(gx)

try:
    ret = seq.waveforms_and_times()
    grads = ret[0]
    print(f"Grads type: {type(grads)}")
    print(f"Grads length: {len(grads)}")
    for i, g in enumerate(grads):
        print(f"Grad {i} shape: {g.shape}")
        print(f"Grad {i} content:\n{g}")
except Exception as e:
    print(e)
