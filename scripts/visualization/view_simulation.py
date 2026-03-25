
from paraview.simple import *

#### 1. Load Anatomy (Context) ####
print("Loading Anatomy: /home/mhough/dev/dmipy/kf_anatomy_centered.xdmf")
anat = OpenDataFile('/home/mhough/dev/dmipy/kf_anatomy_centered.xdmf')
RenameSource('Anatomy_Context', anat)

# Scalp Layer (Threshold == 1)
t_head = Threshold(Input=anat)
t_head.Scalars = ['CELLS', 'tissue']
t_head.ThresholdMethod = 'Between'
t_head.LowerThreshold = 1.0
t_head.UpperThreshold = 8.0 # Show everything for now? Or just Scalp
t_head.LowerThreshold = 1.0
t_head.UpperThreshold = 1.0 # Just Scalp
RenameSource('Scalp_Context', t_head)

d_head = Show(t_head)
d_head.Representation = 'Surface'
d_head.Opacity = 0.1
d_head.DiffuseColor = [0.9, 0.9, 0.9] # Light Gray

# Skull Layer (Threshold == 2)
t_skull = Threshold(Input=anat)
t_skull.Scalars = ['CELLS', 'tissue']
t_skull.ThresholdMethod = 'Between'
t_skull.LowerThreshold = 2.0
t_skull.UpperThreshold = 2.0
RenameSource('Skull_Context', t_skull)

d_skull = Show(t_skull)
d_skull.Representation = 'Surface'
d_skull.Opacity = 0.15
d_skull.DiffuseColor = [0.95, 0.85, 0.7] # Bone Beige

# Brain Layer (Threshold >= 4)
t_brain = Threshold(Input=anat)
t_brain.Scalars = ['CELLS', 'tissue']
t_brain.ThresholdMethod = 'Between'
t_brain.LowerThreshold = 4.0
t_brain.UpperThreshold = 8.0
RenameSource('Brain_Context', t_brain)

d_brain = Show(t_brain)
d_brain.Representation = 'Surface'
d_brain.Opacity = 0.3
d_brain.DiffuseColor = [1.0, 150/255.0, 180/255.0] # Pink

#### 2. Load Optodes (Hardware) ####
print("Loading Optodes: /home/mhough/dev/dmipy/kf_array_hardware.xdmf")
optodes = OpenDataFile('/home/mhough/dev/dmipy/kf_array_hardware.xdmf')
RenameSource('KernelFlow_Array', optodes)

# Glyph as Spheres
glyph = Glyph(Input=optodes)
glyph.GlyphType = 'Sphere'
glyph.ScaleFactor = 4.0 
glyph.GlyphMode = 'All Points'
RenameSource('Array_Spheres', glyph)

d_opt = Show(glyph)
# Color by 'type' (1=Source, 2=Detector)
ColorBy(d_opt, ('POINTS', 'type'))
lut_opt = GetColorTransferFunction('type')
lut_opt.RescaleTransferFunction(1.0, 2.0)
lut_opt.RGBPoints = [1.0, 1.0, 0.0, 0.0,  # 1=Red (Source)
                     2.0, 0.0, 1.0, 0.0]  # 2=Green (Detectors)
d_opt.SetScalarBarVisibility(d_opt, False)

#### 3. Load Pulse (MMC Action) ####
print("Loading Pulse: /home/mhough/dev/dmipy/kf_mmc_centered.xdmf")
pulse = OpenDataFile('/home/mhough/dev/dmipy/kf_mmc_centered.xdmf')
RenameSource('MMC_Fluence', pulse)

# Use ResampleToImage for Volume Rendering stability
resample = ResampleToImage(Input=pulse)
resample.SamplingDimensions = [150, 150, 150]
resample.UseInputBounds = 1
RenameSource('Resampled_Fluence', resample)

d_pulse = Show(resample)
d_pulse.Representation = 'Volume'

# Color by MMC Fluence
ColorBy(d_pulse, ('POINTS', 'log_fluence_mmc'))
lut = GetColorTransferFunction('log_fluence_mmc')

# Hardcoded Opacity for 'log_fluence_mmc'
# Assume range [-10.0, 0.0]
lut.RescaleTransferFunction(-8.0, 0.0)
lut.ApplyPreset('Black-Body Radiation', True)

pwf = GetOpacityTransferFunction('log_fluence_mmc')
# Strict transparency for background
pwf.Points = [-100.0, 0.0, 0.5, 0.0, 
              -8.0, 0.0, 0.5, 0.0, 
               0.0, 1.0, 0.5, 0.0]

#### 4. Final Polish ####
Render()
ResetCamera()
