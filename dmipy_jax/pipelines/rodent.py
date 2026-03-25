import os
import logging
import numpy as np
import nibabel as nib

logger = logging.getLogger(__name__)

# Try to import SimpleITK
try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False
    logger.warning("SimpleITK not found. Rodent pipeline features will be limited.")

class RodentPreprocessing:
    """
    Robust preprocessing pipeline for rodent MRI data.
    Based on the Oxford/Lerch pipeline (CAMRI/CFMM legacy).
    
    Key Features:
    - N4 Bias Field Correction (Essential for 9.4T data)
    - Rigid Registration of DWI to Anatomical (T2w)
    - Atlas Registration (T2w -> Allen CCFv3)
    """
    
    def __init__(self, output_dir: str):
        if not HAS_SITK:
            raise ImportError("SimpleITK is required for RodentPreprocessing.")
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def run_n4_bias_correction(self, input_path: str, suffix: str = "_N4") -> str:
        """
        Applies N4 Bias Field Correction to an image.
        Returns path to corrected image.
        """
        logger.info(f"Running N4 Bias Correction on {input_path}")
        
        # Read image
        input_img = sitk.ReadImage(input_path, sitk.sitkFloat32)
        
        # Create mask (Otsu typically works well for high-contrast T2w)
        mask_img = sitk.OtsuThreshold(input_img, 0, 1, 200)
        
        # N4
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([50, 50, 50, 50])
        output_img = corrector.Execute(input_img, mask_img)
        
        # Save
        filename = os.path.basename(input_path).replace(".nii", f"{suffix}.nii")
        output_path = os.path.join(self.output_dir, filename)
        sitk.WriteImage(output_img, output_path)
        
        return output_path

    def register_dwi_to_anat(self, dwi_mean_b0: str, anat_path: str) -> str:
        """
        Rigidly registers DWI (Mean b0) to Anatomical (T2w) space.
        Returns path to the transform file (.tfm).
        """
        logger.info(f"Registering DWI {dwi_mean_b0} to Anat {anat_path}")
        
        fixed = sitk.ReadImage(anat_path, sitk.sitkFloat32)
        moving = sitk.ReadImage(dwi_mean_b0, sitk.sitkFloat32)
        
        # Initialize (Center of Geometry)
        init_tx = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
        
        # Registration
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(0.1)
        
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, estimateLearningRate=reg.Once)
        reg.SetOptimizerScalesFromPhysicalShift()
        
        reg.SetInitialTransform(init_tx, inPlace=False)
        
        final_tx = reg.Execute(fixed, moving)
        
        logger.info(f"Final metric value: {reg.GetMetricValue()}")
        logger.info(f"Optimizer stop condition: {reg.GetOptimizerStopConditionDescription()}")
        
        # Save Transform
        tx_path = os.path.join(self.output_dir, "dwi_to_anat_rigid.tfm")
        sitk.WriteTransform(final_tx, tx_path)
        
        return tx_path

    def apply_transform(self, moving_path: str, reference_path: str, transform_path: str) -> str:
        """
        Applies a transform to an image.
        """
        moving = sitk.ReadImage(moving_path)
        reference = sitk.ReadImage(reference_path)
        tx = sitk.ReadTransform(transform_path)
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(tx)
        
        out = resampler.Execute(moving)
        
        filename = os.path.basename(moving_path).replace(".nii", "_reg.nii")
        out_path = os.path.join(self.output_dir, filename)
        sitk.WriteImage(out, out_path)
        
        return out_path

    def register_to_allen(self, anat_path: str, allen_template_path: str) -> dict:
        """
        Registers Anatomical (T2w) to Allen CCFv3 Template.
        Currently implements Affine registration.
        
        Args:
            anat_path: Path to subject's T2w scan.
            allen_template_path: Path to Allen CCFv3 template.
            
        Returns:
            dict: {
                'transform_path': Path to Subject->Atlas transform,
                'inverse_transform_path': Path to Atlas->Subject transform,
                'warped_anat': Path to T2w in Allen Space
            }
        """
        logger.info(f"Registering Anat {anat_path} to Allen {allen_template_path}")
        
        fixed = sitk.ReadImage(allen_template_path, sitk.sitkFloat32)
        moving = sitk.ReadImage(anat_path, sitk.sitkFloat32)
        
        # Initial Alignment
        init_tx = sitk.CenteredTransformInitializer(fixed, moving, sitk.AffineTransform(3), sitk.CenteredTransformInitializerFilter.GEOMETRY)
        
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(0.1)
        
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
        reg.SetOptimizerScalesFromPhysicalShift()
        
        reg.SetInitialTransform(init_tx, inPlace=False)
        
        final_tx = reg.Execute(fixed, moving)
        
        # Save Subject->Atlas
        tx_path = os.path.join(self.output_dir, "anat_to_allen_affine.tfm")
        sitk.WriteTransform(final_tx, tx_path)
        
        # Invert for Atlas->Subject (needed for label propagation)
        inv_tx = final_tx.GetInverse()
        inv_tx_path = os.path.join(self.output_dir, "allen_to_anat_affine.tfm")
        sitk.WriteTransform(inv_tx, inv_tx_path)
        
        # Resample for QC
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(final_tx)
        warped = resampler.Execute(moving)
        
        warped_path = os.path.join(self.output_dir, "anat_in_allen_space.nii.gz")
        sitk.WriteImage(warped, warped_path)
        
        return {
            "transform_path": tx_path,
            "inverse_transform_path": inv_tx_path,
            "warped_anat": warped_path
        }
        
    def warp_labels_to_subject(self, label_path: str, inverse_transform_path: str, reference_path: str) -> str:
        """
        Warps Allen CCF Labels (Annotation) to Subject Space.
        Uses Nearest Neighbor interpolation to preserve integer labels.
        
        Args:
            label_path: Path to Allen Annotation volume.
            inverse_transform_path: Path to Atlas->Subject transform.
            reference_path: Path to Subject image (e.g. T2w or DWI b0) defining the grid.
            
        Returns:
            Path to annotation in subject space.
        """
        logger.info(f"Warping Labels {label_path} to Subject Space")
        
        labels = sitk.ReadImage(label_path)
        reference = sitk.ReadImage(reference_path)
        tx = sitk.ReadTransform(inverse_transform_path)
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor) # CRITICAL for labels
        resampler.SetTransform(tx)
        
        warped_labels = resampler.Execute(labels)
        
        out_path = os.path.join(self.output_dir, "allen_labels_in_subject_space.nii.gz")
        sitk.WriteImage(warped_labels, out_path)
        
        return out_path
