"""Resample Schaefer 32k labels to the FieldTrip 4k source grid."""
from pathlib import Path
from tempfile import TemporaryDirectory
import nibabel as nib
import numpy as np
from scipy.io import savemat
from neuromaps.images import dlabel_to_gifti
from neuromaps.transforms import fslr_to_fslr
import data


def prepare(output, rois=(100, 200, 400)):
    arrays = {}
    with TemporaryDirectory() as temporary:
        for roi in rois:
            atlas = data.ASSETS / 'parcellations' / (
                f'Schaefer2018_{roi}Parcels_7Networks_order.dlabel.nii')
            if not atlas.exists():
                raise FileNotFoundError(atlas)
            left, right = dlabel_to_gifti(str(atlas))
            hemispheres = []
            for hemi, img in (('L', left), ('R', right)):
                source = Path(temporary) / f'{roi}_{hemi}.gii'
                nib.save(img, str(source))
                result = fslr_to_fslr(str(source), '4k', hemi=hemi,
                                      method='nearest')[0]
                hemispheres.append(np.asarray(result.darrays[0].data))
            arrays[f'labels{roi}'] = np.concatenate(hemispheres).astype(np.int32)
            print(f'Atlas ROI {roi}: {len(arrays[f"labels{roi}"])} vertices', flush=True)
    savemat(output, arrays)
