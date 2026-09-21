"""Run the fixed-ROI pediatric pituitary model on a prepared T1 NIfTI."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import nibabel as nib
import numpy as np
import torch
from model import UNet3D

DEFAULT_CENTER = (80, 117, 90)


def load_model(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    state = {k.removeprefix('module.').replace('basic_SingleConv', 'basic_module.SingleConv'): v
             for k, v in state.items()}
    model = UNet3D()
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def prepare_crop(image, center, size=32):
    if image.ndim != 3 or not np.isfinite(image).all():
        raise ValueError('Input must be a finite three-dimensional image.')
    center = np.asarray(center, dtype=int)
    shape = np.asarray(image.shape)
    if np.any(center < 0) or np.any(center >= shape):
        raise ValueError('ROI center is outside the input image.')
    mean, std = np.mean(image), np.std(image)
    if std <= 0:
        raise ValueError('Input intensity variance is zero.')
    normalized = (image - mean) / std
    start = center - size // 2
    lo, hi = np.maximum(start, 0), np.minimum(start + size, shape)
    source = tuple(slice(int(a), int(b)) for a,b in zip(lo,hi))
    target = tuple(slice(int(a), int(b)) for a,b in zip(lo-start,hi-start))
    crop = np.zeros((size,)*3, dtype=np.float32)
    crop[target] = normalized[source]
    return crop, source, target


def predict(image, models, center=DEFAULT_CENTER, device='cpu'):
    crop, source, target = prepare_crop(image, center)
    tensor = torch.from_numpy(crop[None,None]).to(device)
    with torch.inference_mode():
        probability = sum(torch.sigmoid(model(tensor)) for model in models) / len(models)
    small = probability.cpu().numpy()[0,0]
    full = np.zeros(image.shape, dtype=np.float32)
    full[source] = small[target]
    mask = (full >= 0.5).astype(np.uint8)
    boundary = np.concatenate([small[0].ravel(), small[-1].ravel(),
                               small[:,0].ravel(), small[:,-1].ravel(),
                               small[:,:,0].ravel(), small[:,:,-1].ravel()])
    return mask, full, bool(np.any(boundary >= 0.5))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--weights', nargs='+', type=Path,
                        default=[Path(__file__).parent/'weights'/'pediatric_mean.pt'])
    parser.add_argument('--center', nargs=3, type=int, default=DEFAULT_CENTER,
                        metavar=('I','J','K'), help='Voxel indices in the stored input array.')
    parser.add_argument('--device', default='cpu', choices=['cpu','cuda'])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--prepared-study-grid', action='store_true', required=True,
                        help='Acknowledge that the image has the study-compatible spatial grid; see README.')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use --device cpu')
    start = time.perf_counter()
    nii = nib.load(args.input)
    if len(nii.shape) != 3 or not np.allclose(nii.header.get_zooms()[:3], 1, atol=1e-3):
        parser.error('Expected a 3D image with 1 mm isotropic voxels; no resampling is performed.')
    if nib.aff2axcodes(nii.affine) != ('R', 'A', 'S'):
        parser.error('Expected RAS-oriented stored data; automatic reorientation would change the fixed ROI.')
    image = nii.get_fdata(dtype=np.float64)
    models = [load_model(p, args.device) for p in args.weights]
    mask, probability, touches_boundary = predict(image, models, args.center, args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in [('mask.nii.gz', mask), ('probability.nii.gz', probability)]:
        header = nii.header.copy()
        header.set_data_dtype(data.dtype)
        result = nib.Nifti1Image(data, nii.affine, header)
        result.set_qform(nii.get_qform(), int(nii.header['qform_code']))
        result.set_sform(nii.get_sform(), int(nii.header['sform_code']))
        nib.save(result, args.output_dir/name)
    flags = []
    if not mask.any(): flags.append('empty_mask')
    if touches_boundary: flags.append('mask_touches_roi_boundary')
    report = {
        'volume_mm3': float(mask.sum()*abs(np.linalg.det(nii.affine[:3,:3]))),
        'foreground_voxels': int(mask.sum()), 'center_voxel': list(args.center),
        'input_shape': list(nii.shape), 'axis_codes': list(nib.aff2axcodes(nii.affine)),
        'threshold': 0.5, 'crop_size': 32, 'ensemble': 'probability_mean' if len(models)>1 else 'single_checkpoint',
        'qc_flags': flags, 'visual_review_required': True,
        'qc_note': 'These flags do not detect all localization failures or establish anatomical validity.',
        'weights_sha256': [hashlib.sha256(p.read_bytes()).hexdigest() for p in args.weights],
        'device': args.device, 'elapsed_seconds': time.perf_counter()-start,
    }
    (args.output_dir/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
