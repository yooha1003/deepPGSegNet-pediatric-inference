"""Measure aligned masks listed in a case,prediction,reference CSV."""
import argparse
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, distance_transform_edt
from metrics import summarize


def measure(prediction, reference):
    if prediction.shape != reference.shape or not np.allclose(prediction.affine,reference.affine,atol=1e-4):
        raise ValueError('Prediction and reference grids differ; align before evaluation.')
    raw_p=prediction.get_fdata();raw_r=reference.get_fdata()
    if raw_p.ndim!=3 or not np.isfinite(raw_p).all() or not np.isfinite(raw_r).all():
        raise ValueError('Masks must be finite three-dimensional arrays.')
    if not np.isin(raw_p,[0,1]).all() or not np.isin(raw_r,[0,1]).all():
        raise ValueError('Expected binary masks containing only 0 and 1.')
    p=raw_p>0;r=raw_r>0
    a=int(p.sum());b=int(r.sum())
    if b==0:raise ValueError('The reference gland is empty; volume-relative error is undefined.')
    spacing=np.linalg.norm(reference.affine[:3,:3],axis=0)
    directions=reference.affine[:3,:3]/spacing
    if not np.allclose(directions.T@directions,np.eye(3),atol=1e-4):
        raise ValueError('Sheared grid: resample masks onto an orthogonal physical grid before HD95 evaluation.')
    unit=abs(np.linalg.det(reference.affine[:3,:3]))
    hd=np.nan
    if a:
        ps=p & ~binary_erosion(p);rs=r & ~binary_erosion(r)
        distances=np.r_[distance_transform_edt(~ps,sampling=spacing)[rs],
                        distance_transform_edt(~rs,sampling=spacing)[ps]]
        hd=float(np.percentile(distances,95))
    return dict(dice=2*np.count_nonzero(p&r)/(a+b),pred_voxels=a,ref_voxels=b,
                pred_mm3=a*unit,ref_mm3=b*unit,signed_error_pct=100*(a-b)/b,
                absolute_error_pct=100*abs(a-b)/b,absolute_error_mm3=abs(a-b)*unit,hd95=hd)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    manifest=pd.read_csv(args.manifest)
    if not {'case','prediction','reference'}.issubset(manifest.columns):
        parser.error('Manifest must contain case,prediction,reference columns.')
    if len(manifest)<3 or manifest.case.duplicated().any():
        parser.error('At least three uniquely identified participants are required for agreement summaries.')
    rows=[]
    for row in manifest.itertuples():
        pred=nib.load(args.manifest.parent/str(row.prediction))
        ref=nib.load(args.manifest.parent/str(row.reference))
        rows.append({'case':row.case,**measure(pred,ref)})
    frame=pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    frame.to_csv(args.output_dir/'cases.csv',index=False)
    (args.output_dir/'summary.json').write_text(json.dumps(summarize(frame),indent=2)+'\n')


if __name__=='__main__':main()
