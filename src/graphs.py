"""Generate and verify the EDR and local graph controls from the pial meshes.

The local graph sums positive cotangent weights across parcel boundaries in
each hemisphere. Interhemispheric pairs that are mutual nearest centroids among
all parcels receive the median positive within-hemisphere edge weight. Both
control Laplacians are scaled to the structural Laplacian's Frobenius norm.

By default verify the six saved controls. --output DIR instead writes newly
generated controls to that directory, including assets/ for a full rebuild.
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist

from data import ASSETS, ROIS


def laplacian(weights):
    weights = np.maximum(weights, 0)
    weights = (weights + weights.T) / 2
    np.fill_diagonal(weights, 0)
    return np.diag(weights.sum(axis=1)) - weights


def mesh_weights(hemi):
    """Positive off-diagonal weights of the negative cotangent stiffness."""
    mesh = nib.load(ASSETS / 'plot_aux_data' / f'fs_LR.32k.{hemi}.pial.surf.gii')
    coords, faces = (array.data for array in mesh.darrays)
    # Preserve the native float32 geometry used by the original script.
    points = coords[faces]
    rows, columns, values = [], [], []
    for corner in range(3):
        a = points[:, (corner + 1) % 3] - points[:, corner]
        b = points[:, (corner + 2) % 3] - points[:, corner]
        area_twice = np.sqrt((np.cross(a, b) ** 2).sum(axis=1))
        if np.any(area_twice == 0):
            raise ValueError(f'Degenerate triangle in {hemi} pial mesh')
        cotangent = (a * b).sum(axis=1) / area_twice / 2
        left, right = faces[:, (corner + 1) % 3], faces[:, (corner + 2) % 3]
        rows.extend((left, right))
        columns.extend((right, left))
        values.extend((cotangent, cotangent))
    weights = coo_matrix((np.concatenate(values),
                          (np.concatenate(rows), np.concatenate(columns))),
                         shape=(len(coords), len(coords))).tocsr()
    # Sum adjacent-triangle contributions before clipping, as in (-K).maximum(0).
    weights.data = np.maximum(weights.data, 0)
    weights.eliminate_zeros()
    return coords, weights


def controls(roi, meshes):
    labels = nib.load(ASSETS / 'parcellations' /
                      f'Schaefer2018_{roi}Parcels_7Networks_order.dlabel.nii')
    labels = labels.get_fdata().squeeze().astype(int)
    coords = np.vstack([mesh[0] for mesh in meshes])
    if labels.shape != (len(coords),):
        raise ValueError('Parcellation labels must follow both complete pial meshes')
    centroids = np.zeros((roi, 3))
    for parcel in range(1, roi + 1):
        vertices = coords[labels == parcel]
        if not len(vertices):
            raise ValueError(f'Empty parcel {parcel} at {roi} parcels')
        centroids[parcel - 1] = vertices.mean(axis=0)
    scale = np.linalg.norm(laplacian(np.load(ASSETS / f'W_Sch{roi}.npy')))
    distances = cdist(centroids, centroids)
    np.fill_diagonal(distances, np.inf)
    edr = np.exp(-0.12 * distances)
    edr *= scale / np.linalg.norm(laplacian(edr))

    local = np.zeros((roi, roi))
    n_left = len(meshes[0][0])
    half = roi // 2
    for hemi, (_, weights) in enumerate(meshes):
        hemi_labels = labels[:n_left] if hemi == 0 else labels[n_left:]
        parcels = range(1, half + 1) if hemi == 0 else range(half + 1, roi + 1)
        vertices = {parcel: np.flatnonzero(hemi_labels == parcel) for parcel in parcels}
        # Keep the original parcel-by-parcel float32 sparse summation order.
        for p in parcels:
            for q in parcels:
                if q <= p:
                    continue
                weight = weights[np.ix_(vertices[p], vertices[q])].sum()
                if weight > 0:
                    local[p - 1, q - 1] = local[q - 1, p - 1] = weight
    count, component = connected_components(local)
    if (count != 2 or len(set(component[:half])) != 1
            or len(set(component[half:])) != 1):
        raise ValueError(f'{roi}: expected one connected surface graph per hemisphere')
    nearest = distances.argmin(axis=1)
    bridges = [(i, int(j)) for i, j in enumerate(nearest[:half])
               if j >= half and nearest[j] == i]
    if not bridges:
        raise ValueError(f'{roi}: no mutual-nearest interhemispheric pairs')
    upper = np.triu(local, 1)
    bridge_weight = np.median(upper[upper > 0])
    for i, j in bridges:
        local[i, j] = local[j, i] = bridge_weight
    local *= scale / np.linalg.norm(laplacian(local))
    for weights in (edr, local):
        if (not np.isfinite(weights).all() or np.any(weights < 0)
                or not np.array_equal(weights, weights.T) or np.any(np.diag(weights))
                or connected_components(weights)[0] != 1
                or not np.isclose(np.linalg.norm(laplacian(weights)), scale, rtol=1e-12)):
            raise ValueError(f'{roi}: invalid graph control')
    if np.count_nonzero(local[:half, half:]) != len(bridges):
        raise ValueError(f'{roi}: unexpected interhemispheric edge support')
    return {f'W_EDR_Sch{roi}.npy': edr, f'W_Geo_Sch{roi}.npy': local}


def check(name, regenerated):
    stored = np.load(ASSETS / name)
    if (regenerated.shape != stored.shape
            or not np.array_equal(regenerated > 0, stored > 0)
            or not np.allclose(regenerated, stored, rtol=1e-6, atol=0)):
        raise ValueError(f'{name}: regenerated control differs from the saved graph')
    error = np.abs(regenerated - stored)
    positive = stored > 0
    relative = np.max(error[positive] / stored[positive])
    edges = np.count_nonzero(np.triu(stored, 1))
    print(f'{name}: {edges} edges; max absolute error {error.max():.3g}; '
          f'max relative error {relative:.3g}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        help='generate controls here instead of comparing with saved assets')
    args = parser.parse_args()
    meshes = [mesh_weights(hemi) for hemi in ('L', 'R')]
    generated = {}
    for roi in ROIS:
        for name, weights in controls(roi, meshes).items():
            if args.output is None:
                check(name, weights)
            generated[name] = weights
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        for name, weights in generated.items():
            np.save(args.output / name, weights)
        print(f'Wrote six graph controls to {args.output.resolve()}')


if __name__ == '__main__':
    main()
