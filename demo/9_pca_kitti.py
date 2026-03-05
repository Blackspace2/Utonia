"""Run Utonia PCA visualization on KITTI Velodyne frames.

The script reads one KITTI-style `.bin` file with shape `(N, 4)` where each
point stores `(x, y, z, intensity)`. Utonia expects `coord`, `color`, and
`normal`, so this script feeds XYZ as coordinates and fills the missing color
and normal modalities with zeros by default. When `--path` points to a
directory, the script sorts all `.bin` files and plays them in a single
Open3D window.

Usage:
    export PYTHONPATH=./
    python demo/9_pca_kitti.py --path data/my-kitti/testing/velodyne/1748335122471.bin

Directory playback:
    export PYTHONPATH=./
    python demo/9_pca_kitti.py --path data/my-kitti/testing/velodyne

Optional intensity visualization:
    export PYTHONPATH=./
    python demo/9_pca_kitti.py --path data/my-kitti/testing/velodyne/1748335122471.bin --use-intensity-as-color
"""

import argparse
import os
import time

import numpy as np
import open3d as o3d
import torch

import utonia

try:
    import flash_attn
except ImportError:
    flash_attn = None

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_pca_color(feat, brightness=1.25, center=True):
    """Project point features to RGB with PCA for visualization."""
    _, _, v = torch.pca_lowrank(feat, center=center, q=12, niter=5)
    projection = feat @ v
    projection = (
        projection[:, :3] * 0.4 + projection[:, 3:6] * 0.2 + projection[:, 9:12] * 0.4
    )
    min_val = projection.min(dim=-2, keepdim=True)[0]
    max_val = projection.max(dim=-2, keepdim=True)[0]
    div = torch.clamp(max_val - min_val, min=1e-6)
    color = (projection - min_val) / div * brightness
    return color.clamp(0.0, 1.0)


def load_kitti_bin(path, use_intensity_as_color=False):
    """Load one KITTI Velodyne frame into the dictionary format expected by Utonia."""
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    coord = points[:, :3]
    intensity = points[:, 3:4]
    if use_intensity_as_color:
        color = np.repeat(np.clip(intensity, 0.0, 1.0), 3, axis=1) * 255.0
    else:
        color = np.zeros_like(coord)
    normal = np.zeros_like(coord)
    return {
        "coord": coord.astype(np.float32),
        "color": color.astype(np.float32),
        "normal": normal.astype(np.float32),
    }


def load_model():
    """Load the pretrained Utonia encoder with the same fallback as the outdoor demo."""
    if flash_attn is not None:
        model = utonia.load("utonia", repo_id="Pointcept/Utonia").to(device)
    else:
        custom_config = dict(
            enc_patch_size=[1024 for _ in range(5)],
            enable_flash=False,
        )
        model = utonia.load(
            "utonia", repo_id="Pointcept/Utonia", custom_config=custom_config
        ).to(device)
    model.eval()
    return model


def infer_pca_color(model, transform, point):
    """Run Utonia and return PCA colors in the original point order."""
    point = transform(point)
    with torch.inference_mode():
        for key in point.keys():
            if isinstance(point[key], torch.Tensor) and device == "cuda":
                point[key] = point[key].cuda(non_blocking=True)
        point = model(point)
        for _ in range(2):
            assert "pooling_parent" in point.keys()
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = point.feat[inverse]
            point = parent
        pca_color = get_pca_color(point.feat, brightness=1, center=True)
    return pca_color[point.inverse].cpu().numpy()


def list_kitti_bins(path):
    """Return one file or all `.bin` files from a directory in sorted order."""
    if os.path.isdir(path):
        files = [
            os.path.join(path, name)
            for name in sorted(os.listdir(path))
            if name.endswith(".bin")
        ]
        if not files:
            raise FileNotFoundError(f"No .bin files found in directory: {path}")
        return files
    if os.path.isfile(path):
        return [path]
    raise FileNotFoundError(f"Path not found: {path}")


def infer_frame_delay(current_path, next_path):
    """Estimate a playback delay from timestamp-like filenames."""
    if next_path is None:
        return 0.0
    current_stem = os.path.splitext(os.path.basename(current_path))[0]
    next_stem = os.path.splitext(os.path.basename(next_path))[0]
    if current_stem.isdigit() and next_stem.isdigit():
        delay = (int(next_stem) - int(current_stem)) / 1000.0
        return min(max(delay, 0.01), 0.5)
    return 0.1


def visualize_sequence(model, transform, paths, use_intensity_as_color=False):
    """Visualize one frame or a directory of frames in one Open3D window."""
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name="Utonia KITTI PCA", width=1280, height=720)
    pcd = o3d.geometry.PointCloud()
    geometry_added = False

    for index, path in enumerate(paths):
        point = load_kitti_bin(path, use_intensity_as_color)
        original_coord = point["coord"].copy()
        original_pca_color = infer_pca_color(model, transform, point)
        pcd.points = o3d.utility.Vector3dVector(original_coord)
        pcd.colors = o3d.utility.Vector3dVector(original_pca_color)

        if not geometry_added:
            visualizer.add_geometry(pcd)
            geometry_added = True
        else:
            visualizer.update_geometry(pcd)

        print(f"[{index + 1}/{len(paths)}] {os.path.basename(path)}")
        if not visualizer.poll_events():
            break
        visualizer.update_renderer()
        time.sleep(
            infer_frame_delay(
                path, paths[index + 1] if index + 1 < len(paths) else None
            )
        )

    while visualizer.poll_events():
        visualizer.update_renderer()
        time.sleep(0.03)
    visualizer.destroy_window()


def main():
    """Run feature extraction and Open3D PCA visualization for one or more KITTI frames."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        required=True,
        help="Path to one KITTI .bin file or a directory of `.bin` files.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.2,
        help="Inference scale passed to utonia.transform.default for outdoor LiDAR.",
    )
    parser.add_argument(
        "--use-intensity-as-color",
        action="store_true",
        help="Map KITTI intensity to grayscale RGB instead of filling color with zeros.",
    )
    args = parser.parse_args()

    utonia.utils.set_seed(6985480)
    model = load_model()
    transform = utonia.transform.default(args.scale, apply_z_positive=False)
    paths = list_kitti_bins(args.path)
    visualize_sequence(model, transform, paths, args.use_intensity_as_color)


if __name__ == "__main__":
    main()
