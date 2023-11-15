### Code is adopted from https://viser.studio/examples/11_colmap_visualizer/
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as onp
import tyro
from tqdm.auto import tqdm
import click
import numpy as np
import viser
import viser.transforms as tf
from viser.extras.colmap import (
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
)

def error_to_confidence(error):
    # Here smaller_beta means slower transition from 0 to 1.
    # Increasing beta raises steepness of the curve.
    beta = 1
    # confidence = 1 / np.exp(beta*error)
    confidence = 1 / (1 + np.exp(beta*error))
    return confidence

def get_center(pts):
    center = pts.mean(0)
    dis = np.linalg.norm(pts - center[None,:], axis=1)
    mean, std = dis.mean(), dis.std()
    q25, q75 = np.quantile(dis, 0.25), np.quantile(dis, 0.75)
    valid = (dis > mean - 1.5 * std) & (dis < mean + 1.5 * std) & (dis > mean - (q75 - q25) * 1.5) & (dis < mean + (q75 - q25) * 1.5)
    center = pts[valid].mean(0)
    return center

def normalize_poses(poses, pts, up_est_method, center_est_method, pts3d_normal=None):
    import numpy as np
    poses = np.array(poses)
    pts = np.array(pts)

    if center_est_method == 'camera':
        center = np.mean(poses[..., 3], axis=0)
    elif center_est_method == 'lookat':
        cams_ori = poses[..., 3]
        cams_dir = np.matmul(poses[:, :3, :3], np.array([0., 0., -1.]))
        cams_dir = cams_dir / np.linalg.norm(cams_dir, axis=-1, keepdims=True)
        A = np.stack([cams_dir, -np.roll(cams_dir, 1, axis=0)], axis=-1)
        b = -cams_ori + np.roll(cams_ori, 1, 0)        
        t = np.linalg.lstsq(A, b)[0]
        center = np.mean(np.stack([cams_dir, np.roll(cams_dir, 1, 0)], axis=-1) * t[:, None, :] + np.stack([cams_ori, np.roll(cams_ori, 1, 0)], axis=-1), axis=(0, 2))
    elif center_est_method == 'point':
        center = np.mean(poses[..., 3], axis=0)
    else:
        raise NotImplementedError(f'Unknown center estimation method: {center_est_method}')

    if up_est_method == 'ground':
        import pyransac3d as pyrsc

        ground = pyrsc.Plane()
        plane_eq, inliers = ground.fit(pts, thresh=0.01)
        plane_eq = np.array(plane_eq)
        z = plane_eq[:3] / np.linalg.norm(plane_eq[:3])
        signed_distance = np.sum(np.concatenate([pts, np.ones_like(pts[..., :1])], axis=-1) * plane_eq, axis=-1)
        if np.mean(signed_distance) < 0:
            z = -z
    elif up_est_method == 'camera':
        z = np.mean(poses[..., 3] - center, axis=0)
        z = z / np.linalg.norm(z)
    else:
        raise NotImplementedError(f'Unknown up estimation method: {up_est_method}')

    y_ = np.array([z[1], -z[0], 0.])
    x = np.cross(y_, z) / np.linalg.norm(np.cross(y_, z))
    y = np.cross(z, x)

    if center_est_method == 'point':
        # rotation
        Rc = np.stack([x, y, z], axis=1)
        R = np.transpose(Rc)
        poses_homo = np.concatenate([poses, np.array([[[0.,0.,0.,1.]]]).repeat(poses.shape[0], axis=0)], axis=1)
        inv_trans = np.concatenate([np.concatenate([R, np.array([[0.,0.,0.]]).T], axis=1), np.array([[0.,0.,0.,1.]])], axis=0)
        poses_norm = (inv_trans @ poses_homo)[:,:3]
        pts = (inv_trans @ np.concatenate([pts, np.ones_like(pts[:,0:1])], axis=-1)[...,None])[:,:3,0]

        # translation and scaling
        poses_min, poses_max = np.min(poses_norm[...,3], axis=0), np.max(poses_norm[...,3], axis=0)
        pts_fg = pts[(poses_min[0] < pts[:,0]) & (pts[:,0] < poses_max[0]) & (poses_min[1] < pts[:,1]) & (pts[:,1] < poses_max[1])]
        center = get_center(pts_fg)
        tc = np.reshape(center, (3, 1))
        t = -tc
        poses_homo = np.concatenate([poses_norm, np.array([[[0.,0.,0.,1.]]]).repeat(poses_norm.shape[0], axis=0)], axis=1)
        inv_trans = np.concatenate([np.concatenate([np.eye(3), t], axis=1), np.array([[0.,0.,0.,1.]])], axis=0)
        poses_norm = (inv_trans @ poses_homo)[:,:3]
        scale = np.min(np.linalg.norm(poses_norm[...,3], ord=2, axis=-1))
        poses_norm[...,3] /= scale
        pts = (inv_trans @ np.concatenate([pts, np.ones_like(pts[:,0:1])], axis=-1)[...,None])[:,:3,0]

        # apply the rotation to the point cloud normal
        if pts3d_normal is not None:
            pts3d_normal = np.transpose(R @ np.transpose(pts3d_normal))

        pts = pts / scale

    else:
        Rc = np.stack([x, y, z], axis=1)
        tc = np.reshape(center, (3, 1))
        R, t = np.transpose(Rc), -np.matmul(np.transpose(Rc), tc)
        poses_homo = np.concatenate([poses, np.ones([poses.shape[0], 1, 1]).repeat(poses.shape[1], axis=1)], axis=-1)
        inv_trans = np.concatenate([np.concatenate([R, t], axis=1), np.array([[0., 0., 0., 1.]])], axis=0)
        poses_norm = np.matmul(inv_trans, poses_homo)[:, :3]

        scale = np.min(np.linalg.norm(poses_norm[..., 3], axis=-1))
        poses_norm[..., 3] /= scale

        pts = np.matmul(inv_trans, np.concatenate([pts, np.ones_like(pts[:, :1])], axis=-1)[:, :, None])[:, :3, 0]
        if pts3d_normal is not None:
            pts3d_normal = np.matmul(R, np.transpose(pts3d_normal)).T
        pts = pts / scale

    return poses_norm, pts, pts3d_normal

@click.command()
@click.option('--data', help='Path to data directory')
@click.option('--downsample_factor', default=8)
def main(
    data,
    downsample_factor,
) -> None:
    """Visualize COLMAP sparse reconstruction outputs.

    Args:
        colmap_path: Path to the COLMAP reconstruction directory.
        images_path: Path to the COLMAP images directory.
        downsample_factor: Downsample factor for the images.
    """
    colmap_path = Path(data) / 'sparse' / '0'
    images_path = Path(data) / 'images'
    server = viser.ViserServer()
    server.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Load the colmap info.
    cameras = read_cameras_binary(colmap_path / "cameras.bin")
    images = read_images_binary(colmap_path / "images.bin")
    points3d = read_points3d_binary(colmap_path / "points3D.bin")
    gui_reset_up = server.add_gui_button(
        "Reset up direction",
        hint="Set the camera control 'up' direction to the current camera's 'up'.",
    )

    # Preprocess colmap data
    all_c2w = []
    for i, d in enumerate(images.values()):
        R = d.qvec2rotmat()
        t = d.tvec.reshape(3, 1)
        c2w = np.concatenate([R.T, -R.T@t], axis=1)
        c2w[:,1:3] *= -1. # COLMAP => OpenGL
        # c2w = np.vstack((c2w, [0, 0, 0, 1])) # making c2w homogeneous
        all_c2w.append(c2w)
    all_c2w = np.array(all_c2w)

    pts3d_confidence = np.array([error_to_confidence(points3d[k].error) for k in points3d])
    pts3d_color = np.array([points3d[k].rgb for k in points3d])
    pts3d = np.array([points3d[k].xyz for k in points3d])
    sorted_indices = np.argsort(pts3d_confidence)
    
    normalized_c2w, normalized_pts3d, _ = normalize_poses(all_c2w, pts3d, up_est_method='ground', center_est_method='point')
        
    @gui_reset_up.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None
        client.camera.up_direction = tf.SO3(client.camera.wxyz) @ onp.array(
            [0.0, -1.0, 0.0]
        )

    gui_points = server.add_gui_slider(
        "Max points",
        min=1,
        max=len(pts3d),
        step=1,
        initial_value=min(len(pts3d), 50_000),
    )
    gui_frames = server.add_gui_slider(
        "Max frames",
        min=1,
        max=len(images),
        step=1,
        initial_value=min(len(images), 100),
    )
    gui_point_size = server.add_gui_number("Point size", initial_value=0.05)

    def visualize_colmap() -> None:
        """Send all COLMAP elements to viser for visualization. This could be optimized
        a ton!"""
        # Set the point cloud.
        points_selection = sorted_indices[:gui_points.value]
        points = pts3d[points_selection]
        colors = pts3d_color[points_selection]
        server.add_point_cloud(
            name="/colmap/pcd",
            points=points,
            colors=colors,
            point_size=gui_point_size.value,
        )

        # Interpret the images and cameras.
        img_ids = [im.id for im in images.values()]
        onp.random.shuffle(img_ids)
        img_ids = sorted(img_ids[: gui_frames.value])

        def attach_callback(
            frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle
        ) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        for img_id in tqdm(img_ids):
            img = images[img_id]
            cam = cameras[img.camera_id]

            # Skip images that don't exist.
            image_filename = images_path / img.name
            if not image_filename.exists():
                continue

            T_world_camera = tf.SE3.from_rotation_and_translation(
                tf.SO3(img.qvec), img.tvec
            ).inverse()
            frame = server.add_frame(
                f"/colmap/frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.1,
                axes_radius=0.005,
            )

            H, W = cam.height, cam.width
            fy = cam.params[1]
            image = iio.imread(image_filename)
            image = image[::downsample_factor, ::downsample_factor]
            frustum = server.add_camera_frustum(
                f"/colmap/frame_{img_id}/frustum",
                fov=2 * onp.arctan2(H / 2, fy),
                aspect=W / H,
                scale=0.15,
                image=image,
            )
            attach_callback(frustum, frame)

    need_update = True

    @gui_points.on_update
    def _(_) -> None:
        nonlocal need_update
        need_update = True

    @gui_frames.on_update
    def _(_) -> None:
        nonlocal need_update
        need_update = True

    @gui_point_size.on_update
    def _(_) -> None:
        nonlocal need_update
        need_update = True

    while True:
        if need_update:
            need_update = False

            server.reset_scene()
            visualize_colmap()

        time.sleep(1e-3)


if __name__ == "__main__":
    main()