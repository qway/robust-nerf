"""Functions used for loading data in format used by Nerfies paper.

Such data can be obtained from video using colab notebook created by Nerfies authors:
https://colab.research.google.com/github/google/nerfies/blob/main/notebooks/Nerfies_Capture_Processing.ipynb#scrollTo=5NR5OGyeUOKU
"""
import os
import json
import torch
import numpy as np
import imageio



trans_t = lambda t: torch.Tensor(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]]
).float()

rot_phi = lambda phi: torch.Tensor(
    [
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1],
    ]
).float()

rot_theta = lambda th: torch.Tensor(
    [
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1],
    ]
).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (
        torch.Tensor(
            np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        )
        @ c2w
    )
    return c2w


def load_nerfies_data(basedir, half_res=False, testskip=1):
    if testskip != 1:
        raise NotImplementedError(
            "Only test skip value of 1 is supported for this project."
        )
    if half_res:
        raise NotImplementedError(
            "Half resolution feature is not implemented for this project."
        )

    with open(os.path.join(basedir, "dataset.json"), "r", encoding="utf-8") as file_pointer:
        dataset = json.load(file_pointer)

    permuted_ids = np.random.permutation(dataset["ids"])

    imgs_paths = [
        os.path.join(basedir, "rgb", "1x", "{}.png".format(id_numer)) for id_numer in permuted_ids
    ]
    imgs = [imageio.imread(path) for path in imgs_paths]
    imgs = (np.array(imgs) / 255.0).astype(np.float32)  # keep all 4 channels (RGBA)

    def orbit_path_from_id(id_numer):
        return os.path.join(basedir, "camera", f"{id_numer}.json")

    orbit_paths = [orbit_path_from_id(id_numer) for id_numer in permuted_ids]

    camera_orbit = []
    for path in orbit_paths:
        with open(path, "r", encoding="utf-8") as file_pointer:
            camera_orbit.append(json.load(file_pointer))

    def to_transformation_matrix(orbit_pos):
        transformation_matrix = np.zeros((4, 4))
        transformation_matrix[:3, :3] = np.array(orbit_pos["orientation"])
        transformation_matrix[:3, 3] = orbit_pos["position"]
        transformation_matrix[3, 3] = 1
        return transformation_matrix

    poses = [to_transformation_matrix(pos) for pos in camera_orbit]
    poses = np.stack(poses)
    render_poses = torch.stack(
        [
            pose_spherical(angle, -30.0, 4.0)
            for angle in np.linspace(-180, 180, 40 + 1)[:-1]
        ],
        0,
    )
    height, width = imgs[0].shape[:2]
    focal = camera_orbit[0]["focal_length"]

    train_count = int(len(permuted_ids) * 0.85)
    test_count = int((len(permuted_ids) - train_count) * 0.6)
    val_count = len(permuted_ids) - test_count - train_count
    counts = [
        0,
        train_count,
        train_count + val_count,
        train_count + val_count + test_count,
    ]
    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    return imgs, poses, render_poses, [height, width, focal], i_split
