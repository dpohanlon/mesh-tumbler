import numpy as np
import trimesh

from tqdm import tqdm

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["axes.facecolor"] = "FFFFFF"
rcParams["savefig.facecolor"] = "FFFFFF"
rcParams["xtick.direction"] = "in"
rcParams["ytick.direction"] = "in"

rcParams.update({"figure.autolayout": True})

rcParams["figure.figsize"] = (16, 9)

import numpy as np

import seaborn as sns

import pandas as pd

from scipy.stats import gaussian_kde

from scipy.spatial import KDTree

from skopt import gp_minimize

import argparse


def calculate_overhangs(
    voxel_points, normals, voxel_grid, max_overhang_angle, voxel_size
):
    overhang_voxel_count = 0
    overhang_voxels = []
    for point, normal in zip(voxel_points, normals.reshape(-1, 3)):
        if is_overhang(point, normal, voxel_grid, max_overhang_angle, voxel_size):
            overhang_voxel_count += 1
            overhang_voxels.append(point)
    return overhang_voxel_count, overhang_voxels


def is_overhang(voxel, normal_vector, voxel_grid, max_overhang_angle, voxel_size):
    normal_vector = np.asarray(normal_vector)
    norm = np.linalg.norm(normal_vector)
    if norm == 0:
        return False

    angle = np.arccos(normal_vector[2] / norm)
    if angle <= np.radians(max_overhang_angle):
        return False

    voxel_indices = np.floor(voxel / voxel_size).astype(int)
    x, y, z = voxel_indices

    # Check for support below the current voxel
    for below_z in range(z - 1, -1, -1):
        if voxel_grid[x, y, below_z]:
            return False

    return True


def rotate_mesh(mesh, rotation):
    r = trimesh.transformations.euler_matrix(
        np.radians(rotation[0]),
        np.radians(rotation[1]),
        np.radians(rotation[2]),
        "sxyz",
    )
    rotated_mesh = mesh.copy()
    rotated_mesh.apply_transform(r)
    return rotated_mesh


def interpolate_normals(voxel_points, mesh):
    kdtree = KDTree(mesh.vertices)

    distances, vertex_indices = kdtree.query(voxel_points)

    normals = mesh.vertex_normals[vertex_indices]

    return normals


def visualize_voxels(voxel_points, overhang_voxels):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for voxel in voxel_points:
        ax.scatter(voxel[0], voxel[1], voxel[2], color="blue", s=10)
    for voxel in overhang_voxels:
        ax.scatter(voxel[0], voxel[1], voxel[2], color="red", s=10)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig("voxels.png")
    plt.clf()


def overhang_count(rotated_mesh, pitch, max_overhang_angle, return_voxels=False):
    voxelized = rotated_mesh.voxelized(pitch)

    voxels = voxelized.matrix

    normals = interpolate_normals(voxelized.points, rotated_mesh)

    overhangs, overhang_voxels = calculate_overhangs(
        voxelized.points, normals, voxelized.matrix, max_overhang_angle, pitch
    )

    if return_voxels:
        return overhangs, overhang_voxels
    else:
        return overhangs


def rotation_overhangs(angles, mesh, pitch, max_overhang_angle):

    # angles *= 360.

    rotated_mesh = rotate_mesh(mesh, angles)

    return overhang_count(rotated_mesh, pitch, max_overhang_angle)


def scan_rotations(mesh, pitch, max_overhang_angle):
    overhangs_list = []
    alphas = []
    betas = []
    gammas = []

    for a in tqdm(range(0, 360, 90)):
        for b in range(0, 360, 10):
            for c in range(0, 360, 10):

                alphas.append(a)
                betas.append(b)
                gammas.append(c)

                rotated_mesh = rotate_mesh(mesh, [a, b, c])

                overhangs = overhang_count(rotated_mesh, pitch, max_overhang_angle)

                overhangs_list.append(overhangs)

    return alphas, betas, gammas, overhangs_list


def plot_scan_df(df):

    sns.barplot(x="a", y="overhangs", data=df)
    plt.savefig("a.png")
    plt.clf()

    sns.barplot(x="b", y="overhangs", data=df)
    plt.savefig("b.png")
    plt.clf()

    sns.barplot(x="c", y="overhangs", data=df)
    plt.savefig("c.png")
    plt.clf()


def fill_scan_df(alphas, betas, gammas, overhangs_list):

    df = pd.DataFrame(
        {"a": alphas, "b": betas, "c": gammas, "overhangs": overhangs_list}
    )

    return df


def load_mesh(file_path):
    mesh = trimesh.load_mesh(file_path)
    trimesh.repair.fix_normals(mesh)
    return mesh


def log_prob(angles, mesh, pitch, max_overhang_angle):
    if np.any(angles < -180) or np.any(angles > 180):
        return -np.inf
    return rotation_overhangs(angles, mesh, pitch, max_overhang_angle)


def main(input_file, n_calls, n_restarts, pitch, max_overhang_angle):

    mesh = load_mesh(input_file)

    def opt(t):
        return -log_prob(np.array(t), mesh, pitch, max_overhang_angle)

    res = gp_minimize(
        opt,
        [(-180, 180), (-180, 180), (-180, 180)],
        acq_func="EI",
        n_calls=n_calls,
        n_random_starts=n_restarts,
        random_state=1234,
    )

    print(res.x, res.fun)

def run():

    parser = argparse.ArgumentParser(
        description="MCMC for 3D mesh overhang minimization."
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input mesh file."
    )
    parser.add_argument(
        "--n_calls", type=int, default=100, help="Number of function calls."
    )
    parser.add_argument(
        "--n_restarts", type=int, default=5, help="Number of optimisation restarts."
    )
    parser.add_argument("--pitch", type=float, default=10.0, help="Pitch value.")
    parser.add_argument(
        "--max_overhang_angle", type=float, default=45.0, help="Maximum overhang angle."
    )

    args = parser.parse_args()

    main(
        args.input_file,
        args.n_calls,
        args.n_restarts,
        args.pitch,
        args.max_overhang_angle,
    )

if __name__ == "__main__":
    run()
