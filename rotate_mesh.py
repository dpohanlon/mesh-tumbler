import numpy as np
from stl import mesh
import trimesh

from tqdm import tqdm

from scipy.ndimage import sobel

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

import pandas as pd

import warnings

import seaborn as sns

from scipy.spatial import KDTree


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


def overhang_count(mesh, pitch, max_overhang_angle, return_voxels=False):
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


best_rotation = None
min_overhangs = float("inf")
max_overhang_angle = 45

# Load the STL file using trimesh
mesh = trimesh.load_mesh("/Users/dan/Downloads/overhang_test.stl")

trimesh.repair.fix_normals(mesh)

pitch = 10.0

overhangs_list = []
as_ = []
bs = []
cs = []

for a in tqdm(range(0, 360, 90)):
    for b in range(0, 360, 10):
        for c in range(0, 360, 10):

            as_.append(a)
            bs.append(b)
            cs.append(c)

            rotated_mesh = rotate_mesh(mesh, [a, b, c])

            overhangs = overhang_count(rotated_mesh, pitch, max_overhang_angle)

            overhangs_list.append(overhangs)

# visualize_voxels(voxelized.points, np.array(overhang_voxels))

import pandas as pd

d = pd.DataFrame({"a": as_, "b": bs, "c": cs, "overhangs": overhangs_list})

d.to_csv("overhangs.csv")

sns.barplot(x="a", y="overhangs", data=d)
plt.savefig("a.png")
plt.clf()

sns.barplot(x="b", y="overhangs", data=d)
plt.savefig("b.png")
plt.clf()

sns.barplot(x="c", y="overhangs", data=d)
plt.savefig("c.png")
plt.clf()
