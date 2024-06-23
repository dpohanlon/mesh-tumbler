<p align="center">
  <img width="588" height="250" src="assets/tumbler_logo.png">
<br>
<br>
Determine mesh rotations to minimise overhangs for 3D printing.

</p>

------

Installation
------------

Install from the GitHub repository:

```bash
git clone git@github.com:dpohanlon/mesh-tumbler.git
cd mesh-tumbler
pip install .
```


Usage
------------

Mesh Tumbler uses projections of the normals in the -z direction to determine the amount of unsupported overhangs above a user defined thereshold (by default 45 degrees). The mesh is rotated according to a Bayesian Gaussian process optimisation procedure in order to find the rotations that minimise the number of these overhangs. The only argument required to the executable is the path to an STL file:

```bash
mesh-tumbler --input_file path/to/mesh.stl
```

and the rotations about the `[x, y, z]` axes, along with the value of the optimised function, are printed. This also takes optional arguments of the number of function calls to make to the optimisation function, `n_calls`, and the pitch of the voxelised mesh, `pitch`, which are useful to optimise the time taken to achieve reasonable results. The maximum angle above which a the projection is considered an overhang can also be specified with `max_overhang_angle`.
