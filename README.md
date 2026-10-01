# nbopy
Generates N-body models through Eddington inversion, and paints stellar probabilities on top using a distribution-function based approach. See [EP20](https://arxiv.org/abs/1906.01642).

**nbody.py** generates the models. See [this link](https://rerrani.github.io/code.html#nbopy) for a brief introduction.

**npaint.py** paints stars on top. See [this link](https://rerrani.github.io/code.html#npaint).

**nbody2hdf5.py** generates Gadget2 compatible HDF5 files from 'nbody.py' models.  See [this link](https://rerrani.github.io/code.html#hdf5).

To run on python2, add `from __future__ import print_function` .



## clumpIC

`clumpIC` wraps `nbody.py` and `nbody2hdf5.py` into one command driven by a config file
(same Eddington inversion and sampling, in float64):

    clumpIC -c my_halo.ini      # run from the directory where the ICs should go

### Installation

Needs python >= 3.8 with numpy and scipy (h5py only for HDF5 output). On a cluster, load a
python module first (e.g. `module load python`).

Editable install (recommended): the `clumpIC` command then always runs the code in this
repository, so a `git pull` is enough to update it.

    git clone https://github.com/luturonunca/clumpy_IC.git
    cd clumpy_IC
    pip install --user -e .     # needs pip >= 21.3; upgrade with: pip install --user -U pip

`--user` installs the command in `~/.local/bin`; if `clumpIC` is not found, add it to your PATH
(e.g. in `~/.bashrc`):

    export PATH=$HOME/.local/bin:$PATH

Inside a virtual environment or conda environment, drop `--user`.

Without installing (no pip available): put the repository on the PYTHONPATH and call the module:

    export PYTHONPATH=/path/to/clumpy_IC:$PYTHONPATH
    python3 -m clumpIC.cli -c my_halo.ini

See `examples/hernquist_sink_orbit.ini` for all options: profile and sampling (`[model]`),
physical scaling (`[scaling]`), extra point particles (`[particles]`) and RAMSES sinks
(`[sinks]`), where a velocity entry `vc`/`-vc` is the circular speed of the sampled halo, and
outputs (`[output]`: dimensionless `.npy`, Gadget2 HDF5, RAMSES ascii `ic_part`/`ic_sink` in the
code units given by `units_length`, `units_density`, `units_time` from the RAMSES namelist).

`mass` is the mass of the untruncated profile; particles only carry the part between `rmin` and
`rmax`. A small `rmax` cuts through bound orbits and the outer halo expands: the printed
virial ratio `2K / sum(m r.dPhi/dr)` is ~1 only when little mass lies beyond `rmax`.
