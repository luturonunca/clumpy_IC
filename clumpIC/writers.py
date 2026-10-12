#  Output formats. Positions in kpc, velocities in km/s, masses in Msol on input.

import os
import sys
import numpy as np

MSOL = 1.98892e33  # g
KPC  = 3.0857e21   # cm


def write_hdf5(filename, pos, vel, partmass):
  """Gadget2 compatible HDF5 (halo as PartType1), as in nbody2hdf5.py."""
  import h5py
  mU= 1.0e10   # Msol
  rU= 1.0      # kpc
  vU= 207.4    # km/s
  N = len(pos)
  f = h5py.File(filename, 'w')
  print ("  (*) Writing output hdf5 to \'%s\'"%filename)
  Header = f.create_group("Header")
  Header.attrs.create('BoxSize',                0.,             dtype=np.float64)
  Header.attrs.create('Flag_Cooling',           0 ,             dtype=np.int32)
  Header.attrs.create('Flag_Entropy_ICs',       (0,0,0,0,0,0),  dtype=np.uint32)
  Header.attrs.create('Flag_Feedback',          0 ,             dtype=np.int32)
  Header.attrs.create('Flag_Metals',            0 ,             dtype=np.int32)
  Header.attrs.create('Flag_Sfr',               0 ,             dtype=np.int32)
  Header.attrs.create('Flag_Flag_StellarAge',   0 ,             dtype=np.int32)
  Header.attrs.create('HubbleParam',            1.,             dtype=np.float64)
  Header.attrs.create('MassTable',              (0.,partmass/mU,0.,0.,0.,0.,), dtype=np.float64)
  Header.attrs.create('NumFilesPerSnapshot',    1 ,             dtype=np.int32)
  Header.attrs.create('NumPart_ThisFile',       (0,N,0,0,0,0) , dtype=np.int32)
  Header.attrs.create('NumPart_Total',          (0,N,0,0,0,0) , dtype=np.uint32)
  Header.attrs.create('NumPart_Total_HighWord', (0,0,0,0,0,0) , dtype=np.uint32)
  Header.attrs.create('Omega0',                 0.,             dtype=np.float64)
  Header.attrs.create('OmegaLambda',            0.,             dtype=np.float64)
  Header.attrs.create('Redshift',               0.,             dtype=np.float64)
  Header.attrs.create('Time',                   0.,             dtype=np.float64)
  PartType1 = f.create_group("PartType1")
  PartType1.create_dataset("Acceleration", (N,3), dtype=np.float32)
  PartType1.create_dataset("Coordinates",  data=np.array(pos/rU, dtype=np.float32))
  PartType1.create_dataset("ParticleIDs",  data=np.arange(N, dtype=np.uint32))
  PartType1.create_dataset("Potential",    (N,),  dtype=np.float32)
  PartType1.create_dataset("Velocities",   data=np.array(vel/vU, dtype=np.float32))
  f.close()


def write_g2(filename, pos, vel, mass, ptype):
  """Gadget-2 binary, SnapFormat 2 (labelled blocks HEAD, POS, VEL, ID, MASS), as read by the
  RAMSES DICE patch (ic_format='Gadget2') with its default gadget_scale_l/v/m: kpc, km/s,
  1e10 Msol. Ported from write_gadget2 in halo&Clumps.ipynb. ptype 0..5 (1 = halo -> FAM_DM,
  2..5 -> FAM_STAR); particles are written grouped by type, ids 1..N in that order."""
  import struct
  def outer_block(fh, tag4, payload):
    # format 2: record with the 4-char label and the size of the next record, then the data record
    fh.write(struct.pack('<i', 8)); fh.write(tag4)
    fh.write(struct.pack('<i', len(payload)+8)); fh.write(struct.pack('<i', 8))
    fh.write(struct.pack('<i', len(payload))); fh.write(payload); fh.write(struct.pack('<i', len(payload)))
  order = np.argsort(ptype, kind='stable')
  npart = np.bincount(np.asarray(ptype, dtype=int), minlength=6)[:6]
  head = b''.join([np.asarray(npart, dtype='<i4').tobytes(),     # npart[6]
                   np.zeros(6, dtype='<f8').tobytes(),            # mass[6]: 0 -> MASS block
                   struct.pack('<d', 0.), struct.pack('<d', 0.),  # time, redshift
                   struct.pack('<i', 0), struct.pack('<i', 0),    # flag_sfr, flag_feedback
                   np.asarray(npart, dtype='<u4').tobytes(),     # npartTotal[6]
                   struct.pack('<i', 0), struct.pack('<i', 1),    # flag_cooling, num_files
                   struct.pack('<d', 0.), struct.pack('<d', 0.),  # BoxSize, Omega0
                   struct.pack('<d', 0.), struct.pack('<d', 0.),  # OmegaLambda, HubbleParam
                   struct.pack('<i', 0), struct.pack('<i', 0),    # flag_stellarage, flag_metals
                   np.zeros(6, dtype='<u4').tobytes(),            # npartTotalHighWord[6]
                   struct.pack('<i', 0)])                         # flag_entropy_instead_u
  head += b'\x00' * (256 - len(head))
  with open(filename, 'wb') as fh:
    outer_block(fh, b'HEAD', head)
    outer_block(fh, b'POS ', np.ascontiguousarray(pos[order], dtype='<f4').tobytes())
    outer_block(fh, b'VEL ', np.ascontiguousarray(vel[order], dtype='<f4').tobytes())
    outer_block(fh, b'ID  ', np.arange(1, len(pos)+1, dtype='<i4').tobytes())
    outer_block(fh, b'MASS', np.ascontiguousarray(mass[order]/1e10, dtype='<f4').tobytes())
  print ("  (*) Writing Gadget-2 (format 2) to \'%s\', npart by type = %s (kpc, km/s, 1e10 Msol)"%(filename, list(npart)) )


def read_g2(filename):
  """Blocks of a Gadget-2 SnapFormat 2 file (e.g. DICE), as a list of (4-char label, raw bytes)."""
  import struct
  blocks = []
  with open(filename, 'rb') as fh:
    while True:
      marker = fh.read(4)
      if not marker:
        break
      label = fh.read(8); fh.read(4)          # label record: 4-char label, size of the next record
      n = struct.unpack('<i', fh.read(4))[0]
      data = fh.read(n); fh.read(4)
      blocks.append((label[:4], data))
  return blocks


def rotation_to(c, t):
  """Rotation matrix taking unit vector c onto unit vector t (Rodrigues; the roll about t is free)."""
  k = np.cross(c, t); s, cth = np.linalg.norm(k), c @ t
  if s < 1e-12:   # parallel or antiparallel
    if cth > 0:
      return np.eye(3)
    a = np.cross(c, [1., 0., 0.] if abs(c[0]) < 0.9 else [0., 1., 0.]); a /= np.linalg.norm(a)
    return 2.*np.outer(a, a) - np.eye(3)
  k /= s
  K = np.array([[0., -k[2], k[1]], [k[2], 0., -k[0]], [-k[1], k[0], 0.]])
  return np.eye(3) + s*K + (1.-cth)*(K @ K)


def write_g2_merged(filename, base, pos, vel, mass, satellites=(), zero_momentum=False, extra_gas=None,
                    max_particles=1e8):
  """Gadget-2 format 2: the file 'base' (e.g. DICE gas + stars + halo) with its halo (type 1)
  replaced by pos, vel, mass (kpc, km/s, Msol), plus optional satellite galaxies: a list of
  (file, position, velocity[, spin]), each a Gadget-2 format-2 galaxy (e.g. DICE) moved to position
  [kpc] with bulk velocity [km/s]; with spin (a vector, or None), the galaxy is first rotated about its
  centre so that the angular momentum of its stars (types 2-5) points along spin. Every block is assembled type by type, base first then the
  satellites; blocks with an entry per particle (POS, VEL, ID, MASS, Z) get the new halo in the
  base type-1 slot (Z = 0); blocks of the gas or stars only (U, RHO, HSML, AGE) are copied; ids are
  renumbered 1..N. extra_gas = (pos, vel, mass, u, Z): gas particles (type 0) appended after those of
  the galaxies, u = internal energy per unit mass [(km/s)^2], RHO and HSML set to 0 (not read by the
  RAMSES DICE patch). zero_momentum: remove the mass-weighted mean velocity of all particles (the
  positions are kept, so the base galaxy stays on its centre). Above max_particles in total the output is
  split into files filename.1 .. filename.N (each with every type in proportion, as read by the RAMSES
  DICE patch with ic_nfile = N; its 32-bit block sizes need < 1.79e8 particles per file); returns N."""
  import struct
  def outer_block(fh, tag4, payload):
    fh.write(struct.pack('<i', 8)); fh.write(tag4)
    fh.write(struct.pack('<i', len(payload)+8)); fh.write(struct.pack('<i', 8))
    fh.write(struct.pack('<i', len(payload))); fh.write(payload); fh.write(struct.pack('<i', len(payload)))
  def covered(n, npart, fname, label):   # particle types that a block of n entries covers
    for types in ([0, 1, 2, 3, 4, 5], [0], [2, 3, 4, 5], [0, 2, 3, 4, 5]):
      if n == npart[types].sum():
        return types
    print ("  (!) %s: block %s of %d entries, not matched to particle types. Aborting!"%(fname, label, n))
    sys.exit()
  galaxies = []   # (file, npart, {label: {type: array}}, shift, boost)
  for sat in [(base, None, None)] + list(satellites):
    fname, shift, boost = sat[:3]
    spin = sat[3] if len(sat) > 3 else None
    blocks = read_g2(fname)
    head = bytes(blocks[0][1])
    if fname == base:
      base_head, labels = head, [label for label, _ in blocks[1:]]
    npart = np.frombuffer(head[0:24], dtype='<i4').copy()
    if np.any(np.frombuffer(head[24:72], dtype='<f8') != 0.):
      print ("  (!) %s: header mass table not zero (MASS block incomplete). Aborting!"%fname)
      sys.exit()
    split = {}
    for label, data in blocks[1:]:
      ncomp = 3 if label in (b'POS ', b'VEL ') else 1
      arr = np.frombuffer(data, dtype='<i4' if label == b'ID  ' else '<f4').reshape(-1, ncomp)
      types = covered(len(arr), npart, fname, label)
      off = np.concatenate(([0], np.cumsum(npart[types])))
      split[label] = {t: arr[off[k]:off[k+1]] for k, t in enumerate(types)}
    if spin is not None:   # rotate about the centre of mass so that the stellar spin is along 'spin'
      m = np.concatenate([split[b'MASS'][t][:, 0] for t in sorted(split[b'MASS'])]).astype(float)
      x = np.concatenate([split[b'POS '][t] for t in sorted(split[b'POS '])]).astype(float)
      v = np.concatenate([split[b'VEL '][t] for t in sorted(split[b'VEL '])]).astype(float)
      xc, vc = (m[:, None]*x).sum(0)/m.sum(), (m[:, None]*v).sum(0)/m.sum()
      st = [t for t in (2, 3, 4, 5) if t in split[b'POS '] and len(split[b'POS '][t])]
      Ls = sum(np.cross(split[b'POS '][t].astype(float)-xc, split[b'VEL '][t].astype(float)-vc).sum(0)
               for t in st)
      target = np.asarray(spin, dtype=float)
      Rm = rotation_to(Ls/np.linalg.norm(Ls), target/np.linalg.norm(target))
      for t in split[b'POS ']:
        split[b'POS '][t] = ((split[b'POS '][t].astype(float)-xc) @ Rm.T + xc).astype('<f4')
        split[b'VEL '][t] = ((split[b'VEL '][t].astype(float)-vc) @ Rm.T + vc).astype('<f4')
      print ("  (*) %s: stellar spin %s rotated onto %s"%(fname, np.round(Ls/np.linalg.norm(Ls), 3),
             np.round(target/np.linalg.norm(target), 3)))
    galaxies.append((fname, npart, split, shift, boost))
  if extra_gas is not None:   # a pseudo-galaxy with gas only, laid out like the base blocks
    gp, gv, gm, gu, gz = extra_gas
    gas_data = {b'POS ': gp, b'VEL ': gv, b'MASS': gm/1e10, b'U   ': gu, b'Z   ': gz,
                b'ID  ': np.zeros(len(gp)), b'RHO ': np.zeros(len(gp)), b'HSML': np.zeros(len(gp))}
    split = {}
    for label in labels:
      ncomp = 3 if label in (b'POS ', b'VEL ') else 1
      dt = '<i4' if label == b'ID  ' else '<f4'
      split[label] = {t: np.zeros((0, ncomp), dtype=dt) for t in galaxies[0][2][label]}
      if 0 in split[label]:
        if label not in gas_data:
          print ("  (!) %s: block %s has gas entries and is not handled for extra gas. Aborting!"%(base, label))
          sys.exit()
        split[label][0] = np.asarray(gas_data[label], dtype=dt).reshape(len(gp), ncomp)
    galaxies.append(("subhalo gas", None, split, None, None))
  for fname, npart, split, shift, boost in galaxies[1:]:
    if sorted(split) != sorted(labels) or any(sorted(split[l]) != sorted(galaxies[0][2][l]) for l in labels):
      print ("  (!) %s: blocks differ from those of %s. Aborting!"%(fname, base))
      sys.exit()
  # base halo replaced, satellites moved
  sp = galaxies[0][2]
  new = {b'POS ': pos, b'VEL ': vel, b'MASS': mass/1e10, b'Z   ': np.zeros(len(pos))}
  for label in labels:
    if 1 in sp[label]:
      if label in new:
        sp[label][1] = np.asarray(new[label], dtype='<f4').reshape(len(pos), -1)
      elif label != b'ID  ':
        print ("  (!) %s: block %s has halo entries and is not handled. Aborting!"%(base, label))
        sys.exit()
  for fname, npart, split, shift, boost in galaxies[1:]:
    if shift is None:
      continue
    for t in split[b'POS ']:
      split[b'POS '][t] = (split[b'POS '][t] + shift).astype('<f4')
      split[b'VEL '][t] = (split[b'VEL '][t] + boost).astype('<f4')
  new_npart = np.zeros(6, dtype=int)
  for t in range(6):
    new_npart[t] = sum(len(g[2][b'POS '][t]) for g in galaxies)
  # per label and type: base first, then the satellites and the extra gas
  cat = {label: {t: np.concatenate([g[2][label][t] for g in galaxies]) for t in sorted(sp[label])}
         for label in labels}
  del galaxies, sp
  first_id = np.concatenate(([1], 1 + np.cumsum(new_npart)))    # ids 1..N in type order
  cat[b'ID  '] = {t: np.arange(first_id[t], first_id[t+1], dtype='<i4').reshape(-1, 1) for t in cat[b'ID  ']}
  if zero_momentum:
    m = np.concatenate([cat[b'MASS'][t][:, 0] for t in sorted(cat[b'MASS'])]).astype(float)
    vel_all = np.concatenate([cat[b'VEL '][t] for t in sorted(cat[b'VEL '])]).astype(float)
    vcm = (m[:, None]*vel_all).sum(axis=0) / m.sum()
    del m, vel_all
    for t in cat[b'VEL ']:
      cat[b'VEL '][t] = (cat[b'VEL '][t] - vcm).astype('<f4')
    print ("  (*) zero total momentum: removed v_cm = (%.3f, %.3f, %.3f) km/s from all particles"%tuple(vcm))
  nfile = max(1, int(np.ceil(new_npart.sum() / float(max_particles))))
  for k in range(nfile):
    # file k: the k-th slice of every particle type
    lo = (new_npart * k) // nfile; hi = (new_npart * (k+1)) // nfile
    head = bytearray(base_head)
    head[0:24]  = (hi-lo).astype('<i4').tobytes()            # npart[6] of this file
    head[96:120] = new_npart.astype('<u4').tobytes()         # npartTotal[6]
    head[124:128] = np.array([nfile], dtype='<i4').tobytes() # num_files
    fname = filename if nfile == 1 else "%s.%d"%(filename, k+1)
    with open(fname, 'wb') as fh:
      outer_block(fh, b'HEAD', bytes(head))
      for label in labels:
        outer_block(fh, label, np.ascontiguousarray(
          np.concatenate([cat[label][t][lo[t]:hi[t]] for t in sorted(cat[label])])).tobytes())
  print ("  (*) Writing Gadget-2 (format 2) to \'%s\'%s: %s with its halo replaced%s, npart by type = %s"%(
         filename, "" if nfile == 1 else ".1 .. .%d (ic_nfile = %d)"%(nfile, nfile), base,
         "".join(" + %s"%f for f, *_ in satellites) + (" + subhalo gas" if extra_gas is not None else ""),
         list(new_npart)) )
  return nfile


def write_grafic(dirname, fields, dx):
  """RAMSES grafic gas files for a non-cosmological run (filetype='grafic'): one
  unformatted file per primitive variable (ic_d, ic_u, ic_v, ic_w, ic_p) in code units,
  read by init_flow_fine at levelmin. Header n1,n2,n3,dx,xoff1-3,astart,omega_m,omega_l,h0,
  then one record per z plane with x running fastest; arrays are indexed [ix,iy,iz]."""
  from scipy.io import FortranFile
  os.makedirs(dirname, exist_ok=True)
  for name, arr in fields.items():
    n1, n2, n3 = arr.shape
    f = FortranFile(os.path.join(dirname, name), 'w')
    f.write_record(np.array([n1, n2, n3], dtype=np.int32),
                   np.array([dx, 0., 0., 0., 1., 0., 0., 0.], dtype=np.float32))
    for i3 in range(n3):
      f.write_record(np.array(arr[:,:,i3].T, dtype=np.float32))
    f.close()
  print ("  (*) Writing RAMSES grafic gas %s to \'%s\' (%i^3)"%(",".join(fields), dirname, n1) )


def write_ramses(dirname, pos, vel, mass, sinks, units_length, units_density, units_time):
  """RAMSES ascii ICs: <dirname>/ic_part (x y z vx vy vz m) and, if any sinks,
  <dirname>/ic_sink (m x y z vx vy vz lx ly lz m_smbh). Code units follow RAMSES:
  length units_length, mass units_density*units_length**3, velocity units_length/units_time.
  Positions are relative to the box centre: RAMSES adds boxlen/2 when reading."""
  lU = KPC / units_length
  mU = MSOL / (units_density*units_length**3)
  vU = 1e5 * units_time / units_length
  os.makedirs(dirname, exist_ok=True)
  print ("  (*) Writing RAMSES ascii to \'%s\' (1 kpc = %.4e, 1 Msol = %.4e, 1 km/s = %.4e code units)"%(dirname, lU, mU, vU) )
  np.savetxt(os.path.join(dirname, "ic_part"),
             np.column_stack((pos*lU, vel*vU, mass*mU)), fmt="%.8e")
  if len(sinks):
    np.savetxt(os.path.join(dirname, "ic_sink"),
               np.column_stack((sinks[:,6]*mU, sinks[:,0:3]*lU, sinks[:,3:6]*vU,
                                np.zeros((len(sinks),3)), sinks[:,7]*mU)), fmt="%.8e")
