#  Output formats. Positions in kpc, velocities in km/s, masses in Msol on input.

import os
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
