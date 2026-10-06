#  clumpIC -c config.ini : equilibrium N-body model of an alpha-beta-gamma profile
#  (Eddington inversion), scaled to physical units, plus optional extra particles and
#  sinks, written as dimensionless .npy, Gadget2 HDF5 and/or RAMSES ascii.

import os
import sys
import configparser
from argparse import ArgumentParser
import numpy as np

from clumpIC.eddington import build_df
from clumpIC.sample import draw
from clumpIC.writers import write_hdf5, write_g2, write_ramses, write_grafic, MSOL, KPC

G = 4.30091e-6   # kpc (km/s)^2 / Msol


def to_float(s):
  """Float from a config value; accepts Fortran exponents (3.0857d21) copied from a namelist."""
  return float(s.strip().lower().replace("d", "e"))


def read_objects(config, section, pos, partmass, centre, extra_cols):
  """Lines 'name = x, y, z, vx, vy, vz, m[, ...]' (kpc, km/s, Msol). A velocity entry 'vc' or
  '-vc' is the circular speed of the sampled halo at the object's distance from the centre."""
  objs = []
  if not config.has_section(section):
    return np.zeros((0, 7+extra_cols))
  r_halo = np.sort(np.linalg.norm(pos-centre, axis=1))
  for name, line in config.items(section):
    vals = [v.strip() for v in line.split(",")]
    xyz = np.array([to_float(v) for v in vals[0:3]])
    r = np.linalg.norm(xyz-centre)
    vc = np.sqrt(G*partmass*np.searchsorted(r_halo, r)/r)
    vel = [ (vc if v == "vc" else -vc if v == "-vc" else to_float(v)) for v in vals[3:6] ]
    rest = [to_float(v) for v in vals[6:]]
    rest += [0.]*(1+extra_cols-len(rest))
    print("  (*) %s %s: r=%.3f kpc, v_c=%.3f km/s"%(section[:-1], name, r, vc))
    objs.append(list(xyz) + vel + rest)
  return np.array(objs)


def write_gas(g, df, M, rscale, centre_kpc, o):
  """Isothermal gas in hydrostatic balance with the sampled (truncated) halo, on the RAMSES
  levelmin grid: rho = rho_ref exp(-(Phi(r)-Phi(r_ref))/cs^2), P = rho cs^2, v = 0.
  Phi is the model potential, shifted inside rmax and Keplerian outside it, so that it is
  the potential of the particles actually sampled. Gas self-gravity is not included."""
  units_length, units_density, units_time = [to_float(o[k]) for k in
                                             ("units_length", "units_density", "units_time")]
  lU = KPC / units_length
  mU = MSOL / (units_density*units_length**3)
  vU = 1e5 * units_time / units_length
  R, psi, Mcum = df["R"], df["psi"], df["Mcum"]
  def psi_t(x):   # dimensionless (M=1, rscale=1) potential of the halo truncated at R[-1]
    return np.where(x < R[-1], np.interp(x, R, psi) - psi[-1] + Mcum[-1]/R[-1], Mcum[-1]/np.maximum(x, R[-1]))

  cs      = to_float(g["cs"])          # km/s
  rho_ref = to_float(g["rho_ref"])     # Msol/kpc^3 at r_ref
  r_ref   = to_float(g["r_ref"])       # kpc
  n       = 2**int(to_float(g["levelmin"]))
  boxlen  = to_float(g["boxlen"])      # code units, as in &AMR_PARAMS
  centre  = boxlen/2. + centre_kpc*lU     # halo centre ([scaling] centre) in code units

  dx = boxlen/n
  x = (np.arange(n)+0.5)*dx
  X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
  r = np.sqrt((X-centre[0])**2 + (Y-centre[1])**2 + (Z-centre[2])**2) / lU   # kpc
  phi = -G*M/rscale * psi_t(r/rscale)                                          # (km/s)^2
  phi_ref = -G*M/rscale * psi_t(r_ref/rscale)
  rho = rho_ref * np.exp(-(phi-phi_ref)/cs**2)                                 # Msol/kpc^3
  print ("  (*) gas: cs=%.2f km/s, rho(%.2f kpc)=%.3e Msol/kpc^3, rho_max=%.3e, M_box=%.4e Msol"%(
         cs, r_ref, rho_ref, rho.max(), rho.sum()*(dx/lU)**3) )
  rho_code = rho * mU/lU**3
  write_grafic(o.get("name", "model") + "_ramses",
               {"ic_d": rho_code, "ic_p": rho_code*(cs*vU)**2}, dx)


def main():
  parser = ArgumentParser(description="Equilibrium N-body ICs by Eddington inversion")
  parser.add_argument("-c", "--config", required=True, help="config file (.ini)")
  args = parser.parse_args()

  if not os.path.isfile(args.config):
    print('Config file "{}" does not exist! Aborting!'.format(args.config))
    sys.exit()
  config = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
  config.read(args.config)

  # dimensionless model
  m = config["model"]
  np.random.seed(int(to_float(m.get("seed", "667408"))))
  N = int(to_float(m["npart"]))
  df = build_df(to_float(m["alpha"]), to_float(m["beta"]), to_float(m["gamma"]),
                to_float(m.get("rmin", "1e-2")), to_float(m.get("rmax", "100")),
                to_float(m.get("ne", "1e4")), to_float(m.get("nr", "1e4")),
                to_float(m.get("epsrel", "1e-6")),
                to_float(m["r_trunc"]) if "r_trunc" in m else None,
                to_float(m["r_decay"]) if "r_decay" in m else None)
  pos, vel = draw(df, N, min(N, to_float(m.get("ndraw", "1e6"))))

  o = config["output"]
  name = o.get("name", "model")
  if o.getboolean("npy", False):
    np.save(name + ".npy", np.column_stack((pos, vel)))

  # physical scaling: (M=1, rscale=1, G=1) -> Msol, kpc, km/s
  s = config["scaling"]
  rscale = to_float(s["rscale"])
  if "rho_s" in s:   # normalisation by the scale density instead of the total mass
    M = to_float(s["rho_s"]) * rscale**3 * df["Mtot"]
  else:
    M = to_float(s["mass"])
  if "r_trunc" in m:
    print ("  (*) M(<r_trunc=%.2f kpc) = %.4e Msol, total profile mass = %.4e Msol"%(
           to_float(m["r_trunc"])*rscale, M*np.interp(to_float(m["r_trunc"]), df["R"], df["Mcum"]), M) )
  centre   = np.array([to_float(v) for v in s.get("centre", "0,0,0").split(",")])
  velocity = np.array([to_float(v) for v in s.get("velocity", "0,0,0").split(",")])
  pos = pos * rscale
  vel = vel * np.sqrt(G*M/rscale)
  if s.getboolean("recentre", False):   # remove the sampling noise in the centre of mass
    pos -= pos.mean(axis=0)
    vel -= vel.mean(axis=0)
  pos += centre
  vel += velocity
  # M is the mass of the untruncated profile; the particles only carry the sampled part,
  # otherwise the density inside rmax is too high for the drawn velocities
  partmass = M * (1.-df["fin"]-df["fout"]) / float(N)
  print ("  (*) N=%i (lg N=%.2f), lg (M/Msol)=%.2f, rs=%.3f kpc, m_part=%.4e Msol"%(N, np.log10(N), np.log10(M), rscale, partmass) )

  # equilibrium check, virial in the force form 2K = sum m r.dPhi/dr = sum m G M(<r)/r,
  # valid also for a truncated sample (mass outside r exerts no force)
  r_part = np.linalg.norm(pos-centre, axis=1)
  rdphidr = G*M*np.interp(r_part/rscale, df["R"], df["Mcum"]) / r_part
  print ("  (*) 2K / sum(m r.dPhi/dr) = %.3f"%( np.sum((vel-velocity)**2) / np.sum(rdphidr) ) )

  particles = read_objects(config, "particles", pos, partmass, centre, 0)
  sinks     = read_objects(config, "sinks",     pos, partmass, centre, 1)

  if config.has_section("gas") and o.getboolean("ramses_ascii", False):
    write_gas(config["gas"], df, M, rscale, centre, o)

  if o.getboolean("hdf5", False):
    write_hdf5(name + ".hdf5", pos, vel, partmass)
  if o.getboolean("g2", False):
    # halo = Gadget type 1 (RAMSES FAM_DM), [particles] = type 4 (FAM_STAR); sinks are not
    # part of a Gadget file (use ramses_ascii for ic_sink)
    if len(sinks):
      print ("  (!) g2: [sinks] not written to the Gadget file")
    write_g2(name + ".g2", np.vstack((pos, particles[:,0:3])), np.vstack((vel, particles[:,3:6])),
             np.concatenate((np.full(N, partmass), particles[:,6])),
             np.concatenate((np.full(N, 1), np.full(len(particles), 4))))
  if o.getboolean("ramses_ascii", False):
    write_ramses(name + "_ramses",
                 np.vstack((pos, particles[:,0:3])), np.vstack((vel, particles[:,3:6])),
                 np.concatenate((np.full(N, partmass), particles[:,6])), sinks,
                 to_float(o["units_length"]), to_float(o["units_density"]),
                 to_float(o["units_time"]))
  print ("  (*) all done :o)" )


if __name__ == "__main__":
  main()
