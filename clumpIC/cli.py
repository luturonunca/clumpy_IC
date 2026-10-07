#  clumpIC -c config.ini : equilibrium N-body model of an alpha-beta-gamma profile
#  (Eddington inversion), scaled to physical units, plus optional extra particles and
#  sinks, written as dimensionless .npy, Gadget2 HDF5 and/or RAMSES ascii.

import os
import io
import sys
import configparser
from contextlib import redirect_stdout
from argparse import ArgumentParser
import numpy as np

from clumpIC.eddington import build_df
from clumpIC.sample import draw, speeds_at
from clumpIC.writers import write_hdf5, write_g2, write_g2_merged, read_g2, write_ramses, write_grafic, MSOL, KPC

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


def read_subhalos(sub, partmass):
  """CLUMPY -g7 .drawn catalogue -> galactocentric position [kpc], (alpha,beta,gamma), r_s [kpc],
  M_tid [Msol], R_tid [kpc] of the subhalos with M_tid >= n_min partmass. CLUMPY gives (l, b, d)
  from the Sun, which sits at (r_sun, 0, 0) from the Galactic centre (gMW_RSOL)."""
  rows = [l.split() for l in open(sub["catalogue"]) if l.strip() and not l.startswith("#")]
  if any(r[9] != "kZHAO" for r in rows):   # #1-#3 are (alpha,beta,gamma) only for kZHAO subhalos
    print('  (!) subhalo profiles must be kZHAO (CLUMPY gMW_SUBS_FLAG_PROFILE=kHOST with a kZHAO host). Aborting!')
    sys.exit()
  # columns: name type l b d z Rdelta rhos rs prof #1 #2 #3 J J/Jc Mdelta Mtid Rtid Mequdens Requdens Dgal
  d = np.array([[float(r[k]) for k in (2, 3, 4, 10, 11, 12, 8, 16, 17, 20)] for r in rows])
  l, b = np.radians(d[:,0]), np.radians(d[:,1])
  pos = np.column_stack((d[:,2]*np.cos(b)*np.cos(l) - to_float(sub["r_sun"]),
                         d[:,2]*np.cos(b)*np.sin(l), d[:,2]*np.sin(b)))
  print ("  (*) subhalos: %d in catalogue, max | |r_gc| - Dgal | = %.3f kpc (rounding of d)"%(
         len(d), np.abs(np.linalg.norm(pos, axis=1) - d[:,9]).max()) )
  keep = d[:,7] >= to_float(sub.get("n_min", "100")) * partmass
  return pos[keep], d[keep,3:6], d[keep,6], d[keep,7], d[keep,8]


def build_subhalo(abg, rs, Mtid, Rtid, partmass, sub):
  """N-body subhalo: Zhao profile with scale radius rs and a Kazantzidis cut-off at its tidal
  radius (r_decay = rdecay_frac R_tid), Mtid/partmass particles of mass partmass, in kpc, km/s,
  centred on (0,0,0) at rest."""
  rt = Rtid/rs
  rd = to_float(sub.get("rdecay_frac", "0.1")) * rt
  n  = max(1, int(round(Mtid/partmass)))
  with redirect_stdout(io.StringIO()):   # keep the per-subhalo inversion quiet
    df = build_df(abg[0], abg[1], abg[2], 1e-3, rt + 15*rd, to_float(sub.get("ne", "500")),
                  to_float(sub.get("nr", "500")), 1e-6, rt, rd)
    p, v = draw(df, n, n)
  # the n particles carry the sampled part of the profile, whose total mass sets the velocities
  Mprof = n*partmass / (1.-df["fin"]-df["fout"])
  p = p*rs; v = v*np.sqrt(G*Mprof/rs)
  return p - p.mean(axis=0), v - v.mean(axis=0)


def read_baryons(bar):
  """Spherically averaged baryons of a Gadget-2 format-2 galaxy (e.g. DICE, centred on (0,0,0)):
  all particles but the halo (type 1), sorted radii [kpc] and enclosed mass [Msol]."""
  blocks = dict(read_g2(bar["file"]))
  npart = np.frombuffer(blocks[b'HEAD'][0:24], dtype='<i4')
  off = np.concatenate(([0], np.cumsum(npart)))
  pos  = np.frombuffer(blocks[b'POS '], dtype='<f4').reshape(-1, 3).astype(float)
  mass = np.frombuffer(blocks[b'MASS'], dtype='<f4').astype(float) * 1e10
  if len(mass) != len(pos):
    print ("  (!) %s: MASS block incomplete (header mass table). Aborting!"%bar["file"])
    sys.exit()
  keep = np.ones(len(pos), dtype=bool)
  keep[off[1]:off[2]] = False
  r = np.linalg.norm(pos[keep], axis=1)
  order = np.argsort(r)
  r, Mb = r[order], np.cumsum(mass[keep][order])
  print ("  (*) baryons: %s, npart by type = %s, M_b = %.4e Msol, half-mass radius %.2f kpc"%(
         bar["file"], list(npart), Mb[-1], np.interp(0.5*Mb[-1], Mb, r)) )
  return r, Mb


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

  # baryons (optional): the halo DF is built in the halo + spherically averaged baryon potential,
  # the baryon particles themselves are not resampled (merge = true writes them with the halo)
  mext = None
  if config.has_section("baryons"):
    if config.has_section("gas"):
      print('  (!) [baryons] and [gas] together: the [gas] potential would miss the baryons. Aborting!')
      sys.exit()
    s = config["scaling"]
    rb, Mb = read_baryons(config["baryons"])
    rs_b = to_float(s["rscale"])
    if "rho_s" in s:
      mext = lambda x, Mtot: np.interp(x*rs_b, rb, Mb, left=0.) / (to_float(s["rho_s"])*rs_b**3*Mtot)
    else:
      mext = lambda x, Mtot: np.interp(x*rs_b, rb, Mb, left=0.) / to_float(s["mass"])

  df = build_df(to_float(m["alpha"]), to_float(m["beta"]), to_float(m["gamma"]),
                to_float(m.get("rmin", "1e-2")), to_float(m.get("rmax", "100")),
                to_float(m.get("ne", "1e4")), to_float(m.get("nr", "1e4")),
                to_float(m.get("epsrel", "1e-6")),
                to_float(m["r_trunc"]) if "r_trunc" in m else None,
                to_float(m["r_decay"]) if "r_decay" in m else None, mext)

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
  # M is the mass of the untruncated profile; the particles only carry the sampled part,
  # otherwise the density inside rmax is too high for the drawn velocities
  partmass = M * (1.-df["fin"]-df["fout"]) / float(N)

  # subhalos (first level only) replace part of the host: the host keeps a fraction 1-f_sub of
  # its particles, so smooth host + subhalos follow the host profile on average (subhalos must be
  # distributed like the host: CLUMPY dP/dV = kHOST); the host DF and potential are unchanged
  if config.has_section("subhalos"):
    sub = config["subhalos"]
    sub_pos, sub_abg, sub_rs, sub_Mtid, sub_Rtid = read_subhalos(sub, partmass)
    f_sub = sub_Mtid.sum() / (N*partmass)
    N = int(round(N*(1.-f_sub)))
    print ("  (*) subhalos: %d with M_tid >= %g m_part, sum M_tid = %.4e Msol = %.4f of the host -> host N = %d"%(
           len(sub_Mtid), to_float(sub.get("n_min", "100")), sub_Mtid.sum(), f_sub, N) )

  pos, vel = draw(df, N, min(N, to_float(m.get("ndraw", "1e6"))))
  o = config["output"]
  name = o.get("name", "model")
  if o.getboolean("npy", False):
    np.save(name + ".npy", np.column_stack((pos, vel)))
  pos = pos * rscale
  vel = vel * np.sqrt(G*M/rscale)
  print ("  (*) N=%i (lg N=%.2f), lg (M/Msol)=%.2f, rs=%.3f kpc, m_part=%.4e Msol"%(N, np.log10(N), np.log10(M), rscale, partmass) )

  # equilibrium check, virial in the force form 2K = sum m r.dPhi/dr = sum m G M(<r)/r,
  # valid also for a truncated sample (mass outside r exerts no force)
  r_part = np.linalg.norm(pos, axis=1)
  rdphidr = G*M*np.interp(r_part/rscale, df["R"], df["Mcum"]) / r_part
  if mext is not None:
    rdphidr += G*np.interp(r_part, rb, Mb, left=0.) / r_part
  print ("  (*) 2K / sum(m r.dPhi/dr) = %.3f"%( np.sum(vel**2) / np.sum(rdphidr) ) )

  if config.has_section("subhalos"):
    # each subhalo: own N-body equilibrium model, orbit = catalogue position with an isotropic
    # velocity drawn from the host DF at that radius
    r_sub = np.linalg.norm(sub_pos, axis=1)
    vdir = np.random.normal(size=(len(r_sub), 3))
    vdir /= np.linalg.norm(vdir, axis=1)[:,None]
    sub_vel = speeds_at(df, r_sub/rscale)[:,None] * vdir * np.sqrt(G*M/rscale)
    for i in range(len(r_sub)):
      p, v = build_subhalo(sub_abg[i], sub_rs[i], sub_Mtid[i], sub_Rtid[i], partmass, sub)
      pos = np.vstack((pos, p + sub_pos[i]))
      vel = np.vstack((vel, v + sub_vel[i]))
    print ("  (*) subhalos: %d particles in %d subhalos (largest %d)"%(len(pos)-N, len(r_sub),
           int(round(sub_Mtid.max()/partmass)) if len(r_sub) else 0) )
    N = len(pos)

  if s.getboolean("recentre", False):   # remove the sampling noise in the centre of mass
    pos -= pos.mean(axis=0)
    vel -= vel.mean(axis=0)
  pos += centre
  vel += velocity

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
    if config.has_section("baryons") and config["baryons"].getboolean("merge", False):
      # baryon file + this halo in its type-1 slot: same frame, so the halo must sit on (0,0,0) at rest
      if np.any(centre != 0.) or np.any(velocity != 0.):
        print ("  (!) g2 merge: [scaling] centre and velocity must be 0,0,0 (frame of the baryon file). Aborting!")
        sys.exit()
      if len(particles):
        print ("  (!) g2 merge: [particles] not written to the Gadget file")
      write_g2_merged(name + ".g2", config["baryons"]["file"], pos, vel, np.full(N, partmass))
    else:
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
