#  clumpIC-rewind -c rewind.ini : take a Milky Way and its satellites back in time, to inject the
#  satellites into an isolated-galaxy IC at a chosen look-back time.
#   [mw_baryons]  rewind of the DICE disc: the stars formed during the look-back time go back to gas,
#                 dM* = (1-R) SFR t (thin disc only), M_gas(-t) = M_gas(0) + dM* - accretion t
#                 -> <output>.params / .config for DICE.
#   [satellite X] orbit rewind of MW + satellites (Patel, Besla & Sohn 2017, sec. 3: movable host,
#                 Chandrasekhar dynamical friction, eq. 5, Coulomb logarithm eq. 6 with L=0, C=1.22,
#                 alpha=1), satellites as Hernquist halos (+ baryons) cut at their tidal radius at the
#                 injection point, i.e. with the mass the simulation gets; the tidal radius (actual
#                 orbit, not circular) and the injection point are iterated to consistency. The MW is
#                 the clumpIC host + spherically averaged baryons of [host] config. DICE params of each
#                 satellite rewritten with cut1 = r_t, and [satellite] sections for the clumpIC merge.
#  Units: kpc, km/s, Msol; time in Gyr (1 kpc/(km/s) = 0.97779 Gyr).

import os
import re
import sys
import configparser
from argparse import ArgumentParser
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import erf

from clumpIC.eddington import build_df
from clumpIC.cli import read_baryons, to_float

G = 4.30091e-6   # kpc (km/s)^2 / Msol
TU = 0.97779     # Gyr per kpc/(km/s)


def vec(s):
  return np.array([to_float(v) for v in s.split(",")])


def dice_value(text, key):
  """Value of 'key value' in a DICE params/config text."""
  m = re.search(r"^%s[ \t]+([^ \t#\n]+)"%re.escape(key), text, re.M)
  if m is None:
    print("  (!) %s not found in DICE file. Aborting!"%key)
    sys.exit()
  return to_float(m.group(1))


def dice_set(text, key, value):
  """Replace the value of 'key value' keeping the line layout and comment."""
  return re.sub(r"^(%s[ \t]+)([^ \t#\n]+)"%re.escape(key), lambda m: m.group(1) + value, text,
                count=1, flags=re.M)


def write_dice(params_text, config_file, name, comment):
  """<name>.params and <name>.config (copy of config_file, Galaxy line and Filename changed)."""
  open(name + ".params", "w").write(comment + params_text)
  c = open(config_file).read()
  c = re.sub(r"^(Galaxy[ \t]+)(\S+)", lambda m: m.group(1) + name + ".params", c, count=1, flags=re.M)
  c = dice_set(c, "Filename", name)
  open(name + ".config", "w").write(comment + c)
  print ("  (*) wrote %s.params, %s.config"%(name, name))


def dice_r200(config_text, m200):
  """r200 [kpc] as DICE computes it: v200 = (10 G H m200)^(1/3), r200 = v200/(10 H)."""
  H = dice_value(config_text, "H0") * 1e-3   # km/s/kpc
  return (10.*G*H*m200)**(1./3.) / (10.*H)


def rewind_baryons(b, lookback):
  """[mw_baryons]: stars formed in the last 'lookback' Gyr back to gas."""
  text = open(b["params"]).read()
  m200 = dice_value(text, "m200") * 1e10
  sfr, R = to_float(b["sfr"]), to_float(b["return_fraction"])
  acc = to_float(b.get("accretion", "0"))
  stars = [int(i) for i in b["stars"].split(",")]
  gas   = [int(i) for i in b["gas"].split(",")]
  dMstar = (1.-R) * sfr * lookback*1e9          # stars formed and kept
  dMgas  = dMstar - acc * lookback*1e9          # gas consumed by them minus gas accreted since
  Mstar = sum(dice_value(text, "mass_frac%d"%i) for i in stars) * m200
  Mgas  = sum(dice_value(text, "mass_frac%d"%i) for i in gas) * m200
  fs, fg = (Mstar-dMstar)/Mstar, (Mgas+dMgas)/Mgas
  print ("  (*) MW baryons at -%.3f Gyr: SFR=%.3g Msol/yr, R=%.3f, accretion=%.3g Msol/yr -> dM*=%.4e Msol"%(
         lookback, sfr, R, acc, dMstar) )
  print ("      stars (components %s): %.4e -> %.4e Msol (x %.4f)"%(stars, Mstar, Mstar-dMstar, fs) )
  print ("      gas   (components %s): %.4e -> %.4e Msol (x %.4f)"%(gas, Mgas, Mgas+dMgas, fg) )
  for comps, f in ((stars, fs), (gas, fg)):
    for i in comps:
      text = dice_set(text, "mass_frac%d"%i, "%.6g"%(dice_value(text, "mass_frac%d"%i)*f))
      text = dice_set(text, "npart%d"%i, "%d"%round(dice_value(text, "npart%d"%i)*f))  # same particle mass
  name = b.get("output", "MW_rewound")
  comment = ("# clumpIC-rewind: %s taken back %.3f Gyr (SFR %.3g Msol/yr, R %.3f, accretion %.3g Msol/yr):\n"
             "# stars x %.4f (components %s), gas x %.4f (components %s); everything else unchanged.\n"%(
             b["params"], lookback, sfr, R, acc, fs, stars, fg, gas))
  config_file = b.get("config", os.path.splitext(b["params"])[0] + ".config")
  write_dice(text, config_file, name, comment)


def host_profile(host_ini):
  """Milky Way of a clumpIC [baryons] config: enclosed mass M(<r) (host + baryons), host density
  rho(r) and isotropic Jeans dispersion sigma(r) of the host in the total potential."""
  cfg = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
  cfg.read(host_ini)
  m, s = cfg["model"], cfg["scaling"]
  bar = dict(cfg["baryons"])
  bar["file"] = os.path.join(os.path.dirname(host_ini), bar["file"])
  rb, Mb = read_baryons(bar)
  rs = to_float(s["rscale"])
  rho_s = to_float(s["rho_s"])
  mext = lambda x, Mtot: np.interp(x*rs, rb, Mb, left=0.) / (rho_s*rs**3*Mtot)
  df = build_df(to_float(m["alpha"]), to_float(m["beta"]), to_float(m["gamma"]),
                to_float(m.get("rmin", "1e-2")), to_float(m.get("rmax", "100")), 200, 2000, 1e-6,
                to_float(m["r_trunc"]) if "r_trunc" in m else None,
                to_float(m["r_decay"]) if "r_decay" in m else None, mext)
  M = rho_s * rs**3 * df["Mtot"]
  r = df["R"] * rs
  Mdm = M * df["Mcum"]
  Mtot = Mdm + np.interp(r, rb, Mb, left=0.)
  rho = np.gradient(Mdm, r) / (4.*np.pi*r**2)
  f = rho * G * Mtot / r**2
  tail = np.concatenate((np.cumsum((0.5*(f[1:]+f[:-1])*np.diff(r))[::-1])[::-1], [0.]))
  sigma = np.sqrt(np.where(rho > 0., tail/np.maximum(rho, 1e-300), 0.))
  print ("  (*) host: M_DM = %.4e Msol, M_b = %.4e Msol, M(<50 kpc) = %.4e Msol"%(
         Mdm[-1], Mb[-1], np.interp(50., r, Mtot)) )
  return r, Mtot, rho, sigma


def m_hernquist(M, a, r):
  return M * r**2 / (r+a)**2


def main():
  parser = ArgumentParser(description="Rewind a Milky Way and its satellites to a look-back time")
  parser.add_argument("-c", "--config", required=True, help="config file (.ini)")
  args = parser.parse_args()
  cfg = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
  cfg.read(args.config)
  os.chdir(os.path.dirname(os.path.abspath(args.config)))   # all paths relative to the config
  lookback = to_float(cfg["time"]["lookback"])

  if cfg.has_section("mw_baryons"):
    rewind_baryons(cfg["mw_baryons"], lookback)

  sats = [sec for sec in cfg.sections() if sec.startswith("satellite ")]
  if not sats:
    return
  r, Mmw, rho, sigma = host_profile(cfg["host"]["config"])
  def Mmw_at(d): return np.interp(d, r, Mmw)
  names = [sec.split()[1] for sec in sats]
  Ms  = np.array([to_float(cfg[sec]["mass"]) for sec in sats])                 # total mass at infall
  Mb  = np.array([to_float(cfg[sec].get("baryon_mass", "0")) for sec in sats])  # stars + gas (in Ms)
  Mh  = Ms - Mb                                                                # Hernquist halo
  aH  = np.array([to_float(cfg[sec]["r_hernquist"]) for sec in sats])
  a_s = np.array([to_float(cfg[sec]["a_s"]) for sec in sats])
  C, alpha = 1.22, 1.0                       # Patel et al. (2017) eq. 6, unequal masses (L = 0)
  ns = len(sats)
  # mass_today (optional): a satellite that loses most of its mass (e.g. Sgr) is integrated with a
  # mass that decreases log-linearly in time from the injected mass at -lookback to mass_today
  mtoday = [to_float(cfg[sec]["mass_today"]) if "mass_today" in cfg[sec] else None for sec in sats]
  # bound_to (optional): a satellite of another satellite (e.g. the SMC of the LMC) is not rewound but
  # placed at -lookback on a bound orbit around its host: apocentre [kpc] and eccentricity
  # e = (r_a - r_p)/(r_a + r_p), in the plane and with the sense of today's relative orbit, at the
  # apocentre along today's separation projected on that plane. It does not act on the rewind.
  bound = [cfg[sec].get("bound_to") for sec in sats]
  free = [k for k in range(ns) if bound[k] is None]
  nf = len(free)
  # each satellite is injected with its halo cut at its tidal radius rcut: in the orbits it is a
  # Hernquist halo truncated at rcut (point mass beyond) + its baryons, of total mass m_inj, the mass
  # the simulation actually gets. rcut and the injection point are iterated to consistency.
  # r_cut (optional, all satellites): fixed cut radii, no iteration (e.g. to plot the orbits of
  # satellites injected at another time with those cuts)
  fixed = all("r_cut" in cfg[sec] for sec in sats)
  rcut = np.array([to_float(cfg[sec]["r_cut"]) for sec in sats]) if fixed else np.full(ns, np.inf)
  def m_enc(k, d):
    dd = min(d, rcut[k])
    return (Mh[k] if np.isinf(dd) else m_hernquist(Mh[k], aH[k], dd)) + Mb[k]
  def m_inj(k):
    return m_enc(k, np.inf)
  t_end = -lookback / TU
  def mscale(k, t):   # mass at time t (<= 0) over the injected mass
    if mtoday[k] is None:
      return 1.
    return (mtoday[k]/m_inj(k))**(1. - min(t/t_end, 1.))

  def deriv(t, y):
    x = y[:3*(nf+1)].reshape(nf+1, 3); v = y[3*(nf+1):].reshape(nf+1, 3)
    a = np.zeros_like(x)
    for i in range(1, nf+1):
      k = free[i-1]; sk = mscale(k, t)
      d = x[i] - x[0]; dn = np.linalg.norm(d)
      a[i] -= G*Mmw_at(dn) * d/dn**3                                   # MW on satellite
      a[0] += G*sk*m_enc(k, dn) * d/dn**3                              # satellite on the movable MW
      vr = v[i] - v[0]; vn = np.linalg.norm(vr)                        # dynamical friction, eq. 5
      X = vn / (np.sqrt(2.)*max(np.interp(dn, r, sigma), 1e-3))
      lnL = max(0., np.log(dn/(C*a_s[k]))**alpha)
      a[i] -= 4.*np.pi*G**2*sk*m_inj(k)*lnL*np.interp(dn, r, rho, right=0.)/vn**2 * \
              (erf(X) - 2.*X/np.sqrt(np.pi)*np.exp(-X*X)) * vr/vn
      for jj in range(1, nf+1):                                        # satellite-satellite
        if jj != i:
          kj = free[jj-1]
          dj = x[i] - x[jj]; djn = np.linalg.norm(dj)
          a[i] -= G*mscale(kj, t)*m_enc(kj, djn) * dj/djn**3
    return np.concatenate((v.ravel(), a.ravel()))

  x0 = np.vstack([np.zeros(3)] + [vec(cfg[sats[k]]["position"]) for k in free])
  v0 = np.vstack([np.zeros(3)] + [vec(cfg[sats[k]]["velocity"]) for k in free])
  t_eval = np.linspace(0., t_end, int(round(lookback/to_float(cfg["time"].get("dt_out", "0.005"))))+1)

  def place_bound(xr, vr):
    """Satellites bound_to another satellite: placed at apocentre on an orbit of eccentricity e in the
    host's (truncated Hernquist + baryons) potential, with the plane and sense of today's relative orbit."""
    for k in range(ns):
      if bound[k] is None:
        continue
      h = names.index(bound[k])
      sec, hsec = cfg[sats[k]], cfg[sats[h]]
      dx0 = vec(sec["position"]) - vec(hsec["position"]); dv0 = vec(sec["velocity"]) - vec(hsec["velocity"])
      n = np.cross(dx0, dv0); n /= np.linalg.norm(n)
      u = dx0 - (dx0 @ n)*n; u /= np.linalg.norm(u)
      w = np.cross(n, u)
      ra, e = to_float(sec["apocentre"]), to_float(sec["eccentricity"])
      rp = ra*(1.-e)/(1.+e)
      q = np.logspace(-3, 4, 4000)
      f = G*np.array([m_enc(h, x) for x in q])/q**2
      tail = np.concatenate((np.cumsum((0.5*(f[1:]+f[:-1])*np.diff(q))[::-1])[::-1], [0.]))
      phi = lambda x: -np.interp(x, q, tail) - G*m_inj(h)/q[-1]
      vt = np.sqrt(2.*(phi(ra)-phi(rp)) / (ra**2/rp**2 - 1.))
      xr[k] = xr[h] + ra*u
      vr[k] = vr[h] + vt*w
    return xr, vr

  def tidal_radii(xr, vr):
    """Tidal radius of each satellite at the injection point, against the MW and every heavier
    satellite, for its actual orbit (not a circular one): G m(<r_t)/r_t^3 = Omega^2 + (G M(<D)/D^3)
    (2 - dlnM/dlnD), Omega = |D x dV|/D^2 the angular velocity about that host (= the circular-orbit
    Jacobi radius when Omega^2 = G M/D^3). m is the satellite's own untruncated mass profile."""
    out = []
    for i in range(ns):
      D = np.linalg.norm(xr[i])
      dlnM = np.interp(np.log(D), np.log(r), np.gradient(np.log(Mmw), np.log(r)))
      Om2 = (np.linalg.norm(np.cross(xr[i], vr[i]))/D**2)**2
      hosts = [("MW", Om2 + G*Mmw_at(D)/D**3*(2.-dlnM))]
      for j in range(ns):
        if m_inj(j) > m_inj(i):
          dx, dv = xr[i]-xr[j], vr[i]-vr[j]; Dj = np.linalg.norm(dx)
          Om2j = (np.linalg.norm(np.cross(dx, dv))/Dj**2)**2
          hosts.append((names[j], Om2j + G*m_enc(j, Dj)/Dj**3*(2.-2.*aH[j]/(Dj+aH[j]))))
      rt = [(brentq(lambda q: G*(m_hernquist(Mh[i], aH[i], q)+Mb[i])/q**3 - h, 1e-3, 1e4), hn)
            for hn, h in hosts]
      out.append((min(rt), rt))
    return out

  for it in range(1, 11):
    sol = solve_ivp(deriv, (0., t_end), np.concatenate((x0.ravel(), v0.ravel())), method="DOP853",
                    t_eval=t_eval, rtol=1e-9, atol=1e-9)
    if not sol.success:
      print ("  (!) orbit integration failed: %s. Aborting!"%sol.message)
      sys.exit()
    X = sol.y[:3*(nf+1)].T.reshape(-1, nf+1, 3); V = sol.y[3*(nf+1):].T.reshape(-1, nf+1, 3)
    xr = np.zeros((ns, 3)); vr = np.zeros((ns, 3))
    xr[free] = X[-1, 1:] - X[-1, 0]; vr[free] = V[-1, 1:] - V[-1, 0]
    xr, vr = place_bound(xr, vr)
    tr = tidal_radii(xr, vr)
    new = np.array([t[0][0] for t in tr])
    print ("  (*) iteration %d: injected masses %s Msol -> start r = %s kpc, tidal radii %s kpc"%(
           it, ", ".join("%.3e"%m_inj(k) for k in range(ns)),
           ", ".join("%.1f"%np.linalg.norm(xr[k]) for k in range(ns)), ", ".join("%.2f"%q for q in new)))
    if fixed:
      print ("  (*) cut radii fixed by r_cut: %s kpc (no iteration)"%", ".join("%.2f"%q for q in rcut))
      break
    converged = np.all(np.isfinite(rcut)) and np.all(abs(new-rcut) < 0.01*new)
    rcut = new
    if converged:
      break
  else:
    print ("  (!) tidal radii not converged after 10 iterations")

  fnames = [names[k] for k in free]
  dist = np.linalg.norm(X[:, 1:] - X[:, :1], axis=2)
  out = cfg["time"].get("orbits", "orbits_rewound.txt")
  pairs = [(i, j) for i in range(nf) for j in range(i+1, nf)]
  sep = np.column_stack([np.linalg.norm(X[:, i+1] - X[:, j+1], axis=1) for i, j in pairs]) if pairs \
        else np.zeros((len(sol.t), 0))
  np.savetxt(out, np.column_stack((sol.t*TU, dist, sep, np.linalg.norm(X[:, 0], axis=1))),
             header="t [Gyr]  " + "  ".join("r_%s [kpc]"%n for n in fnames) + "  " +
             "  ".join("d_%s-%s [kpc]"%(fnames[i], fnames[j]) for i, j in pairs) + "  |x_MW| [kpc] (MW reflex)")
  if "orbits_xyz" in cfg["time"]:   # positions relative to the MW, for plots
    np.savetxt(cfg["time"]["orbits_xyz"], np.column_stack([sol.t*TU] + [X[:, i+1] - X[:, 0] for i in range(nf)]),
               header="t [Gyr]  " + "  ".join("x_%s y_%s z_%s [kpc]"%(n, n, n) for n in fnames) + "  (relative to the MW)")
  print ("  (*) orbits from today back to -%.3f Gyr -> %s"%(lookback, out))
  for i, n in enumerate(fnames):
    rr = dist[:, i]
    peri = [sol.t[q]*TU for q in range(1, len(rr)-1) if rr[q] < rr[q-1] and rr[q] < rr[q+1]]
    print ("      %s: r today %.1f kpc, min %.1f kpc at %.3f Gyr, max %.1f kpc, r(-%.3f Gyr) = %.1f kpc, pericentres at %s Gyr"%(
           n, rr[0], rr.min(), sol.t[np.argmin(rr)]*TU, rr.max(), lookback, rr[-1],
           ", ".join("%.2f"%q for q in peri) or "none") )
  for k, (i, j) in enumerate(pairs):
    print ("      %s-%s: separation today %.1f kpc, min %.1f, max %.1f, at -%.3f Gyr %.1f kpc"%(
           fnames[i], fnames[j], sep[0, k], sep[:, k].min(), sep[:, k].max(), lookback, sep[-1, k]) )

  # injection state relative to the MW
  lines = []
  for i, sec in enumerate(sats):
    (rJ_min, by), rJ = tr[i]
    print ("  (*) %s at -%.3f Gyr: x = (%.2f, %.2f, %.2f) kpc, v = (%.2f, %.2f, %.2f) km/s (relative to the MW)%s"%(
           names[i], lookback, *xr[i], *vr[i],
           "" if bound[i] is None else ", placed on a bound orbit around %s"%bound[i]) )
    print ("      tidal radius %.2f kpc (set by %s; %s), injected mass %.4e Msol%s"%(rJ_min, by,
           ", ".join("%s: %.2f"%(hn, q) for q, hn in rJ), m_inj(i),
           "" if mtoday[i] is None else ", mass today %.3e Msol (log-linear mass loss in the rewind)"%mtoday[i]) )
    s = cfg[sec]
    if "params" in s:
      ptext = open(s["params"]).read()
      ctext = open(s["config"]).read()
      m200 = dice_value(ptext, "m200") * 1e10
      r200 = dice_r200(ctext, m200)
      rH = dice_value(ptext, "scale_length1")
      Mh200 = dice_value(ptext, "mass_frac1") * m200
      mdm = Mh200 / dice_value(ptext, "npart1")
      Mh_dice = Mh200 * (r200+rH)**2 / r200**2
      N = int(round(m_hernquist(Mh_dice, rH, rJ_min) / mdm))
      ptext = dice_set(ptext, "cut1", "%.2f"%rJ_min)
      ptext = dice_set(ptext, "npart1", "%d"%N)
      # cut_components (optional): other spherical components (e.g. a gas corona, DICE model 4 Hernquist
      # or 20 beta-model) also cut at the tidal radius, their mass and npart scaled by the mass kept
      for c in [int(v) for v in s.get("cut_components", "").split(",") if v.strip()]:
        cut_c = dice_value(ptext, "cut%d"%c)
        if cut_c <= rJ_min:
          continue
        model, sc = int(dice_value(ptext, "model%d"%c)), dice_value(ptext, "scale_length%d"%c)
        if model == 4:
          mfun = lambda x: m_hernquist(1., sc, x)
        elif model == 20:
          be = dice_value(ptext, "alpha_struct%d"%c)
          qq = np.linspace(0., cut_c, 20001); dens = (1. + (qq/sc)**2)**(-1.5*be) * qq**2
          cum = np.concatenate(([0.], np.cumsum(0.5*(dens[1:]+dens[:-1])*np.diff(qq))))
          mfun = lambda x: np.interp(x, qq, cum)
        else:
          print ("      (!) cut_components: model %d of component %d not handled. Aborting!"%(model, c))
          sys.exit()
        frac = mfun(rJ_min) / mfun(cut_c)
        ptext = dice_set(ptext, "cut%d"%c, "%.2f"%rJ_min)
        ptext = dice_set(ptext, "mass_frac%d"%c, "%.6g"%(dice_value(ptext, "mass_frac%d"%c)*frac))
        ptext = dice_set(ptext, "npart%d"%c, "%d"%round(dice_value(ptext, "npart%d"%c)*frac))
        print ("      component %d cut from %.2f to %.2f kpc: keeps %.3f of its mass"%(c, cut_c, rJ_min, frac))
      k = 2
      while re.search(r"^cut%d[ \t]"%k, ptext, re.M):
        if dice_value(ptext, "cut%d"%k) > rJ_min:
          print ("      (!) component %d cut %.2f kpc > tidal radius: disc extends beyond r_t"%(
                 k, dice_value(ptext, "cut%d"%k)) )
        k += 1
      name = s.get("output", names[i] + "_rewound")
      comment = ("# clumpIC-rewind: %s with its halo cut at the tidal radius %.2f kpc (set by %s) at -%.3f Gyr,\n"
                 "# npart1 = %d keeps the halo particle mass %.4e Msol.\n"%(s["params"], rJ_min, by, lookback, N, mdm))
      write_dice(ptext, s["config"], name, comment)
      lines.append("[satellite %s]\nfile     = %s.g2\nposition = %.3f, %.3f, %.3f   ; kpc, relative to the MW\n"
                   "velocity = %.3f, %.3f, %.3f   ; km/s\n"%(names[i], name, *xr[i], *vr[i]) +
                   ("spin     = %s   ; stellar spin axis, as given in %s\n"%(s["spin"], sec) if "spin" in s else ""))
  if lines:
    out = cfg["time"].get("satellites", "satellites_rewound.ini")
    open(out, "w").write(
      "; clumpIC-rewind: satellites at -%.3f Gyr, to paste into the clumpIC config (paths relative to\n"
      "; the rewind config, run DICE on the *_rewound.config files first)\n\n"%lookback + "\n".join(lines))
    print ("  (*) [satellite] sections -> %s"%out)


if __name__ == "__main__":
  main()
