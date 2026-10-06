#  Distribution function of an alpha-beta-gamma (Zhao) profile with isotropic velocity
#  dispersion by Eddington inversion, in units M=1, rscale=1, G=1.
#  Ported from nbody.py (RE May 2019, http://arxiv.org/abs/1906.01642).

from math import pi
import sys
import numpy as np
from scipy.integrate import quad


def build_df(a, b, g, Rmin, Rmax, NE, NR, EPSREL, rt=None, rd=None):
  def rho_abg(r) :   return r**(-g) * (1. + r**a )**((g-b)/a)
  if rt is None:
    rho = rho_abg
  else:
    # Kazantzidis, Magorrian & Moore (2004) cut-off beyond rt: rho(rt) (r/rt)^eps exp(-(r-rt)/rd),
    # eps chosen so that density and logarithmic slope are continuous at rt (finite total mass)
    eps = -g - (b-g) * rt**a/(1.+rt**a) + rt/rd
    def rho(r) :
      return rho_abg(r) if r <= rt else rho_abg(rt) * (r/rt)**eps * np.exp(-(r-rt)/rd)
    rho = np.vectorize(rho, otypes=[float])
    print("     (!) Kazantzidis cut-off at r_trunc=%g, r_decay=%g (units of rscale), eps=%.4f"%(rt, rd, eps))
  M =  4 * pi * quad( lambda r: r*r * rho(r) , 0., np.inf)[0] # non-normalized total mass

  R = np.logspace(np.log10(Rmin),np.log10(Rmax),num=int(NR))

  # mass fractions within Rmin and outside Rmax (not sampled)
  fout = 1 - 4 * pi * quad( lambda r: r*r * rho(r)/M , 0., Rmax)[0]
  fin  =     4 * pi * quad( lambda r: r*r * rho(r)/M , 0., Rmin)[0]
  print("     (!) %.2f per cent of the mass < Rmin, %.2f per cent > Rmax"%(100.*fin, 100.*fout) )

  # cumulative mass
  Mr  = np.vectorize(lambda x: 4*pi *  quad( lambda r: r*r * rho(r)/M , 0., x  )[0]  )

  # potential
  Phi = np.vectorize(lambda x: -4*pi * (1/x * quad( lambda r: r*r * rho(r)/M , 0., x    )[0]  +     quad( lambda r: r   * rho(r)/M , x, np.inf)[0] )   )

  print("      *  Building Psi and Nu arrays" )
  psi = - Phi(R)
  nu  =   rho(R) / M
  Mcum = Mr(R)

  print("      *  Calculating gradients" )
  dndp   = np.gradient(nu,   psi)
  d2nd2p = np.gradient(dndp, psi)

  print("      *  Evaluating DF" )
  f = np.vectorize( lambda e: 1./(np.sqrt(8)*pi*pi) * (quad( lambda p:  np.interp(p, psi[::-1], d2nd2p[::-1]) / np.sqrt(e-p) , 0., e,  epsrel=EPSREL)[0]  ) )

  maxE = psi[0]           # most-bound energy
  minE = maxE/float(NE)   # least-bound energy
  E = np.linspace(minE,maxE,num=int(NE))
  DF = f(E)               # here we build the DF

  # check whether DF is physical - if not, check for rounding errors in integration, try increasing/decreasing NR, NE
  if np.any (DF < 0) :
    print("      *  Exit. DF < 0, see df.dat" )
    np.savetxt("df.dat", np.column_stack(( E, DF)))
    sys.exit(0)
  else: print("      *  DF >= 0, all good" )

  print("      *  Finding maximum likelihood" )
  maxPLikelihood = []
  for RR in R:
    allowed = np.where (E <= np.interp(RR,R,psi) )[0]
    ThismaxPLikelihood = 1.1 * np.amax( np.interp(E[allowed],E,DF) * np.sqrt( 2.*(np.interp(RR, R, psi) - E[allowed] )) * RR*RR )   # 10 per cent tollerance
    maxPLikelihood.append(ThismaxPLikelihood)
  maxPLikelihood = np.array(maxPLikelihood)

  return {"R": R, "psi": psi, "Mcum": Mcum, "E": E, "DF": DF,
          "maxPLikelihood": maxPLikelihood, "fin": fin, "fout": fout,
          "Mtot": M}   # total mass of the profile with rho_0 = 1, rscale = 1
