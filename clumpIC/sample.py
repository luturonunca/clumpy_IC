#  Draws N particles from the Eddington DF built in eddington.py (units M=1, rscale=1, G=1).
#  Ported from nbody.py (RE May 2019, http://arxiv.org/abs/1906.01642), in float64.

import numpy as np


def draw(df, N, Ndraw):
  R, psi, E, DF = df["R"], df["psi"], df["E"], df["DF"]
  Mcum, maxPLikelihood = df["Mcum"], df["maxPLikelihood"]
  fin, fout = df["fin"], df["fout"]

  # phase space volume element per energy accessible at given energy and fixed radius
  def dPdr(e,r):  return np.sqrt( 2.*(np.interp(r, R, psi) - e )) * r*r
  # likelihood of a particle to have energy E at a fixed radius
  def PLikelihood(e,r): return np.interp(e,E,DF) * dPdr(e,r)

  xx = np.zeros(int(N))
  yy = np.zeros(int(N))
  zz = np.zeros(int(N))
  vx = np.zeros(int(N))
  vy = np.zeros(int(N))
  vz = np.zeros(int(N))

  print("      *  Draw particles" )
  n=0             # current number of generated 'particles'
  Efails = 0      # rejection sampling failures in Energy

  while (n < N):
    # inverse transform sampling for R, w/o particles inside/outside Rmin/Rmax
    randMcum = fin  + (1.-(fin+fout)) * np.random.rand(int(Ndraw))
    randR = np.interp(randMcum ,Mcum,R)

    # rejection sampling for E
    psiR = np.interp(randR ,R,psi)             # potential at radius
    randE = np.random.rand(int(Ndraw)) * psiR  # random E with constraint that  E > 0 but E < Psi
    rhoE  = PLikelihood(randE,randR)           # likelihood for E at given R
    randY = np.random.rand(int(Ndraw)) * np.interp(randR,R, maxPLikelihood)
    Missidx = np.where(randY > rhoE)[0]        # sampled energies rejected
    Efails += len(Missidx)

    # repeat sampling at fixed R till we got all the energies we need
    while len(Missidx):
      randE[Missidx] = np.random.rand(len(Missidx)) * psiR[Missidx]
      rhoE[Missidx]  = PLikelihood(randE[Missidx],randR[Missidx])
      randY[Missidx] = np.random.rand(len(Missidx)) * np.interp(randR[Missidx],R, maxPLikelihood)
      Missidx = np.where(randY > rhoE)[0]
      Efails += len(Missidx)

    okEidx = np.where(randY <= rhoE)[0]

    # Let's select as many R,E combinations as we're still missing to get N particles in total
    missing = int(N) - int(n)
    if len(okEidx) <= missing:
      arraxIdx = n + np.arange(0,len(okEidx))
    else:
      arraxIdx = n + np.arange(0,missing)
      okEidx = okEidx[:missing]

    # spherical symmetric model, draw random points on sphere
    Rtheta  = np.arccos (2. * np.random.rand(len(okEidx)) - 1.)
    Rphi    = np.random.rand(len(okEidx)) * 2*np.pi

    # isotropic velocity dispersion, draw random points on sphere
    Vtheta  = np.arccos (2. * np.random.rand(len(okEidx)) - 1.)
    Vphi    = np.random.rand(len(okEidx)) * 2*np.pi
    V =  np.sqrt( 2.*(psiR[okEidx] - randE[okEidx] ))

    xx[arraxIdx] = randR[okEidx] * np.sin(Rtheta) * np.cos(Rphi)
    yy[arraxIdx] = randR[okEidx] * np.sin(Rtheta) * np.sin(Rphi)
    zz[arraxIdx] = randR[okEidx] * np.cos(Rtheta)

    vx[arraxIdx] = V * np.sin(Vtheta) * np.cos(Vphi)
    vy[arraxIdx] = V * np.sin(Vtheta) * np.sin(Vphi)
    vz[arraxIdx] = V * np.cos(Vtheta)

    n += len(okEidx)
    print ("         %.2f per cent; E rejection ratio %.2f "%(100* n/float(N) ,  100* Efails/float(n)  ) )

  return np.column_stack((xx,yy,zz)), np.column_stack((vx,vy,vz))
