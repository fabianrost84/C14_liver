import atm_c14_data
import scipy as sp
from scipy import integrate
import sys
import pandas as pd

Klag = atm_c14_data.Klag
t0 = 0.5

def C_scenario_A(Dbirth, Dcoll, r):
  """ predicted C14 for scenario A
      r... cell death rate = turnover rate
  """
  if r > 100.0:
    return Klag(Dcoll)
  else:
    if Dbirth < Dcoll:
      Ddev = Dbirth - t0
      tdeath = Dcoll - Ddev
      C = Klag(Dcoll - tdeath) * sp.exp(-r * tdeath)
      
      
      if r>0:
        step = min(1.0/r/100.0, 1.0)
      else:
        step = 1.0
      num = tdeath / step
      
      ages = sp.linspace(0, tdeath,  num, endpoint = True)
      
      y = Klag(Dcoll - ages) * sp.exp(-r * ages)
      
      integral = sp.integrate.trapz( y, ages)
      C += r * integral
      
      return C
    else:
      return sp.nan

def C_scenario_2POP(Dbirth, Dcoll, r, f):
  Ddev = Dbirth - t0
  tdeath = Dcoll - Ddev
  
  nonrenew = Klag(Dcoll - tdeath)
  renew = C_scenario_A(Dbirth, Dcoll, r)
  
  C = (1.0 - f) * nonrenew + f * renew
  
  return C

def C_scenario_3POP(Dbirth, Dcoll, r1, r2, f, f12):
  Ddev = Dbirth - t0
  tdeath = Dcoll - Ddev
  
  nonrenew = Klag(Dcoll - tdeath)
  renew1 = C_scenario_A(Dbirth, Dcoll, r1)
  renew2 = C_scenario_A(Dbirth, Dcoll, r2)
  
  C = (1.0 - f) * nonrenew + f * (f12 * renew1 + (1.0 - f12) * renew2)
  
  return C