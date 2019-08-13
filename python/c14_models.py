import matplotlib.pyplot as plt
import numba
import numpy as np
import scipy as sp
from scipy import integrate
import seaborn as sns

# Atmospheric C14 function

@numba.njit
def C_atm(x):
    x -=1
    xp = np.array([ 1891.5, 1892.5, 1893.5, 1894.5, 1895.5, 1896.5, 1897.5, 1898.5, 1899.5, 1900.5, 1901.5, 1902.5, 1903.5, 1904.5, 1905.5, 1906.5, 1907.5, 1908.5, 1909.5, 1910.5, 1911.5, 1912.5, 1913.5, 1914.5, 1915.5, 1916.5, 1917.5, 1918.5, 1919.5, 1920.5, 1921.5, 1922.5, 1923.5, 1924.5, 1925.5, 1926.5, 1927.5, 1928.5, 1929.5, 1930.5, 1931.5, 1932.5, 1933.5, 1934.5, 1935.5, 1936.5, 1937.5, 1938.5, 1939.5, 1940.5, 1941.5, 1942.5, 1943.5, 1944.5, 1945.5, 1946.5, 1947.5, 1948.5, 1949.5, 1950.5, 1951.5, 1952.5, 1953.5, 1954.5, 1955.5, 1956.5, 1957.5, 1958.5, 1959.5, 1960.5, 1961.5, 1962.5, 1963.5, 1964.5, 1965.5, 1966.5, 1967.5, 1968.5, 1969.5, 1970.5, 1971.5, 1972.5, 1973.5, 1974.5, 1975.5, 1976.5, 1977.5, 1978.5, 1979.5, 1980.5, 1981.5, 1982.5, 1983.5, 1984.5, 1985.5, 1986.5, 1987.5, 1988.5, 1989.5, 1990.5, 1991.5, 1992.5, 1993.5, 1994.5, 1995.5, 1996.5, 1997.5, 1998.5, 1999.5, 2000.5, 2001.5, 2002.5, 2003.5, 2003.963938,  2004.421821,  2004.879704, 2005.337587,  2005.79547 ,  2006.253353,  2006.711236, 2007.169119,  2007.627002,  2008.084885,  2008.542768, 2009.000651,  2009.458534,  2009.916417,  2010.3743  , 2010.3743  ,  2010.626   ,  2011.125   ,  2011.626   , 2012.125   ,  2012.626   ,  2013.125   ,  2013.626   , 2014.125   ,  2014.626   ,  2015.125   ,  2015.625   , 2016.125   ,  2016.625   ,  2020.      ])
    fp = np.array([-0.002     , -0.002     , -0.002     , -0.002     , -0.002     , -0.00233   , -0.00267   , -0.003     , -0.00333   , -0.00367   , -0.004     , -0.00433   , -0.00467   , -0.005     , -0.00533   , -0.00567   , -0.006     , -0.00633   , -0.00667   , -0.007     , -0.00733   , -0.00767   , -0.008     , -0.00833   , -0.00867   , -0.009     , -0.00933   , -0.00967   , -0.01      , -0.01033   , -0.01067   , -0.011     , -0.01133   , -0.01167   , -0.012     , -0.01233   , -0.01267   , -0.013     , -0.01333   , -0.01367   , -0.014     , -0.01433   , -0.01467   , -0.015     , -0.01533   , -0.01567   , -0.016     , -0.01633   , -0.01667   , -0.0202    , -0.0194    , -0.0196    , -0.0225    , -0.0217    , -0.0221    , -0.0216    , -0.0211    , -0.0223    , -0.0246    , -0.0248    , -0.0248    , -0.0249    , -0.0239    , -0.0211    , -0.0082    , 0.0265    ,  0.073     ,  0.1402    ,  0.228     ,  0.2123    , 0.2216    ,  0.3585    ,  0.7183    ,  0.8357    ,  0.7563    , 0.6919    ,  0.6236    ,  0.5645    ,  0.5454    ,  0.5291    , 0.4994    ,  0.4656    ,  0.4186    ,  0.4008    ,  0.3698    , 0.3525    ,  0.3339    ,  0.3258    ,  0.2958    ,  0.2645    , 0.2567    ,  0.2383    ,  0.2242    ,  0.2093    ,  0.2013    , 0.1911    ,  0.1826    ,  0.1734    ,  0.1635    ,  0.1525    , 0.1429    ,  0.1364    ,  0.1284    ,  0.1221    ,  0.1155    , 0.1099    ,  0.1043    ,  0.0981    ,  0.09      ,  0.0866    , 0.0807    ,  0.0749    ,  0.0689    ,  0.06349874,  0.06145905, 0.0597412 ,  0.05822747,  0.05641489,  0.05479531,  0.05364811, 0.05205465,  0.05067073,  0.04927897,  0.04788014,  0.04647508, 0.04506462,  0.04364948,  0.04223027,  0.04223027,  0.0403    , 0.0347    ,  0.0371    ,  0.0312    ,  0.0299    ,  0.0193    , 0.0219    ,  0.0182    ,  0.018     ,  0.0116    ,  0.0129    , 0.0096    ,  0.0097    ,  0.0097    ])
    
    if x<xp[0]:
        return fp[0]
    elif x>xp[-1]:
        return fp[-1]
    else:
        for i in range(len(xp)):
            if x<xp[i]:
                break
        return fp[i-1] + (fp[i] - fp[i-1]) * (x - xp[i-1]) / (xp[i] - xp[i-1])


# Scenario I1

# I stands for inheritence. Half of the DNA is inherited from the mother to the daughter.
# 1 stands for 1 population.
# Here, the more realistic 2 daughter assumption is used.

## Gillespie


@numba.njit
def I1a(Dbirth, Dcoll, lam, mu, N=1000, C_init=np.inf):
    t = Dbirth
    
    if C_init==np.inf:
        C_init = C_atm(t)
    cc = [C_init for i in range(N)]
    
    # Pre-allocated propensity vector, no values yet
    aa = np.empty(2)
    
    tt = list(np.empty(0))
    meancc = list(np.empty(0))
    
    while True:
        tt.append(t)
        meancc.append(np.array(cc).mean())
        
        N = len(cc)
        
        aa[0] = lam * N
        aa[1] = mu * N
        
        t += np.random.exponential(1/aa.sum())
        
        p_division = aa[0] / aa.sum()
        
        division_event = np.random.binomial(1, p_division)
        
        if division_event:
            # cell division
            # select random cell
            i = np.random.randint(N)
            # get C14 concentration from a random cell and delete this cell
            c = cc.pop(i)
            # calculate new C14 concentration
            c = 0.5 * (c + C_atm(t))
            # add two new cells with updated concentration
            cc.append(c)
            cc.append(c)
        else:
            # cell death
            # select random cell
            i = np.random.randint(N)
            # delete it
            cc.pop(i)
        
        if t > Dcoll:
            break
    return tt, meancc


## Empirical ODE

def I1(Dbirth, Dcoll, lam, C_init=np.inf, t_eval=None):
    if t_eval is None:
        t_eval=[Dbirth, Dcoll]
    
    if C_init==np.inf:
        C_init = C_atm(Dbirth)
    
    def rhs(c, t, lam):
        return lam * (C_atm(t) - c)
    
    sol = sp.integrate.odeint(func=rhs, 
                            y0=[C_init],
                            t=t_eval,
                            args=(lam, ))
    c = sol[:, 0]
    
    
    return t_eval, c

def I1c(Dbirth, Dcoll, lam):
    Ddev = Dbirth
    tdeath = Dcoll - Ddev
    C = C_atm(Dcoll - tdeath) * np.exp(-lam * tdeath)

    step = min(1.0/lam/10.0, 1.0)
    num = int(tdeath / step)
    ages = sp.linspace(0, tdeath,  num, endpoint = True)
    y = np.vectorize(C_atm)(Dcoll - ages) * np.exp(-lam * ages)
    integral = sp.integrate.trapz(y, ages)
    
    C += lam * integral

    return C


def I1T(Dbirth, Dcoll, lam, C_init=np.inf, t_eval=None, lam_arg=()):
    """ Here, lam(t, *lam_arg) is a function of t.
    """
    if t_eval is None:
        t_eval=[Dbirth, Dcoll]
    
    if C_init==np.inf:
        C_init = C_atm(Dbirth)
    
    def rhs(c, t, lam, lam_arg):
        return lam(t, *lam_arg) * (C_atm(t) - c)
    
    sol = sp.integrate.odeint(func=rhs, 
                            y0=[C_init],
                            t=t_eval,
                            args=(lam, lam_arg))
    c = sol[:, 0]
    
    
    return t_eval, c

# I2

def I2(Dbirth, Dcoll, lam, f, C_init=np.inf, t_eval=None):
    if t_eval is None:
        t_eval = [Dbirth, Dcoll]
    assert 0<=f<=1
    C2 = C_atm(Dbirth)
    
    tt, C1 = I1(Dbirth, Dcoll, lam, C_init=C_init, t_eval=t_eval)
    return tt, f*C1 + (1.0-f)*C2

# IX

def IX(Dbirth, Dcoll, deltalaml, ff, C_init=np.inf, t_eval=None):
    if t_eval is None:
        t_eval = [Dbirth, Dcoll]
    f = np.sum(ff)
    assert 0<=np.sum(f)<=1
    C0 = C_atm(Dbirth)
    
    laml = np.cumsum(deltalaml)
    
    CC = np.array([I1(Dbirth, Dcoll, lam, C_init=C_init, t_eval=t_eval)[1] for lam in laml])
    
    ff=np.array(ff)
    ff.shape = (len(CC), 1)
    return t_eval, np.sum(ff*CC, axis=0) + (1.0-f)*C0


# IK

def IK(Dbirth, Dcoll, lam, f, C_init=np.inf, t_eval=None):
    if t_eval is None:
        t_eval=[Dbirth, Dcoll]
    
    if C_init==np.inf:
        C_init = C_atm(Dbirth)
    
    def rhs(c, t, lam, f):
        cA, cB = c
        d_cA = lam * (np.vectorize(C_atm)(t) - cA)
        d_cB = lam * f * (cA - cB)
        return np.array([d_cA, d_cB])
    
    sol = sp.integrate.odeint(func=rhs, 
                            y0=[C_init, C_init],
                            t=t_eval,
                            args=(lam, f))
    
    cA = sol[:,0]
    cB = sol[:,1]

    c = f * cA + (1 - f) * cB 
    
    return t_eval, c

# IL

def IL(Dbirth, Dcoll, lam, f, C_init=np.inf, t_eval=None):
    if t_eval is None:
        t_eval=[Dbirth, Dcoll]
    
    if C_init==np.inf:
        C_init = C_atm(Dbirth)
    
    def rhs(c, t, lam, f):
        cA, cB = c
        d_cA = lam * (np.vectorize(C_atm)(t) - cA)
        d_cB = lam * f / (1-f) * (cA - cB)
        return np.array([d_cA, d_cB])
    
    sol = sp.integrate.odeint(func=rhs, 
                            y0=[C_init, C_init],
                            t=t_eval,
                            args=(lam, f))
    
    cA = sol[:,0]
    cB = sol[:,1]

    c = f * cA + (1 - f) * cB 
    
    return t_eval, c

# old stufff

Klag = np.vectorize(C_atm)
t0 = 0.0

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
