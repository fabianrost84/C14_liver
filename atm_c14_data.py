import numpy as np
import scipy as sp
from scipy import interpolate
import pandas as pd

tlag = 1.0
_c14_data_file = 'data/14C_levin_data_until_2016_for_plotting.xlsx'

c14_data = pd.read_excel(_c14_data_file, names=['year', 'd14C'])

#######################
# extrapolate 14C concentration from late data and add values 
late_data = c14_data.query('year > 2010')

def f2(x, a, b, c):
    return a * np.exp(-b * (x-c))

def fit(f, p0=None):
    p = sp.optimize.curve_fit(f, late_data['year'], late_data['d14C'], p0)[0]
    return lambda x: f(x, *p)

tt = np.arange(c14_data['year'].max() + 0.5, 2025, 0.5)
cc = fit(f2, [1, 1, 2010])(tt)

j = c14_data.index.max()
for i in np.arange(0, len(tt)-0.5, 1).astype(int):
    c14_data.loc[j+i+1, 'year'] = tt[i]
    c14_data.loc[j+i+1, 'd14C'] = cc[i]
    
#######################

last_c14 = c14_data['d14C'].loc[c14_data['year'].argmax()]
extrapolate_value =[2020.0, last_c14]

c14_data.loc['extrapolate_value'] = extrapolate_value

c14_data = c14_data.sort_values(by='year')
c14_data['d14C'] /= 1000.0

def interpolate(xp, fp):
  return lambda x: sp.interp(x, xp, fp)


K = interpolate(c14_data['year'], c14_data['d14C'])
K.__doc__ ='Interpolation function for the atmospheric C14 data I got from Paula' 


Klag = interpolate(c14_data['year'] + tlag, c14_data['d14C'])
Klag.__doc__ ='Interpolation function for the atmospheric C14 data I got from Paula' 
