import autograd.numpy as np

"""Generic, short, simple functions that can be easily separated from a specific model"""
###########################################################################################
"""Calculates the total mass given symmratio and chirp mass"""
def calculate_totalmass(chirpm,symmratio):return chirpm*symmratio**(-3/5)




###########################################################################################
"""Analytic functions for amplitude continutity"""
"""Solved the system analytically in mathematica and transferred the functions --
  probably quicker to evaluate the expressions below than solve the system with matrices and needed for autograd to function properly

  Tried to define derivates manually, but stopped because too little of an effect. *(Actually, something's not working with the second
  function anyway, so just won't use it)"""
def calculate_delta_parameter_0(f1,f2,f3,v1,v2,v3,dd1,dd3,M):
    return -((dd3*f1**5*f2**2*f3 - 2*dd3*f1**4*f2**3*f3 + dd3*f1**3*f2**4*f3 - dd3*f1**5*f2*f3**2 + \
   dd3*f1**4*f2**2*f3**2 - dd1*f1**3*f2**3*f3**2 + dd3*f1**3*f2**3*f3**2 + \
   dd1*f1**2*f2**4*f3**2 - dd3*f1**2*f2**4*f3**2 + dd3*f1**4*f2*f3**3 + \
   2*dd1*f1**3*f2**2*f3**3 - 2*dd3*f1**3*f2**2*f3**3 - dd1*f1**2*f2**3*f3**3 + \
   dd3*f1**2*f2**3*f3**3 - dd1*f1*f2**4*f3**3 - dd1*f1**3*f2*f3**4 - \
   dd1*f1**2*f2**2*f3**4 + 2*dd1*f1*f2**3*f3**4 + dd1*f1**2*f2*f3**5 - \
   dd1*f1*f2**2*f3**5 + 4*f1**2*f2**3*f3**2*v1 - 3*f1*f2**4*f3**2*v1 - \
   8*f1**2*f2**2*f3**3*v1 + 4*f1*f2**3*f3**3*v1 + f2**4*f3**3*v1 + 4*f1**2*f2*f3**4*v1 + \
   f1*f2**2*f3**4*v1 - 2*f2**3*f3**4*v1 - 2*f1*f2*f3**5*v1 + f2**2*f3**5*v1 - \
   f1**5*f3**2*v2 + 3*f1**4*f3**3*v2 - 3*f1**3*f3**4*v2 + f1**2*f3**5*v2 - \
   f1**5*f2**2*v3 + 2*f1**4*f2**3*v3 - f1**3*f2**4*v3 + 2*f1**5*f2*f3*v3 - \
   f1**4*f2**2*f3*v3 - 4*f1**3*f2**3*f3*v3 + 3*f1**2*f2**4*f3*v3 - 4*f1**4*f2*f3**2*v3 + \
   8*f1**3*f2**2*f3**2*v3 - 4*f1**2*f2**3*f3**2*v3)/\
 ((f1 - f2)**2*(f1 - f3)**3*(-f2 + f3)**2))
def calculate_delta_parameter_1(f1,f2,f3,v1,v2,v3,dd1,dd3,M):
    return -((-(dd3*f1**5*f2**2) + 2*dd3*f1**4*f2**3 - dd3*f1**3*f2**4 - dd3*f1**4*f2**2*f3 + \
   2*dd1*f1**3*f2**3*f3 + 2*dd3*f1**3*f2**3*f3 - 2*dd1*f1**2*f2**4*f3 - \
   dd3*f1**2*f2**4*f3 + dd3*f1**5*f3**2 - 3*dd1*f1**3*f2**2*f3**2 - \
   dd3*f1**3*f2**2*f3**2 + 2*dd1*f1**2*f2**3*f3**2 - 2*dd3*f1**2*f2**3*f3**2 + \
   dd1*f1*f2**4*f3**2 + 2*dd3*f1*f2**4*f3**2 - dd3*f1**4*f3**3 + dd1*f1**2*f2**2*f3**3 + \
   3*dd3*f1**2*f2**2*f3**3 - 2*dd1*f1*f2**3*f3**3 - 2*dd3*f1*f2**3*f3**3 + \
   dd1*f2**4*f3**3 + dd1*f1**3*f3**4 + dd1*f1*f2**2*f3**4 - 2*dd1*f2**3*f3**4 - \
   dd1*f1**2*f3**5 + dd1*f2**2*f3**5 - 8*f1**2*f2**3*f3*v1 + 6*f1*f2**4*f3*v1 + \
   12*f1**2*f2**2*f3**2*v1 - 8*f1*f2**3*f3**2*v1 - 4*f1**2*f3**4*v1 + 2*f1*f3**5*v1 + \
   2*f1**5*f3*v2 - 4*f1**4*f3**2*v2 + 4*f1**2*f3**4*v2 - 2*f1*f3**5*v2 - 2*f1**5*f3*v3 + \
   8*f1**2*f2**3*f3*v3 - 6*f1*f2**4*f3*v3 + 4*f1**4*f3**2*v3 - 12*f1**2*f2**2*f3**2*v3 + \
   8*f1*f2**3*f3**2*v3)/((f1 - f2)**2*(f1 - f3)**3*(f2 - f3)**2*M))
def calculate_delta_parameter_2(f1,f2,f3,v1,v2,v3,dd1,dd3,M):
    return  -((dd3*f1**5*f2 - dd1*f1**3*f2**3 - 3*dd3*f1**3*f2**3 + dd1*f1**2*f2**4 + \
    2*dd3*f1**2*f2**4 - dd3*f1**5*f3 + dd3*f1**4*f2*f3 - dd1*f1**2*f2**3*f3 + \
    dd3*f1**2*f2**3*f3 + dd1*f1*f2**4*f3 - dd3*f1*f2**4*f3 - dd3*f1**4*f3**2 + \
    3*dd1*f1**3*f2*f3**2 + dd3*f1**3*f2*f3**2 - dd1*f1*f2**3*f3**2 + dd3*f1*f2**3*f3**2 - \
    2*dd1*f2**4*f3**2 - dd3*f2**4*f3**2 - 2*dd1*f1**3*f3**3 + 2*dd3*f1**3*f3**3 - \
    dd1*f1**2*f2*f3**3 - 3*dd3*f1**2*f2*f3**3 + 3*dd1*f2**3*f3**3 + dd3*f2**3*f3**3 + \
    dd1*f1**2*f3**4 - dd1*f1*f2*f3**4 + dd1*f1*f3**5 - dd1*f2*f3**5 + 4*f1**2*f2**3*v1 - \
    3*f1*f2**4*v1 + 4*f1*f2**3*f3*v1 - 3*f2**4*f3*v1 - 12*f1**2*f2*f3**2*v1 + \
    4*f2**3*f3**2*v1 + 8*f1**2*f3**3*v1 - f1*f3**4*v1 - f3**5*v1 - f1**5*v2 - \
    f1**4*f3*v2 + 8*f1**3*f3**2*v2 - 8*f1**2*f3**3*v2 + f1*f3**4*v2 + f3**5*v2 + \
    f1**5*v3 - 4*f1**2*f2**3*v3 + 3*f1*f2**4*v3 + f1**4*f3*v3 - 4*f1*f2**3*f3*v3 + \
    3*f2**4*f3*v3 - 8*f1**3*f3**2*v3 + 12*f1**2*f2*f3**2*v3 - 4*f2**3*f3**2*v3)/\
  ((f1 - f2)**2*(f1 - f3)**3*(f2 - f3)**2*M**2))
def calculate_delta_parameter_3(f1,f2,f3,v1,v2,v3,dd1,dd3,M):
    return -((-2*dd3*f1**4*f2 + dd1*f1**3*f2**2 + 3*dd3*f1**3*f2**2 - dd1*f1*f2**4 - dd3*f1*f2**4 + \
   2*dd3*f1**4*f3 - 2*dd1*f1**3*f2*f3 - 2*dd3*f1**3*f2*f3 + dd1*f1**2*f2**2*f3 - \
   dd3*f1**2*f2**2*f3 + dd1*f2**4*f3 + dd3*f2**4*f3 + dd1*f1**3*f3**2 - dd3*f1**3*f3**2 - \
   2*dd1*f1**2*f2*f3**2 + 2*dd3*f1**2*f2*f3**2 + dd1*f1*f2**2*f3**2 - \
   dd3*f1*f2**2*f3**2 + dd1*f1**2*f3**3 - dd3*f1**2*f3**3 + 2*dd1*f1*f2*f3**3 + \
   2*dd3*f1*f2*f3**3 - 3*dd1*f2**2*f3**3 - dd3*f2**2*f3**3 - 2*dd1*f1*f3**4 + \
   2*dd1*f2*f3**4 - 4*f1**2*f2**2*v1 + 2*f2**4*v1 + 8*f1**2*f2*f3*v1 - 4*f1*f2**2*f3*v1 - \
   4*f1**2*f3**2*v1 + 8*f1*f2*f3**2*v1 - 4*f2**2*f3**2*v1 - 4*f1*f3**3*v1 + 2*f3**4*v1 + \
   2*f1**4*v2 - 4*f1**3*f3*v2 + 4*f1*f3**3*v2 - 2*f3**4*v2 - 2*f1**4*v3 + \
   4*f1**2*f2**2*v3 - 2*f2**4*v3 + 4*f1**3*f3*v3 - 8*f1**2*f2*f3*v3 + 4*f1*f2**2*f3*v3 + \
   4*f1**2*f3**2*v3 - 8*f1*f2*f3**2*v3 + 4*f2**2*f3**2*v3)/\
 ((f1 - f2)**2*(f1 - f3)**3*(f2 - f3)**2*M**3))
def calculate_delta_parameter_4(f1,f2,f3,v1,v2,v3,dd1,dd3,M):
    return  -((dd3*f1**3*f2 - dd1*f1**2*f2**2 - 2*dd3*f1**2*f2**2 + dd1*f1*f2**3 + dd3*f1*f2**3 - \
   dd3*f1**3*f3 + 2*dd1*f1**2*f2*f3 + dd3*f1**2*f2*f3 - dd1*f1*f2**2*f3 + \
   dd3*f1*f2**2*f3 - dd1*f2**3*f3 - dd3*f2**3*f3 - dd1*f1**2*f3**2 + dd3*f1**2*f3**2 - \
   dd1*f1*f2*f3**2 - 2*dd3*f1*f2*f3**2 + 2*dd1*f2**2*f3**2 + dd3*f2**2*f3**2 + \
   dd1*f1*f3**3 - dd1*f2*f3**3 + 3*f1*f2**2*v1 - 2*f2**3*v1 - 6*f1*f2*f3*v1 + \
   3*f2**2*f3*v1 + 3*f1*f3**2*v1 - f3**3*v1 - f1**3*v2 + 3*f1**2*f3*v2 - 3*f1*f3**2*v2 + \
   f3**3*v2 + f1**3*v3 - 3*f1*f2**2*v3 + 2*f2**3*v3 - 3*f1**2*f3*v3 + 6*f1*f2*f3*v3 - \
   3*f2**2*f3*v3)/((f1 - f2)**2*(f1 - f3)**3*(f2 - f3)**2*M**4))
