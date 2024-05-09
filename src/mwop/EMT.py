####
# Methods for calculation of the dielectric constant using effective medium theory
# author: Phillip Kirwin
# date: 2024-05-08
# references: Bruggeman 1935, https://en.wikipedia.org/wiki/Effective_medium_approximations
#
#
####


import numpy as np


def EMT_epr(ep1:float,ep2:float,p1:float,p2:float) -> float:
    """Function that returns the effective dielectric constant of a
    compound material with two phases.

    arguments:
    ep1: dielectric constant of medium 1
    ep2: dielectric constant of medium 2
    p1: volume portion of medium 1
    p2: volume portion of medium 2

    returns:
    epr: the effective dielectric constant of the compound medium
    """
    if(p1+p2 != 1):
        raise Exception("p1 and p2 must sum to 1.")
    
    Hb = (3*p1-1)*ep1 + (3*p2-1)*ep2
    ep_eff = 1/4 * (Hb + np.sqrt(Hb**2 + 8*ep1*ep2))
    return ep_eff

