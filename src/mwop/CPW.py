####
# Methods for designing coplanar waveguides on multilayer substrates
# author: Phillip Kirwin (pkirwin@ece.ubc.ca)
# date: 2024-02-23
# references: Ashutosh's thesis, Simons ch2
#
#
####

import numpy as np
from numpy import sin,cos,pi,sqrt,exp,cosh,sinh
import mpmath as mpm
from scipy.constants import epsilon_0,mu_0,speed_of_light
ep_0 = epsilon_0
c = speed_of_light

# set the decimal point precision for mpmath. This may be
# needed for when the CPW gap is large compared to one of the 
# layer thicknesses (as can be the case for optical litho CPWs on 220nm SOI)
mpm.mp.dps = 1000

#  calculates the kinetic, geometric, and total line inductances (per unit length) of the CPW. We assume that all substrate layers are non-magnetic, that is, their relative permeability is 1. width and gap must have the same units. Kinetic inductance is calculated with an approximation that is strictly speaking only valid for gap >> width. Using this approximation outside of this limit systematically underestimates the kinetic inductance.
# takes the CPW width and gap, and the sheet inductance of the SC, as arguments.
def inductances(width,gap,sheet_inductance):
    k0 = width/(width + 2*gap) # modulus of the elliptic integral for infinite layer
    k0p = np.sqrt(1-k0**2) # complimentary modulus
    mpm.ellipk(k0**2)
    inductance_geo = mu_0/4 * mpm.ellipk(k0p**2) / mpm.ellipk(k0**2) # magnetic inductance [H/m]
    inductance_kinetic = sheet_inductance / width # kinetic inductance [H/m]
    inductance_tot = inductance_kinetic + inductance_geo # total inductance [H/m]
    return inductance_kinetic, inductance_geo, inductance_tot

# calculates the capacitance per unit length of a semi-infinite layer.
# takes the dielectric constant, and CPW width and gap as arguments. The later must have the same units.
def inf_cap(width,gap,ep_r):
    k = width/(width + 2*gap) # modulus of the elliptic integral for infinite layer
    kp = np.sqrt(1-k**2) # complimentary modulus
    capacitance_inf = 2 * ep_0 * ep_r * mpm.ellipk(k**2)/mpm.ellipk(kp**2)
    return capacitance_inf

# calculates the capacitance per unit length of a finite layer.
# takes the height (r), dielectric constant, and CPW width and gap as arguments. These must have the same units.
# The dielectric constant may be negative.
# the height in r is the distance from the top of the
# layer to the CPW, NOT the actual thickness of
# the layer.
def fin_cap(width,gap,height,ep_r):
    k = mpm.sinh(pi*width/4/height) / sinh(pi*(width+2*gap)/4/height) # modulus of the elliptic integral for finite layer
    kp = kp = np.sqrt(1-k**2) # complementary modulus
    capacitance_fin = 2 * ep_0 * ep_r * mpm.ellipk(k**2)/mpm.ellipk(kp**2)
    return capacitance_fin

# computes the total capacitance for all layers.
def total_cap(width,gap,substrate:dict,superstrate:dict) -> float:
    # loop and add substrate capacitances
    C_sub = 0
    height=0
    for ii in range(len(substrate)):

        layer = "layer " + str(ii+1)
        next_layer = "layer " + str(ii+2)
        ep_r = substrate[layer]["ep_r"]
        thickness = substrate[layer]["thickness"]
        height = height + thickness
        if next_layer in substrate:
            next_ep_r = substrate[next_layer]["ep_r"]
            reduced_ep_r = ep_r - next_ep_r
            C_layer = fin_cap(width,gap,height,reduced_ep_r)
        else:
            C_layer = inf_cap(width,gap,ep_r)
            
        C_sub = C_sub + C_layer

    # loop and add superstrate capacitances
    C_super = 0
    height=0
    for ii in range(len(superstrate)):
        layer = "layer " + str(ii+1)
        next_layer = "layer " + str(ii+2)
        ep_r = superstrate[layer]["ep_r"]
        thickness = superstrate[layer]["thickness"]
        height = height + thickness
        if next_layer in superstrate:
            next_ep_r = superstrate[next_layer]["ep_r"]
            reduced_ep_r = ep_r - next_ep_r
            C_layer = fin_cap(width,gap,height,reduced_ep_r)
        else:
            C_layer = inf_cap(width,gap,ep_r)
            
        C_super = C_super + C_layer
    #sum up
    C_tot = C_sub + C_super
    return C_tot

# calculate characteristic impedance,including the effects of kinetic inductance
def calc_Z0(width,gap,sheet_inductance,substrate:dict,superstrate:dict):
    capacitance = total_cap(width,gap,substrate,superstrate)
    __,__,inductance = inductances(width,gap,sheet_inductance)
    Z0 = np.sqrt(inductance/capacitance)
    return Z0

# calculate dielectric constant, which includes the effects of kinetic inductance.
def calc_ep_eff(width,gap,sheet_inductance,substrate:dict,superstrate:dict):
    capacitance = total_cap(width,gap,substrate,superstrate)
    __,__,inductance = inductances(width,gap,sheet_inductance)
    ep_eff = c**2 * inductance * capacitance
    return ep_eff