####
# Methods for designing interdigital capacitors on multilayer substrates.
# author: Phillip Kirwin
# date: 2024-02-01
# references: Rui and Dias 2004, Sensors and Actuators A: Physical
#
#
####

import numpy as np
from mpmath import sin,cos,pi,sqrt,exp,cosh
import mpmath as mpm
from scipy.constants import epsilon_0
ep_0 = epsilon_0

# set the decimal point precision for mpmath. This is needed for when the IDC wavelength is large compared to one of the layer thicknesses (as is typically the case for 220nm SOI)
mpm.mp.dps = 1000
# calculates the interior capacitance of a semi-infinite layer.
# takes the dielectric constant of the layer, and eta, the metallization ratio
# as arguments. 
def inf_interior_cap(ep_r,eta,length):   
    k_I_inf = sin(pi/2 * eta) # modulus
    k_I_inf_p = sqrt(1-k_I_inf**2) # complementary modulus
    C_I = ep_0*ep_r*length*mpm.ellipk(k_I_inf**2)/mpm.ellipk(k_I_inf_p**2)
    return C_I

# calculates the exterior capacitance of a semi-infinite layer.
# takes the dielectric constant of the layer, and eta, the metallization ratio
# as arguments. 
def inf_exterior_cap(ep_r,eta,length):   
    k_E_inf = 2*sqrt(eta)/(1+eta) # modulus
    k_E_inf_p = sqrt(1-k_E_inf**2) # complementary modulus
    C_E = ep_0*ep_r*length*mpm.ellipk(k_E_inf**2)/mpm.ellipk(k_E_inf_p**2)
    return C_E

# calculates the interior capacitance of a finite layer.
# takes the (reduced) dielectric constant of the layer, 
# eta, the metallization ratio, and the height-to-period ratio r
# as arguments. The dielectric constant may be negative.
# the height in r is the distance from the top of the
# layer to the electrodes, NOT the actual thickness of
# the layer.
def fin_interior_cap(ep_r,eta,r,length):   
    q = exp(-4*pi*r)
    k = (mpm.jtheta(2,0,q)/mpm.jtheta(3,0,q))**2
    t2 = mpm.ellipfun('sn',mpm.ellipk(k**2)*eta,k)
    t4 = 1/k
    k_I = t2*sqrt( (t4**2 - 1) / (t4**2 - t2**2) ) # modulus
    k_I_p = sqrt(1-k_I**2) # complementary modulus
    C_I = ep_0*ep_r*length*mpm.ellipk(k_I**2)/mpm.ellipk(k_I_p**2)
    return C_I


# calculates the exterior capacitance of a finite layer.
# takes the (reduced) dielectric constant of the layer, 
# eta, the metallization ratio, and the height-to-period ratio r
# as arguments. The dielectric constant may be negative.
# the height in r is the distance from the top of the
# layer to the electrodes, NOT the actual thickness of
# the layer.
def fin_exterior_cap(ep_r,eta,r,length):   
    t3 = mpm.cosh(pi*(1-eta)/8/r)
    t4 = mpm.cosh(pi*(1+eta)/8/r)
    k_E = 1/t3 * sqrt( (t4**2 - t3**2) / (t4**2 - 1) )# modulus
    k_E_p = sqrt(1-k_E**2) # complementary modulus
    C_E = ep_0*ep_r*length*mpm.ellipk(k_E**2)/mpm.ellipk(k_E_p**2)
    return C_E

# computes the total interior capacitance of all layers.
def total_interior_cap(eta,wavelength,length,substrate:dict,superstrate:dict) -> float:
    
    # loop and add substrate capacitances
    C_I_sub = 0
    height=0
    for ii in range(len(substrate)):

        layer = "layer " + str(ii+1)
        next_layer = "layer " + str(ii+2)
        ep_r = substrate[layer]["ep_r"]
        thickness = substrate[layer]["thickness"]
        height = height + thickness
        r = height/wavelength
        if next_layer in substrate:
            next_ep_r = substrate[next_layer]["ep_r"]
            reduced_ep_r = ep_r - next_ep_r
            C_I_layer = fin_interior_cap(reduced_ep_r,eta,r,length)
        else:
            C_I_layer = inf_interior_cap(ep_r,eta,length)
            
        C_I_sub = C_I_sub + C_I_layer

    # loop and add superstrate capacitances
    C_I_super = 0
    height=0
    for ii in range(len(superstrate)):
        layer = "layer " + str(ii+1)
        next_layer = "layer " + str(ii+2)
        ep_r = superstrate[layer]["ep_r"]
        thickness = superstrate[layer]["thickness"]
        height = height + thickness
        r = height/wavelength
        if next_layer in superstrate:
            next_ep_r = superstrate[next_layer]["ep_r"]
            reduced_ep_r = ep_r - next_ep_r
            C_I_layer = fin_interior_cap(reduced_ep_r,eta,r,length)
        else:
            C_I_layer = inf_interior_cap(ep_r,eta,length)
            
        C_I_super = C_I_super + C_I_layer
    #sum up
    C_I_tot = C_I_sub + C_I_super
    return C_I_tot


# computes the total exterior capacitance of all layers.
def total_exterior_cap(eta,wavelength,length,substrate:dict,superstrate:dict) -> float:
    
    # loop and add substrate capacitances
    C_E_sub = 0
    height=0
    for ii in range(len(substrate)):

        layer = "layer " + str(ii+1)
        next_layer = "layer " + str(ii+2)
        ep_r = substrate[layer]["ep_r"]
        thickness = substrate[layer]["thickness"]
        height = height + thickness
        r = height/wavelength
        if next_layer in substrate:
            next_ep_r = substrate[next_layer]["ep_r"]
            reduced_ep_r = ep_r - next_ep_r
            C_E_layer = fin_exterior_cap(reduced_ep_r,eta,r,length)
        else:
            C_E_layer = inf_exterior_cap(ep_r,eta,length)
            
        C_E_sub = C_E_sub + C_E_layer

    # loop and add superstrate capacitances
    C_E_super = 0
    height=0
    for ii in range(len(superstrate)):
        layer = "layer " + str(ii+1)
        next_layer = "layer " + str(ii+2)
        ep_r = superstrate[layer]["ep_r"]
        thickness = superstrate[layer]["thickness"]
        height = height + thickness
        r = height/wavelength
        if next_layer in superstrate:
            next_ep_r = superstrate[next_layer]["ep_r"]
            reduced_ep_r = ep_r - next_ep_r
            C_E_layer = fin_exterior_cap(reduced_ep_r,eta,r,length)
        else:
            C_E_layer = inf_exterior_cap(ep_r,eta,length)
        C_E_super = C_E_super + C_E_layer
    #sum up
    C_E_tot = C_E_sub + C_E_super
    return C_E_tot


# compute total capacitance of the IDC
def IDC_total_cap(geometry:dict,substrate:dict,superstrate:dict)->float:
    length = geometry["length"]
    gap = geometry["gap"]
    width = geometry["width"] 
    Nfingers = geometry["Nfingers"]
    eta = width / (width + gap)
    wavelength = 2*(width + gap)

    C_I_tot = total_interior_cap(eta,wavelength,length,substrate,superstrate)
    C_E_tot = total_exterior_cap(eta,wavelength,length,substrate,superstrate)
    
    C_tot_IDC = (Nfingers-3)*C_I_tot/2 + 2*C_I_tot*C_E_tot/(C_I_tot+C_E_tot)

    return C_tot_IDC

# computes the coupling quality factor due to a coupling capacitor for a half-wave TL resonator open at both ends and coupled to a feedline in a hanger configuration.
def coupling_Q_halfwave(Zr,Z0,capacitance,fr,n=1):
    Qc = n*pi/Zr/Z0/(2*pi*fr*capacitance)**2
    return Qc

# computes the capacitance due to a coupling Qc for a half-wave TL resonator open at both ends and coupled to a feedline in a hanger configuration. inverse of above function.
def coupling_cap_halfwave(Zr,Z0,Qc,fr,n=1):
    capacitance = sqrt(n*pi/Zr/Z0/Qc)/(2*pi*fr)
    return capacitance