import numpy as np

import ross as rs
from ross.faults import *
from ross.units import Q_
from ross.probe import Probe

# uncomment the lines below if you are having problems with plots not showing
# import plotly.io as pio
# pio.renderers.default = "notebook"

"""Create example rotor with given number of elements."""
steel2 = rs.Material(name="Steel", rho=7850, E=2.17e11, Poisson=0.2992610837438423)
#  Rotor with 6 DoFs, with internal damping, with 10 shaft elements, 2 disks and 2 bearings.
i_d = 0
o_d = 0.019
n = 33

# fmt: off
L = np.array(
        [0  ,  25,  64, 104, 124, 143, 175, 207, 239, 271,
         303, 335, 345, 355, 380, 408, 436, 466, 496, 526,
         556, 586, 614, 647, 657, 667, 702, 737, 772, 807,
         842, 862, 881, 914]
         )/ 1000
# fmt: on

L = [L[i] - L[i - 1] for i in range(1, len(L))]

shaft_elem = [
    rs.ShaftElement6DoF(
        material=steel2,
        L=l,
        idl=i_d,
        odl=o_d,
        idr=i_d,
        odr=o_d,
        alpha=8.0501,
        beta=1.0e-5,
        rotary_inertia=True,
        shear_effects=True,
    )
    for l in L
]

Id = 0.003844540885417
Ip = 0.007513248437500

disk0 = rs.DiskElement6DoF(n=12, m=2.6375, Id=Id, Ip=Ip)
disk1 = rs.DiskElement6DoF(n=24, m=2.6375, Id=Id, Ip=Ip)

kxx1 = 4.40e5
kyy1 = 4.6114e5
kzz = 0
cxx1 = 27.4
cyy1 = 2.505
czz = 0
kxx2 = 9.50e5
kyy2 = 1.09e8
cxx2 = 50.4
cyy2 = 100.4553

bearing0 = rs.BearingElement6DoF(
    n=4, kxx=kxx1, kyy=kyy1, cxx=cxx1, cyy=cyy1, kzz=kzz, czz=czz
)
bearing1 = rs.BearingElement6DoF(
    n=31, kxx=kxx2, kyy=kyy2, cxx=cxx2, cyy=cyy2, kzz=kzz, czz=czz
)

rotor = rs.Rotor(shaft_elem, [disk0, disk1], [bearing0, bearing1])

"""Inserting a mass and phase unbalance and defining the local response."""
massunbt = np.array([5e-4, 0])
phaseunbt = np.array([-np.pi / 2, 0])
probe1 = Probe(14, 0)
probe2 = Probe(22, 0)

"""Calculate the response when a rotor makes contact with a casing or stator."""
rubbing = rotor.run_rubbing(
    dt=0.0001,
    tI=0,
    tF=5,
    deltaRUB=7.95e-5,
    kRUB=1.1e6,
    cRUB=40,
    miRUB=0.3,
    posRUB=12,
    speed=Q_(1200, "RPM"),
    unbalance_magnitude=massunbt,
    unbalance_phase=phaseunbt,
    print_progress=False,
)

"""Plots the time response for the two given probes."""
results = rubbing.run_time_response()
results.plot_1d([probe1, probe2]).show()