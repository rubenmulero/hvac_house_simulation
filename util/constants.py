#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The simulation engine contains different type of mathematical formulas. Each formulas, has its own parameters
and constant values. This file contains those values which are constant and they could barely be changed over the time

To improve the understanding of this file, I have divided each set of constant values by their purpose.


@author: Rubén Mulero
"""

import pathlib

# Radiator related constants
N = 1.33
# MAX_P_EL = 9000                         # We choose 9000W to be compatible with Bacher model.
MAX_P_EL = 9
K_RAD = MAX_P_EL / ((50 - 20) ** N)

# Define comfort threshold values
COMFORT_TEMP = 20
M_BAND = 1

# TiTeThTsWithAe model values
C_i = 0.0928
R_is = 1.89
C_s = 0.0549
R_ih = 0.146
C_h = 0.889
R_ie = 0.897
C_e = 3.32
R_ea = 4.38
A_w = 5.75
A_e = 3.87

# Internal gain occupancy Factor (Fac_ig) obtained from ASHRAE Fundamentals 2005 cap 8 Table 4.
# Male-Female mean presence 85%, Metabolic ratio in office building 77W/m2, physical space needed by a human 1.8m^2.
# The result is in W/Person
FACT_ig = 0.85 * 1.8 * 77

# Data folder
DATA_FOLDER = str(pathlib.Path(__file__).parent.absolute()) + '/../data'

# Occupant simulation engine (Dataset used for probabilities, Total N_occupants per room, simulation year.
DATASET_PATH = str(pathlib.Path(__file__).parent.absolute()) + '/../occupant_estimator/data/UNIVERSITY.csv'
N_OCCUPANTS = 12
# A Python tuple containing the years to be simulated in the simulation process.
VALID_YEARS = (2016, 2017, 2018, 2019, )

# A Python tuple containing the future values to take from the given datasets in hours
FUTURE_VALUES = (1, 6, 12, 24)

# Building properties
ROOMS = 10

# Ventilation related constants. Extracted from: ASHRAE 62.1.2013. Table 6.2.2.1. Offices
# v_pers = 2.5                             # l/s.pers
# v_space = 0.3                            # l/m2
#
# V_per is calculated with the maximum people in the current profile defined in the constant N_Occupants
# The V_space is defined with the total space of the building
#
# The Modelisez building has 120m^2 thus:
#
#   0.3l/s.m2 * 120m2 =36l/s=0.036 m3/s
V_ACT_ON = 8     # "08:00"
V_ACT_OFF = 18   # "18:00"
# V_PERS = v_pers * FACTOR * occupants in the current occupant estimation profile
V_PERS = 2.5 * 0.001 * N_OCCUPANTS
V_SPACE = (0.3 * 120) / 1000
V = max(V_PERS, V_SPACE)
R_VENT = 1 / (V * 1.211224)         # V in m3/s; 1.211224 kJ/m3K    --> 1.211224 kJ/m3K
CP = 1.006  # kJ/kgK
DENS = 1.204  # kg/K
# Heat pump coefficient of performance values to calculate the required electricity power.
# The values are in kW, and they were extracted from: https://doi.org/10.1016/j.enbuild.2015.11.014
Q_HP_MIN = 1.
Q_HP_MAX = 100.
Q_HP_BOOST = 8.

# Reward formula configurations and variables
# The reward formula is composed by two main parts. The Comfort part and the Energy (Cost) part.
# Each part has its own configuration depending on the target desired applications.
#
# The first part is aimed to configure the optimal min and max conform values
# with a deviation penalty of 1.5 (semi-exponential).
#
# Defining optimal comfort global values
T_OPT_MIN = 21.0
T_OPT_MAX = 25.0
A = 1  # Optimal comfort reward
n = 1.5  # Deviation exponential value
total_deviation = 5  # Total deviation in degrees allowed in the model. (Min max comfort values).,
# Penalty cost calculation given the total deviation.
B = round(A / total_deviation ** n, 4)

# Defining terminal degree values (terminal state).
TERM_MIN = (T_OPT_MIN - total_deviation) - 2
TERM_MAX = (T_OPT_MAX + total_deviation) + 2

# The second part is aimed to configure the Energy or Cost usage using the Economical and penalty values of the
# target country or region.
#
# Defining optimal cost global values
normalisation_factor = 0.2  # Represent a total of 20% of the total incomes used for paying comfort.
gdp_per_capita = 68007.76  # Gross domestic product per capita. Using values from Denmark. USD/year
currency_exchange = 1.1827  # Dollar --> Euro currency exchange using prices from 2021. 1€ --> 1.1827 Dollars
activation_penalty = 0  # Penalty applied to a target device for using it.
