#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This class is used to calculate the economic costs of different parts of simulation engine.

The costs are based on different external and context values, like the number of building occupants, expected
electricity price, expected weather and so on.

@author: Rubén Mulero
"""

import os
from util.logger import LoggingClass

logger = LoggingClass('simulator_engine').get_logger()
file_name = os.path.basename(__file__)


def calculate_operational_cost(p_p_el, p_electricity_price):
    """
    This method calculates the operational cost (C_op) of the heat pump at timestep T given the current Power usage and
    the electricity price

    :param p_p_el: The current electricity power of the heat pump.
    :param p_electricity_price: The current electricity price in €/kWh (converted from the original value in €/MWh).

    :return the calculated operational cost in €/h
    """

    return p_p_el * p_electricity_price
