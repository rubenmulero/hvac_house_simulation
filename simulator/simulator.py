#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a class implementation of the Simulator engine

The idea of this class it to have an implementation to execute the simulator in timesteps and fit it with the
OpenAIGym library.


@author: Rubén Mulero
"""

import calendar
import copy
import random

import arrow
import numpy as np
import pandas as pd

from context import csv_parser, numpy_parser
# Simulation equations
from model import equations, radiator_heat_pump, costs, ventilation
# Occupant estimator methods
from occupant_estimator.occusim import utils, simulator
# Constant values used for formulas, and other purposes
from util.constants import C_i, R_is, C_s, R_ih, C_h, R_ie, C_e, R_ea, A_w, A_e, FACT_ig, V_ACT_ON, V_ACT_OFF, \
    FUTURE_VALUES, DATASET_PATH, N_OCCUPANTS, DATA_FOLDER, VALID_YEARS
# Util libraries
from util.logger import LoggingClass

logger = LoggingClass('simulator_engine').get_logger()


class Simulator:
    """
    This class contains the POO implementation of the defined simulation environment. The main idea is to use it with
    external programs like OpenAI Gym

    """

    def __init__(self, p_occupant_profile=None, p_time_gap=10, p_end_date=None, p_simulated_years=(2016, )):
        self.timestep = 0
        self.time_gap = p_time_gap  # number of seconds for each simulated timestemp
        self.year = p_simulated_years if set(p_simulated_years).issubset(VALID_YEARS) else (2016, )
        # Loading context data (exo data)
        self.max_timesteps = self._get_simulator_timesteps(self.year, p_end_date=p_end_date)
        self.weather_data, self.electricity_data = self._load_context_data()
        self.is_custom_occupant_profile = True if p_occupant_profile else False
        self.total_occupations_hourly = self._load_occupant_data(p_occupant_profile)
        # self.total_occupied_slots = len(self.total_occupations_hourly[np.where(self.total_occupations_hourly > 0)])
        # Initial values of the model temperatures
        self.t_s = self.t_s_1 = 20.0  # Sensor temp
        self.t_h = self.t_h_1 = 20.0  # Heater temp
        self.t_i = self.t_i_1 = 20.0  # Interior temp
        self.t_e = self.t_e_1 = 20.0  # Envelope temp
        self.t_a = self.weather_data['air_temperature'][0]  # Current ambient temperature (outdoor temperature)
        # self.t_a_1 = self.previous_weather_data['air_temperature'].iloc[-1]
        self.t_a_1 = self.weather_data['air_temperature'][
            0]  # Default value of "yesterday" weather outdoor temperature.
        # Setting numpy error to raise if there are problems in the simulation process
        np.seterr('raise')
        print("====================================================")
        print(f"Simulation instantiated \n"
              "------------------------ \n"
              f"current time gap is {p_time_gap} \n"
              f"current occupant profile is {p_occupant_profile} \n"
              f"current end date is {p_end_date} \n"
              f"total years to simulate {self.year}")
        print("====================================================")

    ######
    # Public methods
    ######

    def simulate_step(self, p_act_hp, p_t_sup):
        """
        This is the main method of this class. This will simulate only one step in the simulation process given the
        current timestep, and action of the heat pump pass as a parameters of this method.

        :param p_act_hp: 0/1 action to use or not the installed heat pump
        :param p_t_sup: the setpoint temperature wanted to use.

        :return:
        """
        # Sanity checks of the given values and statuses
        if p_act_hp not in [0, 1] or not 20.0 <= p_t_sup <= 50.0:
            # The given values by the agent are incorrect, raising an error.
            logger.error("Error in values given by the agent. ACT_HP: {act_hp}, T_sup: {t_sup}".
                         format(act_hp=p_act_hp, t_sup=p_t_sup))
        if self.timestep > self.max_timesteps:
            logger.error(f'We are under an invalid timestep value: {self.timestep}')
        # Increasing simulation timestep. We are doing this here because we have already the initial step from reset().
        self.timestep += 1
        if self.timestep >= self.max_timesteps:
            logger.info("Maximum simulation timesteps reached. Resetting to 0")
            self.timestep = 0
        # Everything is correct, we are starting to simulate the process
        # Getting the full timestamp
        current_date = self._get_current_timestamp()  # Obtaining current timestamp
        # Getting external non-controllable data.
        exogenous_data = self._get_exogenous_data()
        # Extracting data from the exogenous values
        i_s = exogenous_data.get('i_s', 0.0)
        current_el_price = exogenous_data.get('current_el_price', 0.0)
        occupancy = exogenous_data.get('occupancy', 0)
        # Calculating Hydroponic heating radiator and Power Electricity given the heat pump actuation
        q_h, p_el, cop = radiator_heat_pump.calculate_cop(p_act_hp, p_t_sup, self.t_i_1, self.t_a)
        # Getting the operational cost of the heat pump
        C_op = costs.calculate_operational_cost(p_el, current_el_price)
        # Calculating Internal gains based on the current persons in the building
        Q_ig = (FACT_ig * occupancy) * 0.001  # in kW
        # Getting the current daily hour based on the current timestep to activate the ventilation system
        daily_hour = current_date.hour
        act_vent = 1 if V_ACT_ON <= daily_hour <= V_ACT_OFF else 0
        Q_Vent = ventilation.calculate_ventilation_resistance_variance(self.t_a_1, self.t_i_1, act_vent)
        # Getting updated temperatures from simulator (state variables).
        self.t_s = equations.calculate_t_s(self.time_gap, C_s, R_is, self.t_i_1, self.t_s_1)
        self.t_h = equations.calculate_t_h(self.time_gap, C_h, R_ih, q_h, self.t_i_1, self.t_h_1)
        self.t_i = equations.calculate_t_i(self.time_gap, C_i, R_is, R_ih, R_ie, Q_ig, self.t_s_1, self.t_i_1,
                                           self.t_h_1,
                                           self.t_e_1, A_w, i_s, Q_Vent)
        self.t_e = equations.calculate_t_e(self.time_gap, C_e, R_ie, R_ea, self.t_i_1, self.t_a_1, self.t_e_1, A_e, i_s)
        # Sanity check to avoid bad, missing calculations.
        if self._check_is_nan(self.t_s, self.t_h, self.t_i, self.t_e):
            print("Nan values detected, something estrange was happened")
            logger.error("Nan values detected, something estrange was happened")
            print(
                f"Current values are: T_s: {self.t_s}, T_h: {self.t_h}, T_i: {self.t_i}, T_e: {self.t_e}, Q_h = {q_h}")
            raise Exception
        # Reassign old values 5 values
        self.t_s_1 = self.t_s
        self.t_h_1 = self.t_h
        self.t_i_1 = self.t_i
        self.t_e_1 = self.t_e
        self.t_a_1 = self.t_a
        # Building return data in np format and returning it
        data = {
            "t_s": self.t_s,
            "t_h": self.t_h,
            "t_i": self.t_i,
            "t_e": self.t_e,
            "t_a": self.t_a,
            "c_op": C_op,                                       # Operational Cost
            "cop": cop,                                         # Coefficient of performance
            "i_s": i_s,                                         # Solar radiation
            "current_el_price": current_el_price,
            "occupancy": occupancy,
            "current_date": current_date,
            "p_el": p_el,                                       # Used electricity power
            "q_h": q_h,
            "q_ig": Q_ig,
            "act_vent": act_vent                                # Ventilation actuation
        }
        # Adding future values based on the defined FUTURE_VALUES constant value (if configured)
        for i in FUTURE_VALUES:
            data[f'next_i_s_{i}'] = exogenous_data.get(f'next_i_s_{i}', 0.0)                # next Solar radiation
            data[f'next_el_price_{i}'] = exogenous_data.get(f'next_el_price_{i}', 0.0)      # next electricity price
            data[f'next_occupancy_{i}'] = exogenous_data.get(f'next_occupancy_{i}', 0)      # next expected occupancy
        return data

    def reset_simulator(self, p_random_reset=False):
        """
        Performs a reset of the current simulation process. The method will reset the current timestep and the
        temperatures.

        :param p_random_reset: Randomizes the starting timestep value.

        :return: A numpy object containing the reset values. None if something went wrong.
        """

        print("A simulation reset was executed")
        logger.warning("Current values before reset the simulation engine are:"
                       f"Simulation engine timestep: {self.timestep}"
                       f"Sensor temperature: {self.t_s}"
                       f"Heater temperature: {self.t_h}"
                       f"Indoor temperature: {self.t_i}"
                       f"Envelope temperature: {self.t_e}"
                       f"Current Ambient temperature: {self.t_a}"
                       f"Previous Ambient temperature: {self.t_a_1}")
        logger.info("Resetting current environment with default values")
        print("====================================================")
        print("Reset simulator engine")
        print("====================================================")
        self.timestep = random.randrange(0, self.max_timesteps) if p_random_reset else 0
        print(f"Current initial timestep set to {self.timestep}")
        if self.is_custom_occupant_profile:
            print("We have custom occupant profile. We are not to random it")
        else:
            self.total_occupations_hourly = self._load_occupant_data(None)
        self.t_s = self.t_s_1 = 20.0
        self.t_h = self.t_h_1 = 20.0
        self.t_i = self.t_i_1 = 20.0
        self.t_e = self.t_e_1 = 20.0
        # Getting weather data from
        s = int(self.timestep / (60 / self.time_gap))
        self.t_a = self.weather_data['air_temperature'][s]
        self.t_a_1 = self.weather_data['air_temperature'][s]
        # Building return values.
        if not p_random_reset and self.timestep == 0 and self.t_s == self.t_h == self.t_i == self.t_e == 20.0 or \
                p_random_reset and 0 <= self.timestep < self.max_timesteps and \
                self.t_s == self.t_h == self.t_i == self.t_e == 20.0:
            # building res data
            initial_date = self._get_current_timestamp()
            initial_exo_data = self._get_exogenous_data()
            i_s = initial_exo_data.get('i_s', 0.0)
            current_el_price = initial_exo_data.get('current_el_price', 0.0)
            occupancy = initial_exo_data.get('occupancy', 0)
            Q_ig = (FACT_ig * occupancy) * 0.001  # in kW
            act_vent = 1 if V_ACT_ON <= initial_date.hour <= V_ACT_OFF else 0
            # Return data
            res = {
                "t_s": self.t_s,
                "t_h": self.t_h,
                "t_i": self.t_i,
                "t_e": self.t_e,
                "t_a": copy.copy(self.t_a),
                "c_op": 0,                                          # Operational Cost is 0 because reset
                "cop": 0,                                           # Coefficient of Performance
                "i_s": i_s,                                         # Solar irradiance
                "current_el_price": current_el_price,
                "occupancy": occupancy,
                "current_date": initial_date,
                "p_el": 0,                                          # Used electricity power is 0 because reset
                "q_h": 0,                                           # q_h value is 0 because reset
                "q_ig": Q_ig,
                "act_vent": act_vent
            }
            # Adding future data
            for i in FUTURE_VALUES:
                res[f'next_i_s_{i}'] = initial_exo_data.get(f'next_i_s_{i}', 0.0)              # next Solar radiation
                res[f'next_el_price_{i}'] = initial_exo_data.get(f'next_el_price_{i}', 0.0)    # next electricity price
                res[f'next_occupancy_{i}'] = initial_exo_data.get(f'next_occupancy_{i}', 0)    # next expected occupancy

            logger.warning("Current environment was reset. The current values are:"
                           f"Simulation engine timestep: {self.timestep}"
                           f"Sensor temperature: {self.t_s}"
                           f"Heater temperature: {self.t_h}"
                           f"Indoor temperature: {self.t_i}"
                           f"Envelope temperature: {self.t_e}"
                           f"Current Ambient temperature: {self.t_a}"
                           f"Previous Ambient temperature: {self.t_a_1}"
                           f"Current occupancy: {occupancy}"
                           f"current initial date is: {initial_date}"
                           )
        else:
            res = None
            logger.error("The reset operation failed. Please, check your implementation")

        return res

    # Getters and setters
    def get_t_s(self):
        """
        Gets the current value of t_s

        :return: T_s value
        """
        return self.t_s

    def get_t_h(self):
        """
        Gets the current value of t_h

        :return: t_h value
        """
        return self.t_h

    def get_t_i(self):
        """
        Gets the current value of t_i

        :return: t_i value
        """
        return self.t_i

    def get_t_e(self):
        """
        Gets the current value of t_e

        :return: t_e value
        """
        return self.t_e

    def get_t_a(self):
        """
        Gets the current value of t_a

        :return: t_a value
        """
        return self.t_a

    def get_timestep(self):
        """
        Gets the current value of simulator timestep

        :return: timestep value
        """
        return self.timestep

    def get_max_occupants(self):
        """
        Given the current occupant distribution. This method gives the maximum occupants in the current profile.

        p_current_year: The current simulation year

        :return: An integer containing the maximum occupants in the generated occupant distribution.
        """

        return int(self.total_occupations_hourly.max())

    def get_total_occupied_slots(self):
        """
        Given the current occupant distribution. This method gives the total occupied slots.

        :return:
        """
        return self.total_occupied_slots

    ######
    # Private methods
    ######

    def _get_current_timestamp(self):
        """
        Knowing the years to be simulated and the current timestep, this method returns the current date of the
        simulation process.

        YEARS, are given by the tuple in constants file.

        :return: An Arrow based timestamp containing the data to be analysed
        """
        res = None
        # Getting the current yearly date.
        yearly_day = int(self.timestep / (24 * 60 * (60 / self.time_gap))) + 1
        # Getting the month
        day_diff = yearly_day
        for year in self.year:
            for i in range(1, 13):
                month_days = calendar.monthrange(year, i)[1]
                day_diff = day_diff - month_days
                if day_diff <= 0:
                    # We found the current month and day of the year.
                    current_day = calendar.monthrange(year, i)[1] if day_diff == 0 else month_days + day_diff
                    current_month = i
                    current_year = year
                    # Getting the current second, minute and daily hour.
                    current_seconds = (self.timestep * self.time_gap) % 60
                    current_minute = (int((self.timestep * self.time_gap) / 60)) % 60
                    current_hour = (int((self.timestep * self.time_gap) / 3600)) % 24
                    # Building the response using Arrow object
                    try:
                        res = arrow.get(current_year, current_month, current_day, current_hour,
                                        current_minute, current_seconds)
                    except Exception as e:
                        print("Error detected in conversion to arrow")
                        print("Values are")
                        print("=========")
                        print("Current year: {current_year}".format(current_year=current_year))
                        print("Current month: {current_month}".format(current_month=current_month))
                        print("Current day: {current_day}".format(current_day=current_day))
                        print("Current hour: {current_hour}".format(current_hour=current_hour))
                    break
            if res:
                break

        return res

    def _load_context_data(self):
        """
        This method will load the context data  (Weather and electricity)

        :return: A list of Panda Dataframes containing the loaded data
        """
        parser = csv_parser.CSVParser(DATA_FOLDER)
        logger.info("Loading context data...")
        complete_weather_data = pd.DataFrame()
        complete_electricity_data = pd.DataFrame()
        for year in self.year:
            # Parsing data using the CSV parser.
            weather_data = parser.parse_data(f'weather_data_{year}_minute.csv')
            electricity_data = parser.parse_data(f'electricity_price_{year}.csv')
            # Electricity data has an extra column. Removing it.
            del electricity_data['3B']
            del electricity_data['date']
            # Validating the loaded data, to ensure that it is ok.
            # Weather data is given in a matrix of i-days, j-days
            len_weather_data = weather_data.shape[0] * (60 / self.time_gap)
            # Electricity data contains the 31 day of the previous year.
            len_electricity_data = ((electricity_data.shape[0]) * (electricity_data.shape[1])) * (3600 / self.time_gap)
            # Checking if each value has the same length
            if len_weather_data == len_electricity_data:
                # Data is ok and fits with the year
                complete_weather_data = pd.concat([complete_weather_data, weather_data], axis=0, ignore_index=True)
                complete_electricity_data = pd.concat([complete_electricity_data, electricity_data], axis=0,
                                                      ignore_index=True)
            else:
                logger.error(f"The data loaded has different length, thus is not correct. review it \n"
                             f"Weather data length: {len_weather_data} \n"
                             f"Electricity data length: {len_electricity_data} \n"
                             f"Yearly data len: {self.max_timesteps}")
                raise Exception
        logger.info("Contextual / Exogenous data loaded successfully into memory")
        return complete_weather_data, complete_electricity_data

    def _load_occupant_data(self, p_occupant):
        """
        This method will load occupancy data. In case of not having any data stored in the system
        the method will call to the occupant estimation algorithm.

        @:param p_occupant: the occupant file

        :return: An occupant pattern containing the total
        """
        parser = numpy_parser.NUMPYParser(DATA_FOLDER)
        logger.info("Loading occupant data...")
        occupant_data = parser.parse_data(p_occupant)
        # Sanity check
        if occupant_data is None or int(max(occupant_data[0])) != N_OCCUPANTS \
                or len(occupant_data[0]) != sum(
            [(366 if calendar.isleap(year) else 365) * 24 for year in self.year]):  # Same hourly length
            logger.warning("No valid occupant data provided or not occupant data file found. Generating an occupant "
                           "profile")
            occupant_data = self._generate_people_estimation()
        else:
            # We have valid custom data
            occupant_data = occupant_data[0]
        logger.info("Contextual / Exogenous data loaded successfully into memory")
        return occupant_data

    def _generate_people_estimation(self):
        """
        This method launches several instances of the people estimation algorithm to simulate several rooms
        in the building. The ROOM values is the total number of rooms of the building. The method will produce a vector
        of hourly occupant by room and then perform a sum of the values.

        :return: a nd array containing the total number of building occupants per hour.

        """
        logger.info("Generating occupant simulation")
        list_of_rooms = []
        # Making an iteration per year.
        for year in self.year:
            # One occupant estimation per year.
            occupant_estimation = self._calculate_people_estimation(year)
            list_of_rooms.append(occupant_estimation)
        # List is flattened to improve the occupant search procedure.
        list_of_rooms_flattened = [item for sublist in list_of_rooms for item in sublist]
        # list_of_rooms = np.asarray(list_of_rooms_flattened, dtype=object)  # object because leap years has different shapes.
        res = np.asarray(list_of_rooms_flattened)
        # total_occupations_hourly = list_of_rooms.sum(axis=0)
        logger.info("Occupant simulation data generated")
        return res

    def _calculate_people_estimation(self, p_year):
        """
        This method returns a numpy array containing the number of people
        simulated with the J. Page occupant simulation algorithm.

        :param p_year The year to be simulated.

        :return: number of occupants in each hour of the year
        """
        schedule = utils.read_cea_schedule(DATASET_PATH)
        # Extract the schedule building and the monthly multiplier (Probability per month)
        daily_schedule_building = schedule[0]
        monthly_multiplier = schedule[1]['MONTHLY_MULTIPLIER']
        # Number of day in schedule
        # days_in_schedule = len(list(set(daily_schedule_building['DAY'])))  # WEEKDAY, SATURDAY, SUNDAY
        # Extract the schedule for people occupancy.
        array = daily_schedule_building['OCCUPANCY']
        # Building the user probability distribution
        profile = {
            'WEEKDAY': array[0 * 24:(0 + 1) * 24],
            'SATURDAY': array[1 * 24:(1 + 1) * 24],
            'SUNDAY': array[2 * 24:(2 + 1) * 24],
        }
        # Executing simulation process.
        occupancy_simulation = simulator.simulate_occupancy(profile, N_OCCUPANTS, p_year, monthly_multiplier)
        return occupancy_simulation

    def _check_is_nan(self, p_t_s, p_t_h, p_t_i, p_t_e):
        """
        Checks if the given values contains NaN values. Merely a Sanity check method

        :return: True or False depending on there is NaN values
        """
        res = False
        if np.isnan(p_t_s) or np.isnan(p_t_h) or np.isnan(p_t_i) or np.isnan(p_t_e):
            res = True
        return res

    def _get_simulator_timesteps(self, p_year, p_end_date=None):
        """
        Returns the number of timesteps needed in the current year. A timesteps is equal to 10 yearly seconds.
        We extract the number of days from the given year every 10 seconds:

        For example, total seconds in intervals of 10 seconds for the year 2019:

        365 * 24 * 60 * 6 = 3.153.600 timesteps
        Each timestep = 10 seconds.

        --------------------------

        :param p_year The given year in a tuple.
        :param p_end_date Maximum date in which the simulation will work (optional).

        :return: Number of timesteps needed to perform in the simulator
        :type: integer
        """
        if isinstance(p_year, tuple):
            timesteps = 0
            end_reached = False
            for year in p_year:
                days = 0
                for i in range(1, 13):
                    if p_end_date and p_end_date.year == year and p_end_date.month == i:
                        days += p_end_date.day
                        end_reached = True
                        break
                    days += calendar.monthrange(year, i)[1]
                timesteps += (days * 24 * 60 * (60 / self.time_gap))
                if end_reached:
                    break
        else:
            raise Exception("The given year parameter is not a complete list")
        logger.info("Total timesteps to be simulated: {max_timesteps}"
                    "\n Number of years to be simulated: {number_years}".format(max_timesteps=timesteps,
                                                                                number_years=len(p_year)))
        return int(timesteps)

    def _get_exogenous_data(self):
        """
        Using the current simulation timestep, this method will get the current exogenous data.

        This will be formed by: Ambient temperature (t_a), Global Horizontal Irradiance (GHI),
        current electricity price (current_el_price), next electricity price (net_el_price) and occupancy

        :param: p_p_current_date the current simulation date.

        :return: A Python dictionary containing the desired values.
        :type:  Python Dictionary
        """
        # Getting current values based on the current date and current timestep
        res = {}
        # Getting current yearly timestep in seconds --> 1 timestep = 10seconds in the default approach
        s = int(self.timestep / (60 / self.time_gap))
        self.t_a = self.weather_data['air_temperature'][s]
        # We do not want to use the radiation exchanges with the exterior (they give us negative values)
        res['i_s'] = max(self.weather_data['GHI'][s], 0) * 0.001
        hour = int(self.timestep / (3600 / self.time_gap))
        d = int(hour / 24)  # Electricity days
        h = hour % 24  # Electricity hours
        res['current_el_price'] = self.electricity_data.iloc[d, h] / 1000  # Electricity price is in EUR/MWh --> EUR/kWh
        # Obtaining the current occupancy year based on the current timestamp
        res['occupancy'] = int(self.total_occupations_hourly[hour])
        # Getting future data based on the given FUTURE_VALUES constant value
        for i in FUTURE_VALUES:
            # Getting next timestep
            next_timestep = int(self.timestep + ((3600 * i) / self.time_gap))
            if next_timestep >= self.max_timesteps:
                diff = next_timestep - self.max_timesteps
                next_timestep = 0 + diff
            next_s = int(next_timestep / (60 / self.time_gap))  # Rounding total future seconds.
            res[f'next_i_s_{i}'] = max(self.weather_data['GHI'][next_s], 0) * 0.001
            next_hour = int(next_timestep / (3600 / self.time_gap))                         # Rounding total future hours.
            next_d = int(next_hour / 24)  # Electricity days
            next_h = next_hour % 24  # Electricity hours
            res[f'next_el_price_{i}'] = self.electricity_data.iloc[next_d, next_h] / 1000   # Electricity price is in EUR/MWh --> EUR/kWh
            res[f'next_occupancy_{i}'] = int(self.total_occupations_hourly[next_hour])
        return res
