# Integrated bilding HVAC and building simulation

This is novel implementaiton of a building simulation based on a grey-box RC model from Peder Batcher et al [^1]. The idea behind this implementation is to be used as a base for Model Predictive Control and Machine Learning implementation.

The model has been extended by adding new features such as Hydronic heating system [^2] and a vestilation system to simlate a more realistic usage.

In addition, the model has been enhacen by adding an occupant estimation algortihm develped by Page et al [^3] taking the code base from [City Energy Analyst](https://zenodo.org/record/1487867) and real weather data provided by the Technical Univerty of Denmark [^4].

The simulation environtment simulataes the usage of a heat pump which uses a Hydronic heating system. To ensure a balance between heat pump usage and occupants' comfort a cost calculation of th eusage of the heat pump is implemented. The electriciy prices are directly extracted by [NordPool](https://www.nordpoolgroup.com/) compnay.


## How to use it

First of all you need a Python 3.8.x > system installed in your machine. It is recommended to use a Virtualenvirontment to isolate the instalation of libraries.

* Use pip to install the required libraries from *requirements.txt* file.
* Create a **main** file to instantiate the *simulator* class.
  * Example
```

from simulator.simulator import Simulator

model = Simulator(
    simulated_years=(2018, 2019,)
  )
```
* Do a step and gather data
  * Example
```

# Do one step
# p_act_hp --> 0 Heat pump is off , --> 1 Heat pump is on
# p_t_sup --> supplied temperature from heat pump

simulation_data = model.simulate_step(p_act_hp, p_t_sup)
# After doing a step data is collected
t_s = simulation_data.get('t_s', None)
t_h = simulation_data.get('t_h', None)
t_i = simulation_data.get('t_i', None)
t_e = simulation_data.get('t_e', None)
ambient_temperature = simulation_data.get('t_a', None)
solar_irradiance = simulation_data.get('i_s', None)                             
current_el_price = simulation_data.get('current_el_price', None)
occupancy = simulation_data.get('occupancy', None)
```
* Once finished it is possible to "reset" the simulation process
  * Example
```
# Once finished, it is possible to reset the engine
# Random reset --> True indicates that the intiial timestamp is random. False indicas that the initial timestsamp is 1st Jaunary of the given initial year (in this example 2018)

self.simulation_engine.reset_simulator(p_random_reset=self.random_reset)

```

Available configuration of Simulator class instance are: 

1. _p_occupant_profile_: Allows to pass an array based custom occupant file with the format [0, 1, 3, 6, 8, ...] In case that this value is None, the system will generate one automatically.
2. _p_time_gap:_: Delta time defined for each simulation step. Default value is 10 seconds.
3. _p_end_date_: Allows to define a final date. If None is present, then the final date will be the end of the year.
4. _p_simulated_years_: Total years to be simulated. if None is present a default year will be picked.




## Acknowledgements

I would like to thank you the following persons/services to allow this to make happen:

* Meteosat weather data provider
* NordPool electricity provider
* Peder Batcher.
* Jessen Page.
* People behind [City Energy Analyst](https://zenodo.org/record/1487867) program.


_______

[^1] Bacher, P., & Madsen, H. (2011). Identifying suitable models for the heat dynamics of buildings. Energy and buildings, 43(7), 1511-1522.

[^2]Radiator model provided by: Hydronic heating systems - the effect of design on system sensitivity. PhD, Department of Building Services Engineering, Chalmers University of Technology (2002)

[^3] Page, J., Robinson, D., Morel, N., & Scartezzini, J. L. (2008). A generalised stochastic model for the simulation of occupant presence. Energy and buildings, 40(2), 83-98

[^4] DTU Climate Station, Department of Civil Engineering, Technical University of Denmark, http://climatestationdata.byg.dtu.dk
