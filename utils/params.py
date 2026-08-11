"""
Simulation parameters for the T-Cell contact model.

Constants used across multiple scripts are centralized here to keep the 
code more manageable and readable. Used in: contact_simulation.py, 
distribution_test.py, params_computation, and tcell_model_base.py.
"""

CLASS = ['ag', 'nag', 'bg']
T_CONTACT = 1000

# From Huang et al. 2010, Table 1, 37°C (2D kinetics)
K_OFF = {'ag': 10.8, 'nag': 1.3, 'bg': 50.0}  # s^(-1)
K_ON  = {'ag': 1.2e-2, 'nag': 2.7e-5, 'bg': 1e-6}  # μm^4s^(-1)

L_MAX = {'ag': 5, 'nag': 2, 'bg': 500} # treated as lambda for poisson sampling
R_MAX = [50]