""" Example runscript to perform structural-only optimization.

Call as `python run_spatialbeam.py 0` to run a single analysis, or
call as `python run_spatialbeam.py 1` to perform optimization.

To run with multiple structural components instead of a single one,
call as `python run_spatialbeam.py 0m` to run a single analysis, or
call as `python run_spatialbeam.py 1m` to perform optimization.

"""

from __future__ import division, print_function
import sys
from time import time
import numpy

from run_classes import OASProblem

if __name__ == "__main__":

    # Make sure that the user-supplied input is one of the valid options
    input_options = ['0', '0m', '1', '1m']
    print_str = ''.join(str(e) + ', ' for e in input_options)

    # Parse the user-supplied command-line input and store it as input_arg
    try:
        input_arg = sys.argv[1]
        if input_arg not in input_options:
            raise(IndexError)
    except IndexError:
        print('\n +---------------------------------------------------------------+')
        print(' | ERROR: Please supply a correct input argument to this script. |')
        print(' | Possible options are ' + print_str[:-2] + '                             |')
        print(' | See the docstring at the top of this file for more info.      |')
        print(' +---------------------------------------------------------------+\n')
        raise

    # Set problem type
    prob_dict = {'type' : 'struct'}

    if sys.argv[1].startswith('0'):  # run analysis once
        prob_dict.update({'optimize' : False})
    else:  # perform optimization
        prob_dict.update({'optimize' : True})

    # Instantiate problem and add default surface
    OAS_prob = OASProblem(prob_dict)
    OAS_prob.add_surface({'name' : 'wing',
                          'num_y' : 5})

    # Single lifting surface
    if not sys.argv[1].endswith('m'):

        # Setup problem and add design variables, constraint, and objective
        OAS_prob.setup()
        OAS_prob.add_desvar('wing.thickness_cp', lower=0.001, upper=0.25, scaler=1e2)
        OAS_prob.add_constraint('wing.failure', upper=0.)
        OAS_prob.add_objective('wing.weight', scaler=1e-3)

    # Multiple lifting surfaces
    else:

        # Add additional lifting surface
        OAS_prob.add_surface({'name' : 'tail',
                              'span' : 3.,
                              'offset' : numpy.array([10., 0., 0.])})

        # Setup problem and add design variables, constraints, and objective
        OAS_prob.setup()

        OAS_prob.add_desvar('wing.thickness_cp', lower=0.001, upper=0.25, scaler=1e2)
        OAS_prob.add_constraint('wing.failure', upper=0.)
        OAS_prob.add_objective('wing.weight', scaler=1e-3)

        # Note that these tail variables have no effect on the wing and thus
        # have no need to be changed except to satisfy the failure constraint
        OAS_prob.add_desvar('tail.thickness_cp', lower=0.001, upper=0.25, scaler=1e2)
        OAS_prob.add_constraint('tail.failure', upper=0.)

    # Actually run the problem
    OAS_prob.run()

    print("\nWing weight:", OAS_prob.prob['wing.weight'])
    # jac = OAS_prob.prob.calc_gradient(['wing.disp_aug'], ['wing.disp'], return_format="dict")
    # print(jac['wing.disp']['wing.disp_aug'])
