""" Example runscript to perform aerodynamics-only optimization.

Call as `python run_vlm.py 0` to run a single analysis, or
call as `python run_vlm.py 1` to perform optimization.

To run with multiple lifting surfaces instead of a single one,
Call as `python run_vlm.py 0m` to run a single analysis, or
call as `python run_vlm.py 1m` to perform optimization.

"""

from __future__ import division
import sys
from time import time
import numpy

from run_classes import OASProblem

if __name__ == "__main__":

    # Set problem type
    prob_dict = {'type' : 'aero'}

    if sys.argv[1].startswith('0'):  # run analysis once
        prob_dict.update({'optimize' : False})
    else:  # perform optimization
        prob_dict.update({'optimize' : True})

    # Instantiate problem and add default surface
    OAS_prob = OASProblem(prob_dict)
    OAS_prob.add_surface({'name' : 'wing',
                          'symmetry' : False,
                          'num_y' : 5,
                          'num_x' : 3,
                          'span_cos_spacing' : 0.})

    # Single lifting surface
    if not sys.argv[1].endswith('m'):

        # Setup problem and add design variables, constraint, and objective
        OAS_prob.setup()
        OAS_prob.add_desvar('wing.twist_cp', lower=-10., upper=15.)
        OAS_prob.add_desvar('wing.sweep', lower=10., upper=30.)
        OAS_prob.add_desvar('wing.dihedral', lower=-10., upper=20.)
        OAS_prob.add_desvar('wing.taper', lower=.5, upper=2.)
        OAS_prob.add_constraint('wing_perf.CL', equals=0.5)
        OAS_prob.add_objective('wing_perf.CD', scaler=1e4)

    # Multiple lifting surfaces
    else:

        # Add additional lifting surface
        OAS_prob.add_surface({'name' : 'tail',
                              'span' : 3.,
                              'offset' : numpy.array([5., 0., 1.])})

        # Setup problem and add design variables, constraints, and objective
        OAS_prob.setup()

        # Set up wing variables
        OAS_prob.add_desvar('wing.twist_cp', lower=-10., upper=15.)
        OAS_prob.add_desvar('wing.sweep', lower=10., upper=30.)
        OAS_prob.add_desvar('wing.dihedral', lower=-10., upper=20.)
        OAS_prob.add_desvar('wing.taper', lower=.5, upper=2.)
        OAS_prob.add_constraint('wing_perf.CL', equals=0.5)
        OAS_prob.add_objective('wing_perf.CD', scaler=1e4)

        # Set up tail variables
        OAS_prob.add_desvar('tail.twist_cp', lower=-10., upper=15.)
        OAS_prob.add_desvar('tail.sweep', lower=10., upper=30.)
        OAS_prob.add_desvar('tail.dihedral', lower=-10., upper=20.)
        OAS_prob.add_desvar('tail.taper', lower=.5, upper=2.)
        OAS_prob.add_constraint('tail_perf.CL', equals=0.5)

    # Actually run the problem
    OAS_prob.run()

    print "\nWing CL:", OAS_prob.prob['wing_perf.CL']
    print "Wing CD:", OAS_prob.prob['wing_perf.CD']
    print 'L/D', OAS_prob.prob['wing_perf.CL'] / OAS_prob.prob['wing_perf.CD']
