""" Example runscript to perform structural-only optimization.

Call as `python run_spatialbeam.py 0` to run a single analysis, or
call as `python run_spatialbeam.py 1` to perform optimization.

To run with multiple structural components instead of a single one,
call as `python run_spatialbeam.py 0m` to run a single analysis, or
call as `python run_spatialbeam.py 1m` to perform optimization.

"""

from __future__ import division
import sys
from time import time
import numpy

from run_classes import OASProblem

if __name__ == "__main__":

    # Set problem type
    prob_dict = {'type' : 'struct'}

    # Instantiate problem and add default surface
    OAS_prob = OASProblem(prob_dict)
    OAS_prob.add_surface({'name' : 'wing',
                          'num_y' : 201,
                          'num_x' : 5,
                          'symmetry' : True,
                          'span' : 32.,
                          'chord' : 1.,
                          'fem_origin' : 0.5,
                          'E' : 70.e9,
                          'G' : 30.e9,
                          'mrho' : 3.e3,
                          'W0' : 0.,
                          'span_cos_spacing' : 0.})

    # Setup problem and add design variables, constraint, and objective
    OAS_prob.setup()
    OAS_prob.add_desvar('wing.thickness_cp', lower=0.01, upper=0.25, scaler=1e2)
    OAS_prob.add_constraint('wing.failure', upper=0.)
    OAS_prob.add_objective('wing.weight', scaler=1e-3)


    # Actually run the problem
    OAS_prob.run()

    print "\nWing weight:", OAS_prob.prob['wing.weight']
    print "nat freqs", OAS_prob.prob['wing.freqs'][:20]
