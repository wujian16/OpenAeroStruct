from __future__ import division
import numpy

from openmdao.api import Component



class FunctionalBreguetRange(Component):
    """ Computes the fuel burn using the Breguet range equation """

    def __init__(self, surfaces, prob_dict):
        super(FunctionalBreguetRange, self).__init__()

        self.surfaces = surfaces
        self.prob_dict = prob_dict

        for surface in surfaces:
            name = surface['name']

            self.add_param(name+'CL', val=0.)
            self.add_param(name+'CD', val=0.)
            self.add_param(name+'weight', val=0.)

        self.add_output('fuelburn', val=0.)

        self.deriv_options['type'] = 'cs'
        self.deriv_options['form'] = 'central'

    def solve_nonlinear(self, params, unknowns, resids):
        CT = self.prob_dict['CT']
        a = self.prob_dict['a']
        R = self.prob_dict['R']
        M = self.prob_dict['M']
        fuelburn = 0.

        for surface in self.surfaces:
            name = surface['name']
            W0 = surface['W0']

            CL = params[name+'CL']
            CD = params[name+'CD']
            Ws = params[name+'weight']

            fuelburn += numpy.sum((W0 + Ws) * (numpy.exp(R * CT / a / M * CD / CL) - 1))

        unknowns['fuelburn'] = fuelburn

class FunctionalEquilibrium(Component):
    """ L = W constraint """

    def __init__(self, surfaces):
        super(FunctionalEquilibrium, self).__init__()

        self.surfaces = surfaces

        for surface in surfaces:
            name = surface['name']

            self.add_param(name+'L', val=0.)
            self.add_param(name+'weight', val=1.)

        self.add_param('fuelburn', val=1.)
        self.add_output('eq_con', val=0.)

        self.deriv_options['type'] = 'cs'
        self.deriv_options['form'] = 'central'
        #self.deriv_options['extra_check_partials_form'] = "central"

    def solve_nonlinear(self, params, unknowns, resids):
        weight = 0.
        L = 0.
        W0 = 0.
        for surface in self.surfaces:
            name = surface['name']
            weight += params[name+'weight']
            L += params[name+'L']
            W0 += surface['W0']

        unknowns['eq_con'] = (weight + params['fuelburn'] + W0 - L) / W0
