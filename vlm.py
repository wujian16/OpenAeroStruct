"""
Define the aerodynamic analysis component using a vortex lattice method.

We input a nodal mesh and properties of the airflow to calculate the
circulations of the horseshoe vortices. We then compute the forces, lift,
and drag acting on the lifting surfaces.

Note that some of the parameters and unknowns has the surface name
prepended on it. E.g., 'def_mesh' on a surface called 'wing' would be
'wing_def_mesh', etc. Please see an N2 diagram (the .html file produced by
this code) for more information about which parameters are renamed.

.. todo:: fix depiction of wing
Depiction of wing:

    y -->
 x  -----------------  Leading edge
 |  | + | + | + | + |
 v  |   |   |   |   |
    | o | o | o | o |
    -----------------  Trailing edge

The '+' symbols represent the bound points on the 1/4 chord line, which are
used to compute drag.
The 'o' symbols represent the collocation points on the 3/4 chord line, where
the flow tangency condition is inforced.

For this depiction, num_x = 2 and num_y = 5.
"""

from __future__ import division
import numpy

from openmdao.api import Component, Group
from scipy.linalg import lu_factor, lu_solve
try:
    import OAS_API
    fortran_flag = True
except:
    fortran_flag = False

def view_mat(mat):
    """ Helper function used to visually examine matrices. """
    import matplotlib.pyplot as plt
    if len(mat.shape) > 2:
        mat = numpy.sum(mat, axis=2)
    im = plt.imshow(mat.real, interpolation='none')
    plt.colorbar(im, orientation='horizontal')
    plt.show()


def norm(vec):
    """ Finds the 2-norm of a vector. """
    return numpy.sqrt(numpy.sum(vec**2))


def _calc_vorticity(A, B, P):
	""" Define the vorticity induced by the segment AB to P """
	rAP = P - A
	rBP = P - B
	rAP_len = norm(rAP)
	rBP_len = norm(rBP)
	cross = numpy.cross(rAP, rBP)
	may = numpy.sum(cross**2)

    # Cutoff to avoid singularity on b_pts, I think when collocation is on
    # a vortex
	r_cut = 1e-10
	cond = any([rAP_len < r_cut, rBP_len < r_cut, may < r_cut])
	if cond:
		return numpy.array([0., 0., 0.])

	return (rAP_len + rBP_len) * cross / \
           (rAP_len * rBP_len * (rAP_len * rBP_len + rAP.dot(rBP)))


def _assemble_AIC_mtx(b_pts_name, c_pts_name, mtx, params, surfaces,
                      transient, skip=False, wake=False):
    """
    Compute the aerodynamic influence coefficient matrix
    for either solving the linear system or solving for the drag.

    We use a nested for loop structure to loop through the lifting surfaces to
    obtain the corresponding mesh, then for each mesh we again loop through
    the lifting surfaces to obtain the collocation points used to compute
    the horseshoe vortex influence coefficients.

    This creates mtx with blocks corresponding to each lifting surface's
    effects on other lifting surfaces. The block diagonal portions
    correspond to each lifting surface's influence on itself. For a single
    lifting surface, this is the entire mtx.

    Parameters
    ----------
    mtx[num_y-1, num_y-1, 3] : array_like
        Aerodynamic influence coefficient (AIC) matrix, or the
        derivative of v w.r.t. circulations.
    params : dictionary
        OpenMDAO params dictionary for a given aero problem
    surfaces : dictionary
        Dictionary containing all surfaces in an aero problem.
    skip : boolean
        If false, the bound vortex contributions on the collocation point
        corresponding to the same panel are not included. Used for the drag
        computation.

    Returns
    -------
    mtx[tot_panels, tot_panels, 3] : array_like
        Aerodynamic influence coefficient (AIC) matrix, or the
        derivative of v w.r.t. circulations.
    """

    alpha = params['alpha']
    mtx[:, :, :] = 0.0
    cosa = numpy.cos(alpha * numpy.pi / 180.)
    sina = numpy.sin(alpha * numpy.pi / 180.)
    u = numpy.array([cosa, 0, sina])
    u = numpy.array([1., 0, 0.])

    i_ = 0
    i_bpts_ = 0
    i_panels_ = 0

    # Loop over the lifting surfaces to compute their influence on the flow
    # velocity at the collocation points
    for surface_ in surfaces:

        # Variable names with a trailing underscore correspond to the lifting
        # surface being examined, not the collocation point
        name_ = surface_['name']
        bpts = params[name_ + b_pts_name]

        # The shape of the bpts array will change for each timestep if we
        # have a transient case. If we have a steady-state case, nx_ and ny_
        # are equal to the values for num_x and num_y in the surface dict.
        nx_, ny_ = bpts.shape[:2]
        n_ = nx_ * ny_
        n_bpts_ = nx_ * ny_
        n_panels_ = (nx_ - 1) * (ny_ - 1)

        # Set counters to know where to index the sub-matrix within the full mtx
        i = 0
        i_bpts = 0
        i_panels = 0

        for surface in surfaces:
            # These variables correspond to the collocation points
            name = surface['name']

            # Obtain the collocation points used to compute the AIC mtx.
            # If setting up the AIC mtx, we use the collocation points (c_pts),
            # but if setting up the matrix to solve for drag, we use the
            # midpoints of the bound vortices.
            if c_pts_name == 'wake_mesh_local_frame' or c_pts_name == 'wake_mesh':
                pts = params[name + c_pts_name][1:, :, :]
            else:
                pts = params[name + c_pts_name]

            nx, ny = pts.shape[:2]
            nx += 1
            ny += 1
            n = nx * ny
            n_bpts = nx * ny
            n_panels = (nx - 1) * (ny - 1)
            symmetry = surface['symmetry']

            # Initialize sub-matrix to populate within full mtx
            small_mat = numpy.zeros((n_panels, n_panels_, 3), dtype='complex')

            # Fortran assembly for the AIC matrix
            if fortran_flag:
                small_mat[:, :, :] = OAS_API.oas_api.assembleaeromtx(alpha, pts, bpts,
                                                         skip, symmetry, transient)
            # Python matrix assembly
            else:
                # Chordwise loop through horseshoe elements in
                # reversed order, starting with the panel closest
                # to the leading edge. This is done to sum the
                # AIC contributions from the side vortex filaments
                # as we loop through the elements
                for el_i in xrange(nx_ - 1):
                    el_loc_i = el_i * (ny_ - 1)

                    # Spanwise loop through vortex rings
                    for el_j in xrange(ny_ - 1):
                        el_loc = el_loc_i + el_j

                        A = bpts[el_i + 0, el_j + 0, :]
                        B = bpts[el_i + 0, el_j + 1, :]
                        C = bpts[el_i + 1, el_j + 1, :]
                        D = bpts[el_i + 1, el_j + 0, :]

                        if symmetry:
                            sym = numpy.array([1., -1., 1.])
                            A_sym = A * sym
                            B_sym = B * sym
                            C_sym = C * sym
                            D_sym = D * sym

                        # Chordwise loop through control points
                        for cp_i in xrange(nx - 1):
                            cp_loc_i = cp_i * (ny - 1)

                            # Spanwise loop through control points
                            for cp_j in xrange(ny - 1):
                                cp_loc = cp_loc_i + cp_j

                                P = pts[cp_i, cp_j]

                                gamma = 0.
                                gamma += _calc_vorticity(B, C, P)
                                gamma += _calc_vorticity(D, A, P)
                                if skip:
                                    if el_i == nx_ - 2:
                                        gamma += _calc_vorticity(C, D, P)

                                else:
                                    gamma += _calc_vorticity(A, B, P)
                                    gamma += _calc_vorticity(C, D, P)

                                if symmetry:
                                    gamma += _calc_vorticity(C_sym, B_sym, P)
                                    gamma += _calc_vorticity(A_sym, D_sym, P)

                                    if skip:
                                        if el_i == nx_ - 2:
                                            gamma += _calc_vorticity(D_sym, C_sym, P)
                                    else:
                                        gamma += _calc_vorticity(B_sym, A_sym, P)
                                        gamma += _calc_vorticity(D_sym, C_sym, P)

                                # If skip, do not include the contributions
                                # from the panel's bound vortex filaments, as
                                # this causes a singularity when we're taking
                                # the influence of a panel on its own
                                # collocation point. This true for the drag
                                # computation and false for circulation
                                # computation, due to the different collocation
                                # points.
                                small_mat[cp_loc, el_loc, :] = gamma

            # Populate the full-size matrix with these surface-surface AICs
            mtx[i_panels:i_panels+n_panels, i_panels_:i_panels_+n_panels_, :] = small_mat

            i += n
            i_bpts += n_bpts
            i_panels += n_panels

        i_ += n_
        i_bpts_ += n_bpts_
        i_panels_ += n_panels_

    mtx /= 4 * numpy.pi


class VLMGeometry(Component):
    """ Compute various geometric properties for VLM analysis.

    Parameters
    ----------
    def_mesh[nx, ny, 3] : array_like
        Array defining the nodal coordinates of the lifting surface.

    Returns
    -------
    b_pts[nx, ny, 3] : array_like
        Bound points for the horseshoe vortices, found along the 1/4 chord.
    c_pts[nx-1, ny-1, 3] : array_like
        Collocation points on the 3/4 chord line where the flow tangency
        condition is satisfed. Used to set up the linear system.
    widths[nx-1, ny-1] : array_like
        The spanwise widths of each individual panel.
    normals[nx-1, ny-1, 3] : array_like
        The normal vector for each panel, computed as the cross of the two
        diagonals from the mesh points.
    S_ref : float
        The reference area of the lifting surface.

    """

    def __init__(self, surface, t, dt):
        super(VLMGeometry, self).__init__()

        self.surface = surface

        ny = surface['num_y']
        nx = surface['num_x']

        self.add_param('v', val=0.)
        self.add_param('alpha', val=0.)
        self.add_param('def_mesh', val=numpy.zeros((nx, ny, 3),
                       dtype="complex"))

        self.add_output('b_pts', val=numpy.zeros((nx, ny, 3),
                        dtype="complex"))
        self.add_output('c_pts', val=numpy.zeros((nx-1, ny-1, 3)))
        self.add_output('c_pts_inertial_frame', val=numpy.zeros((nx-1, ny-1, 3)))
        self.add_output('widths', val=numpy.zeros((nx-1, ny-1)))
        self.add_output('lengths', val=numpy.zeros((nx-1, ny-1)))
        self.add_output('normals', val=numpy.zeros((nx-1, ny-1, 3)))
        self.add_output('S_ref', val=0.)
        self.add_output('starting_vortex', val=numpy.zeros((1, ny, 3)))

        self.t = t
        self.dt = dt
        self.nx = nx
        self.ny = ny

        self.deriv_options['type'] = 'cs'
        self.deriv_options['form'] = 'central'

    	self.all_lengths = numpy.zeros((nx - 1, ny), dtype="complex")

    def get_lengths(self, A, B, axis):
    	return numpy.sqrt(numpy.sum((B - A)**2, axis=axis))

    def solve_nonlinear(self, params, unknowns, resids):

        # Create rotation vector based on the angle of attack, alpha.
        # This rotation vector is used to rotate b_pts and c_pts based on the
        # mesh and angle of attack.
        alpha_conv = params['alpha'] * numpy.pi / 180.
        cosa = numpy.cos(-alpha_conv)
        sina = numpy.sin(-alpha_conv)
        rot_x = numpy.array([cosa, 0, -sina])
        rot_z = numpy.array([sina, 0,  cosa])

        nx, ny = self.nx, self.ny
        mesh = params['def_mesh']

        b_pts = unknowns['b_pts']
        b_unrot = b_pts.copy()

        # Compute the bound points at 1/4 chord, except for the last one.
        # The last points cannot be computed because there are no mesh points
        # past the trailing edge of the wing.
        b_unrot[:-1, :, :] = mesh[:-1, :, :] * .75 + mesh[1:, :, :] * .25

        # Instead we compute the chordwise distance for the final bound points
        # using the flow conditions.
        x_dist = .3 * params['v'] * self.dt

        # The last bound points are this distance away from the trailing edge.
        b_unrot[-1, :, :] = mesh[-1, :, :]
        b_unrot[-1, :, 0] +=  + x_dist

        # Rotate the bound points based on the angle of attack.
        for i in xrange(nx):
            for j in xrange(ny):
                unknowns['b_pts'][i, j, 0] = b_unrot[i, j, :].dot(rot_x)
            	unknowns['b_pts'][i, j, 1] = b_unrot[i, j, 1]
            	unknowns['b_pts'][i, j, 2] = b_unrot[i, j, :].dot(rot_z)

        # Compute the collocation points at the midpoints of each
        # panel's 3/4 chord line.
        c_unrot = 0.5 * 0.25 * mesh[:-1, :-1, :] + \
                  0.5 * 0.75 * mesh[1:, :-1, :] + \
                  0.5 * 0.25 * mesh[:-1,  1:, :] + \
                  0.5 * 0.75 * mesh[1:,  1:, :]

        # Actually rotate the c_pts
        for i in xrange(nx - 1):
        	for j in xrange(ny - 1):
        		unknowns['c_pts'][i, j, 0] = c_unrot[i, j, :].dot(rot_x)
        		unknowns['c_pts'][i, j, 1] = c_unrot[i, j, 1]
        		unknowns['c_pts'][i, j, 2] = c_unrot[i, j, :].dot(rot_z)

        unknowns['starting_vortex'] = unknowns['b_pts'][-1, :, :]

        v_freestream = numpy.zeros((nx-1, ny-1, 3))
        v_freestream[:, :, 0] = params['v']

        # Shift the c_pts in the negative direction based on the velocity
        # at the wing. This velocity is just freestream velocity.
        unknowns['c_pts_inertial_frame'] = unknowns['c_pts'] - v_freestream * self.dt * self.t

        # Get lengths of panels, used later to compute forces
        self.all_lengths[:-1, :] = self.get_lengths(b_unrot[1:-1, :, :], b_unrot[:-2, :, :], 2)

        # The final length should be the chordwise length, same as other panels
        self.all_lengths[-1, :] = self.get_lengths(mesh[-1, :, :] - b_unrot[-2, :, :], \
                                     mesh[0, :, :] - b_unrot[0, :, :], 1)

        # Save the averaged lengths to get lengths per panel
        unknowns['lengths'] = (self.all_lengths[:, 1:] + self.all_lengths[:, :-1])/2

        # Save the widths of each panel
        unknowns['widths'] = self.get_lengths(b_unrot[:-1, 1:, :], b_unrot[:-1, :-1, :], 2)

        # Compute the normal of each panel by taking the cross-product of
        # its diagonals. Note that this could be a nonplanar surface
        normals = numpy.cross(b_pts[:-1,  1:, :] - b_pts[ 1:, :-1, :],
                              b_pts[:-1, :-1, :] - b_pts[ 1:,  1:, :], axis=2)

        # Compute the area of the panels based on the normals.
        # Note that the area is computed from the normals based on the mesh.
        S_ref = 0.5 * numpy.sum(numpy.sqrt(numpy.sum(normals**2, axis=2)))

        # Compute the normals of the bound points.
        normals = numpy.cross(
            b_pts[:-1,  1:, :] - b_pts[1:, :-1, :],
            b_pts[:-1, :-1, :] - b_pts[1:,  1:, :],
            axis=2)

        # Normalize the normals based on their magnitudes.
        norms = numpy.sqrt(numpy.sum(normals**2, axis=2))
        for j in xrange(3):
            normals[:, :, j] /= norms

        # Store each array
        unknowns['S_ref'] = S_ref
        unknowns['normals'] = normals

class VLMCirculations(Component):
    """
    Compute the circulations based on the AIC matrix and the panel velocities.
    Note that the flow tangency condition is enforced at the 3/4 chord point.

    Parameters
    ----------
    def_mesh[nx, ny, 3] : array_like
        Array defining the nodal coordinates of the lifting surface.
    b_pts[nx, ny, 3] : array_like
        Bound points for the horseshoe vortices, found along the 1/4 chord.
    c_pts[nx-1, ny-1, 3] : array_like
        Collocation points on the 3/4 chord line where the flow tangency
        condition is satisfed. Used to set up the linear system.
    normals[nx-1, ny-1, 3] : array_like
        The normal vector for each panel, computed as the cross of the two
        diagonals from the mesh points.
    v : float
        Freestream air velocity in m/s.
    alpha : float
        Angle of attack in degrees.

    Returns
    -------
    circulations : array_like
        Flattened vector of horseshoe vortex strengths calculated by solving
        the linear system of AIC_mtx * circulations = rhs, where rhs is
        based on the air velocity at each collocation point.

    """

    def __init__(self, surfaces, transient, t, dt):
        super(VLMCirculations, self).__init__()

        self.surfaces = surfaces
        self.transient = transient
        self.t = t
        self.dt = dt

        tot_panels = 0
        for surface in surfaces:
            tot_panels += (surface['num_x'] - 1) * (surface['num_y'] - 1)
        self.tot_panels = tot_panels

        for surface in surfaces:
            self.surface = surface
            ny = surface['num_y']
            nx = surface['num_x']
            name = surface['name']

            self.add_param(name+'b_pts', val=numpy.zeros((nx, ny, 3),
                           dtype="complex"))
            self.add_param(name+'c_pts', val=numpy.zeros((nx-1, ny-1, 3),
                           dtype="complex"))
            self.add_param(name+'c_pts_inertial_frame', val=numpy.zeros((nx-1, ny-1, 3),
                           dtype="complex"))
            self.add_param(name+'normals', val=numpy.zeros((nx-1, ny-1, 3)))

            size_wake = (ny - 1) * t
            size = (ny - 1) * (nx - 1)
            size_a1 = size_wake
            if t > 0:
                self.add_param(name+'prev_circ', val=numpy.zeros((tot_panels), dtype='complex'))
                self.add_param(name+'prev_wake_mesh', val=numpy.zeros((t+1, ny, 3), dtype='complex'))
                self.add_output(name+'wake_circ', val=numpy.zeros((t, ny-1), dtype='complex'))

                self.wake_circ = numpy.zeros((size_wake), dtype="complex")
            	self.v_wake = numpy.zeros((size, 3), dtype="complex")
            	self.a1_mtx = numpy.zeros((size, size_a1, 3), dtype="complex")

            if t > 1:
                self.add_param(name+'prev_wake_circ', val=numpy.zeros((t-1, ny-1), dtype='complex'))

        self.add_param('v', val=0.)
        self.add_param('alpha', val=0.)

        self.deriv_options['linearize'] = True  # only for circulations

        self.add_state('circulations', val=numpy.zeros((tot_panels),
                       dtype="complex"))
        if t > 0:
            self.add_output('v_wake_on_wing', val=numpy.zeros((tot_panels, 3), dtype='complex'))

        self.AIC_mtx = numpy.zeros((tot_panels, tot_panels, 3),
                                   dtype="complex")
        self.mtx = numpy.zeros((tot_panels, tot_panels), dtype="complex")
        self.rhs = numpy.zeros((tot_panels), dtype="complex")

    def _assemble_system(self, params, unknowns):

        print 'time step ', self.t
        # Actually assemble the AIC matrix for the wing
        _assemble_AIC_mtx('b_pts', 'c_pts', self.AIC_mtx, params, self.surfaces, self.transient)

        # Construct an flattend array with the normals of each surface in order
        # so we can do the normals with velocities to set up the right-hand-side
        # of the system.
        flattened_normals = numpy.zeros((self.tot_panels, 3), dtype='complex')
        i = 0
        for surface in self.surfaces:
            name = surface['name']
            nx, ny = surface['num_x'], surface['num_y']
            num_panels = (nx - 1) * (ny - 1)
            flattened_normals[i:i+num_panels, :] = params[name+'normals'].reshape(-1, 3, order='C')

            if self.t == 1:
                unknowns[name+'wake_circ'] = params[name+'prev_circ'][-(ny-1):]

            if self.t > 1:
                unknowns[name+'wake_circ'][0, :] = params[name+'prev_circ'][-(ny-1):]
                unknowns[name+'wake_circ'][1:, :] = params[name+'prev_wake_circ']

            i += num_panels

        if self.t > 0:
            # The inertial frame is body-fixed.
            # Therefore we must update the points based on the velocities.

            # a1_mtx is the matrix used to get the induced velocities on the wing from the wake.
            # This uses the wake_mesh which is really the b_pts for the wake
            _assemble_AIC_mtx('prev_wake_mesh', 'c_pts_inertial_frame', self.a1_mtx, params, self.surfaces, self.transient, wake=True)

            # Dot the matrix with the wake circulations to get the induced velocity
            # on the wing from the wake.
            # This is used to set the RHS of the system to solve for the new circulations.
            for ind in xrange(3):
                unknowns['v_wake_on_wing'][:, ind] = self.a1_mtx[:, :, ind].dot(unknowns[name+'wake_circ'].reshape(-1, order='C'))

        # Construct a matrix that is the AIC_mtx dotted by the normals at each
        # collocation point. This is used to compute the circulations
        self.mtx[:, :] = 0.
        for ind in xrange(3):
            self.mtx[:, :] += (self.AIC_mtx[:, :, ind].T * flattened_normals[:, ind]).T

        # Populate the right-hand side of the linear system with the
        # expected velocities at each collocation point
        v_c_pts = numpy.zeros((self.rhs.shape[0], 3), order='C')
        v_c_pts[:, ::3] = params['v']

        if self.t > 0:
            v_c_pts += unknowns['v_wake_on_wing']

        # Reshape the normals so that we can correctly produce the rhs
        norm = params[name+'normals'].reshape(-1, 3, order='C')

        # Populate the rhs vector
        self.rhs = numpy.sum(-norm * v_c_pts, axis=1)


    def solve_nonlinear(self, params, unknowns, resids):
        """ Solve the linear system to obtain circulations. """
        self._assemble_system(params, unknowns)
        unknowns['circulations'] = numpy.linalg.solve(self.mtx, self.rhs)

    def apply_nonlinear(self, params, unknowns, resids):
        """ Compute the residuals of the linear system. """
        self._assemble_system(params, unknowns)

        circ = unknowns['circulations']
        resids['circulations'] = self.mtx.dot(circ) - self.rhs

    def linearize(self, params, unknowns, resids):
        """ Jacobian for circulations."""
        self.lup = lu_factor(self.mtx.real)
        jac = self.alloc_jacobian()

        for surface in self.surfaces:
            name = surface['name']

            fd_jac = self.complex_step_jacobian(params, unknowns, resids,
                                             fd_params=[name+'normals', 'alpha',
                                                        name+'b_pts', name+'c_pts'],
                                             fd_states=[])
            jac.update(fd_jac)

        jac['circulations', 'circulations'] = self.mtx.real

        jac['circulations', 'v'][:, 0] = -self.rhs.real / params['v'].real

        return jac

    def solve_linear(self, dumat, drmat, vois, mode=None):

        if mode == 'fwd':
            sol_vec, rhs_vec = self.dumat, self.drmat
            t = 0
        else:
            sol_vec, rhs_vec = self.drmat, self.dumat
            t = 1

        for voi in vois:
            sol_vec[voi].vec[:] = lu_solve(self.lup, rhs_vec[voi].vec, trans=t)

class InducedVelocities(Component):
    """ Define induced velocities acting on each wing (v) or wake (w) panel
    'v_wing_on_wing_' + str(t) = velocity of c_pts (wing panels) induced by wing panels, at the time step t
    'v_wakewing_on_wake_' + str(t) = velocity of wake_mesh (wake points) induced by wing and wake panels, at the time step t """

    def __init__(self, surfaces, t, dt, transient):
        super(InducedVelocities, self).__init__()

        tot_panels = 0
        for surface in surfaces:
            self.surface = surface
            ny = surface['num_y']
            nx = surface['num_x']
            name = surface['name']

            self.add_param(name+'b_pts', val=numpy.zeros((nx, ny, 3),
                           dtype="complex"))
            self.add_param(name+'c_pts', val=numpy.zeros((nx-1, ny-1, 3),
                           dtype="complex"))
            if t > 0:
                self.add_param(name+'wake_mesh', val=numpy.zeros((t+1, ny, 3), dtype="complex"))
                self.add_param(name+'wake_mesh_local_frame', val=numpy.zeros((t+1, ny, 3), dtype="complex"))
            tot_panels += (surface['num_x'] - 1) * (surface['num_y'] - 1)

        self.tot_panels = tot_panels

        self.surfaces = surfaces
        self.transient = transient
        self.t = t
        self.dt = dt

        self.add_param('v', val=10.)
        self.add_param('alpha', val=10.)

        size = (nx-1) * (ny-1)
        self.add_param('circulations', val=numpy.zeros((tot_panels), dtype="complex"))
        self.add_output('v_wing_on_wing', val=numpy.zeros((tot_panels, 3), dtype="complex"))

        size_wake_mesh = ny * t
        size_a3 = (ny-1) * t

        # Only need wake induced velocity if it exists
        if t > 0:
            self.add_param(name+'wake_circ', val=numpy.zeros((t, ny-1), dtype="complex"))
            self.add_output('v_wakewing_on_wake', val=numpy.zeros((size_wake_mesh, 3), dtype="complex"))

            self.w_wake = numpy.zeros((size_wake_mesh, 3), dtype="complex")
            self.a3_mtx = numpy.zeros((size_wake_mesh, size_a3, 3), dtype="complex")
            self.a2_mtx = numpy.zeros((size_wake_mesh, tot_panels, 3), dtype="complex")

        self.v_wing = numpy.zeros((tot_panels, 3), dtype="complex")
        self.w_wing = numpy.zeros((size_wake_mesh, 3), dtype="complex")

        # Cap the size of the wake mesh based on the desired number of wake rows
        self.wake_mesh_local_frame = numpy.zeros((t+1, ny, 3), dtype="complex")

        self.a_mtx = numpy.zeros((tot_panels, tot_panels, 3), dtype="complex")
        self.b_mtx = numpy.zeros((tot_panels, tot_panels, 3), dtype="complex")

    def solve_nonlinear(self, params, unknowns, resids):

        t = self.t
        dt = self.dt

        _assemble_AIC_mtx('b_pts', 'c_pts', self.a_mtx, params, self.surfaces, self.transient, skip=True)

        # Obtain the induced velocity on the wing caused by the wing
        # by using the b_mtx previously obtained.
        # We currently do this in OAS too.
        for ind in xrange(3):
        	self.v_wing[:, ind] = self.a_mtx[:, :, ind].dot(params['circulations'])

        # Induced velocity on wing caused by wing
        unknowns['v_wing_on_wing'] = self.v_wing

        name = self.surfaces[0]['name']

        # Wake rollup (w)
        if t > 0:
            # params['wake_mesh_' + str(t)] doesn't change with each timestep
            # but self.wake_mesh_local_frame does change with each timestep

            # Assemble a2_mtx which is used to calculate the wing induced velocity
            # caused by the wake
            _assemble_AIC_mtx('b_pts', 'wake_mesh_local_frame', self.a2_mtx, params, self.surfaces, self.transient)

            # Assemble a3_mtx which is used to calculate the wake induced velocity
            # caused by the wake
            _assemble_AIC_mtx('wake_mesh', 'wake_mesh', self.a3_mtx, params, self.surfaces, self.transient)

            # Obtain the induced velocities on the wake caused by the wing and the wake
            for ind in xrange(3):
            	self.w_wing[:, ind] = self.a2_mtx[:, :, ind].dot(params['circulations'])
            	self.w_wake[:, ind] = self.a3_mtx[:, :, ind].dot(params[name+'wake_circ'].reshape(-1, order='C'))

            # Induced velocity on the wake caused by wing and wake
            unknowns['v_wakewing_on_wake'][:] = self.w_wing + self.w_wake

class WakeGeometry(Component):
    """ Update position of wake mesh in the body frame, adding a line for each time step
    'wake_mesh_' + str(nt) = position of wake mesh points (wake rings corners), at the time step t+1 """

    def __init__(self, surface, t, dt):
    	super(WakeGeometry, self).__init__()

        nx = surface['num_x']
        ny = surface['num_y']

    	self.add_param('v', val=10.)
        self.add_param('starting_vortex', val=numpy.zeros((1, ny, 3), dtype="complex"))
    	self.add_output('wake_mesh', val=numpy.zeros((t+2, ny, 3), dtype="complex"))
        self.add_output('wake_mesh_local_frame', val=numpy.zeros((t+2, ny, 3), dtype="complex"))

    	size_wake_mesh = ny * t

    	if t > 0:
    		self.add_param('v_wakewing_on_wake', val=numpy.zeros((size_wake_mesh, 3), dtype="complex"))
    		self.add_param('prev_wake_mesh', val=numpy.zeros((t+1, ny, 3), dtype="complex"))

    	self.deriv_options['form'] = 'central'

    	self.ny = ny
    	self.t = t
        self.dt = dt

    	self.v_wakewing_on_wake_resh = numpy.zeros((t, ny, 3), dtype="complex")
    	self.new_wake_row = numpy.zeros((1, ny, 3), dtype="complex")
    	self.old_wake = numpy.zeros((t+1, ny, 3), dtype="complex")

    def solve_nonlinear(self, params, unknowns, resids):

        ny = self.ny
        t = self.t
        dt = self.dt

        # Reshape of v_wakewing_on_wake so it can easily be applied to the wake mesh
        if t > 0:
        	for ind in xrange(3):
        		self.v_wakewing_on_wake_resh[:, :, ind] = params['v_wakewing_on_wake'][:, ind].reshape(t, ny, order='C')

        # Set old_wake based on if this is the first timestep or subsequent ones
        if t == 0:
            self.old_wake = params['starting_vortex']
        else:
            self.old_wake = params['prev_wake_mesh']

            # Here we apply the reshaped induced velocity on the wake caused by
            # the wing and the wake to the wake mesh.
            # This gives us the wake mesh from the third row to the end.
            unknowns['wake_mesh'][2:(t+2), :, :] = self.old_wake[1:(t+1), :, :] \
                                                   + self.v_wakewing_on_wake_resh * self.dt
            unknowns['wake_mesh'][(t+2):, :, :] = self.old_wake[(t+1):, :, :]

        # Update the second wake_mesh row
        unknowns['wake_mesh'][1, :, :] = self.old_wake[0, :, :]

        # Addition of a new wake row
        self.new_wake_row = params['starting_vortex']
        self.new_wake_row[0, :, 0] -= params['v'] * self.dt * (t+1)

        # Set the first wake_mesh row based on the starting vortex and the distance
        # traveled since then.
        unknowns['wake_mesh'][0, :, :] = self.new_wake_row

        # Translate the wake mesh into the local frame
        translation = numpy.array([params['v'] * dt * (t+1), 0., 0.])
        unknowns['wake_mesh_local_frame'] = unknowns['wake_mesh'] + translation

class VLMForces(Component):
    """ Compute aerodynamic forces acting on each section.
    Note that some of the parameters and unknowns has the surface name
    prepended on it. E.g., 'def_mesh' on a surface called 'wing' would be
    'wing_def_mesh', etc.
    Parameters
    ----------
    def_mesh[nx, ny, 3] : array_like
        Array defining the nodal coordinates of the lifting surface.
    b_pts[nx, ny, 3] : array_like
        Bound points for the horseshoe vortices, found along the 1/4 chord.
    circulations : array_like
        Flattened vector of horseshoe vortex strengths calculated by solving
        the linear system of AIC_mtx * circulations = rhs, where rhs is
        based on the air velocity at each collocation point.
    alpha : float
        Angle of attack in degrees.
    v : float
        Freestream air velocity in m/s.
    rho : float
        Air density in kg/m^3.
    Returns
    -------
    sec_forces[nx-1, ny-1, 3] : array_like
        Flattened array containing the sectional forces acting on each panel.
        Stored in Fortran order (only relevant when more than one chordwise
        panel).
    """

    def __init__(self, surfaces, transient, t, dt):
        super(VLMForces, self).__init__()

        tot_panels = 0
        for surface in surfaces:
            name = surface['name']
            tot_panels += (surface['num_x'] - 1) * (surface['num_y'] - 1)
            ny = surface['num_y']
            nx = surface['num_x']

            self.add_param(name+'b_pts', val=numpy.zeros((nx, ny, 3), dtype='complex'))
            self.add_param(name+'c_pts', val=numpy.zeros((nx-1, ny-1, 3), dtype='complex'))
            self.add_param(name+'widths', val=numpy.zeros((nx-1, ny-1), dtype='complex'))
            self.add_param(name+'normals', val=numpy.zeros((nx-1, ny-1, 3), dtype='complex'))
            self.add_param(name+'lengths', val=numpy.zeros((nx-1, ny-1)))
            self.add_output(name+'sec_forces', val=numpy.zeros((nx-1, ny-1, 3), dtype='complex'))

        self.tot_panels = tot_panels
        self.add_param('circulations', val=numpy.zeros((tot_panels)))
        self.add_param('alpha', val=3.)
        self.add_param('v', val=10.)
        self.add_param('rho', val=3.)
        self.add_output('sigma', val=numpy.zeros((nx-1, ny-1), dtype='complex'))

        self.add_param('v_wing_on_wing', val=numpy.zeros((tot_panels, 3), dtype="complex"))
        if t > 0:
            self.add_param('v_wake_on_wing', val=numpy.zeros((tot_panels, 3), dtype='complex'))
            self.add_param('prev_sigma', val=numpy.zeros((nx-1, ny-1), dtype='complex'))

        self.surfaces = surfaces
        self.transient = transient

        self.mtx = numpy.zeros((tot_panels, tot_panels, 3), dtype="complex")
        self.loc_circ = numpy.zeros((nx-1, ny-1), dtype="complex")
        self.t = t
        self.dt = dt

        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'

    def solve_nonlinear(self, params, unknowns, resids):
        nx, ny = self.surfaces[0]['num_x'], self.surfaces[0]['num_y']

        velo = numpy.zeros((nx-1, ny-1, 3))

        velo[:, :,  0] += params['v']
        vind = params['v_wing_on_wing'][:, 2].reshape(nx-1, ny-1, order='C')

        if self.t > 0:
            for ind in xrange(3):
                velo[:, :, ind] += params['v_wake_on_wing'][:, ind].reshape(nx-1, ny-1, order='C')

            vind += params['v_wake_on_wing'][:, 2].reshape(nx-1, ny-1, order='C')

        # Reshape the circulations into a matrix so we can more easily manipulate the values
        circ_mtx = params['circulations'].reshape(nx-1, ny-1, order='C')

        # For the first row of circulations, use the values.
        # For all other rows, use the difference between that value and the previous value.
        # This is necessary when using vortex rings to get the correct effects.
        self.loc_circ[0, :] = circ_mtx[0, :]
        self.loc_circ[1:, :] = circ_mtx[1:, :] - circ_mtx[:-1, :]

        # Velocity-potential time derivative (dCirc_dt) is obtained by integrating
        # from the leading edge
        unknowns['sigma'] = 0.5 * self.loc_circ
        unknowns['sigma'][1:, :] += circ_mtx[:-1, :]
        unknowns['sigma'] *= params['wing_lengths']

        # Obtain the change in circulation per timestep
        if self.t == 0:
        	dCirc_dt = unknowns['sigma'] / self.dt
        else:
            dCirc_dt = (unknowns['sigma'] - params['prev_sigma']) / self.dt

        # Lift for each panel
        forces_L = (velo[:, :, 0] * self.loc_circ + dCirc_dt) * params['wing_widths'] * params['wing_normals'][:, :, 2] * params['rho']

        # Induced drag for each panel
        forces_D = params['wing_widths'] * (-vind * self.loc_circ + dCirc_dt * params['wing_normals'][:, :, 0]) * params['rho']

        # section forces for structural part
        projected_forces = numpy.array(params['wing_normals'], dtype="complex")
        for ind in xrange(3):
        	projected_forces[:, :, ind] *= forces_L

        unknowns['wing_sec_forces'] = numpy.zeros((nx-1, ny-1, 3))

        print "L: {},  D: {}".format(numpy.sum(forces_L).real, numpy.sum(forces_D).real)


class VLMLiftDrag(Component):
    """
    Calculate total lift and drag in force units based on section forces.

    If the simulation is given a non-zero Reynolds number, it is used to
    compute the skin friction drag. If Reynolds number == 0, then there is
    no skin friction drag dependence. Currently, the Reynolds number
    must be set by the user in the run script and used as in IndepVarComp.

    Parameters
    ----------
    sec_forces[nx-1, ny-1, 3] : array_like
        Flattened array containing the sectional forces acting on each panel.
        Stored in Fortran order (only relevant when more than one chordwise
        panel).
    alpha : float
        Angle of attack in degrees.
    Re : float
        Reynolds number.
    M : float
        Mach number.
    v : float
        Freestream air velocity in m/s.
    rho : float
        Air density in kg/m^3.
    S_ref : float
        The reference area of the lifting surface.

    Returns
    -------
    L : float
        Total lift force for the lifting surface.
    D : float
        Total drag force for the lifting surface.

    """

    def __init__(self, surface):
        super(VLMLiftDrag, self).__init__()

        self.surface = surface
        ny = surface['num_y']
        nx = surface['num_x']

        self.add_param('sec_forces', val=numpy.zeros((nx - 1, ny - 1, 3)))
        self.add_param('alpha', val=3.)
        self.add_param('Re', val=5.e6)
        self.add_param('M', val=.84)
        self.add_param('v', val=10.)
        self.add_param('rho', val=3.)
        self.add_param('S_ref', val=0.)
        self.add_output('L', val=0.)
        self.add_output('D', val=0.)

        self.deriv_options['type'] = 'cs'
        self.deriv_options['form'] = 'central'

    def solve_nonlinear(self, params, unknowns, resids):
        forces = params['sec_forces'].reshape(-1, 3)
        alpha = params['alpha'] * numpy.pi / 180.
        cosa = numpy.cos(alpha)
        sina = numpy.sin(alpha)

        Re = params['Re']
        M = params['M']
        v = params['v']
        rho = params['rho']
        S_ref = params['S_ref']

        # Compute the skin friction coefficient
        # Use eq. 12.27 of Raymer for turbulent Cf
        # Avoid divide by zero warning if Re == 0
        if Re == 0:
            Cf = 0.
        else:
            Cf = 0.455 / (numpy.log10(Re)**2.58 * (1 + .144 * M**2)**.65)

        # Compute the induced lift force on each lifting surface
        unknowns['L'] = \
            numpy.sum(-forces[:, 0] * sina + forces[:, 2] * cosa)

        # Compute the induced drag force on each lifting surface
        unknowns['D'] = \
            numpy.sum( forces[:, 0] * cosa + forces[:, 2] * sina)

        # Compute the drag contribution from skin friction
        D_f = Cf * rho * v**2 / 2. * S_ref
        unknowns['D'] += D_f

        if self.surface['symmetry']:
            unknowns['D'] *= 2
            unknowns['L'] *= 2

class VLMCoeffs(Component):
    """ Compute lift and drag coefficients.

    Parameters
    ----------
    S_ref : float
        The reference areas of the lifting surface.
    L : float
        Total lift for the lifting surface.
    D : float
        Total drag for the lifting surface.
    v : float
        Freestream air velocity in m/s.
    rho : float
        Air density in kg/m^3.

    Returns
    -------
    CL1 : float
        Induced coefficient of lift (CL) for the lifting surface.
    CDi : float
        Induced coefficient of drag (CD) for the lifting surface.

    """


    def __init__(self, surface):
        super(VLMCoeffs, self).__init__()

        self.surface = surface

        self.add_param('S_ref', val=0.)
        self.add_param('L', val=0.)
        self.add_param('D', val=0.)
        self.add_param('v', val=0.)
        self.add_param('rho', val=0.)
        self.add_output('CL', val=0.)
        self.add_output('CD', val=0.)

        self.deriv_options['type'] = 'cs'
        self.deriv_options['form'] = 'central'

    def solve_nonlinear(self, params, unknowns, resids):
        S_ref = params['S_ref']
        rho = params['rho']
        v = params['v']
        L = params['L']
        D = params['D']

        if self.surface['symmetry']:
            S_ref *= 2

        unknowns['CL'] = L / (0.5 * rho * v**2 * S_ref) + self.surface['CL0']
        unknowns['CD'] = D / (0.5 * rho * v**2 * S_ref) + self.surface['CD0']


class VLMStates(Group):
    """ Group that contains the aerodynamic states. """

    def __init__(self, surfaces, t, dt, transient):
        super(VLMStates, self).__init__()

        self.add('circulations',
                 VLMCirculations(surfaces, transient, t, dt),
                 promotes=['*'])
        self.add('ind_vel',
              InducedVelocities(surfaces, t, dt, transient),
              promotes=['*'])
        self.add('forces',
                 VLMForces(surfaces, transient, t, dt),
                 promotes=['*'])

        if t == 0:
            print 'Transient =', transient


class VLMFunctionals(Group):
    """ Group that contains the aerodynamic functionals used to evaluate
    performance. """

    def __init__(self, surface, t, dt):
        super(VLMFunctionals, self).__init__()

        self.add('liftdrag',
                 VLMLiftDrag(surface),
                 promotes=['*'])
        self.add('coeffs',
                 VLMCoeffs(surface),
                 promotes=['*'])
