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
    """ Calculates the influence coefficient for a vortex filament.

    Parameters
    ----------
    A[3] : array_like
        Coordinates for the start point of the filament.
    B[3] : array_like
        Coordinates for the end point of the filament.
    P[3] : array_like
        Coordinates for the collocation point where the influence coefficient
        is computed.

    Returns
    -------
    out[3] : array_like
        Influence coefficient contribution for the described filament.

    """

    r1 = P - A
    r2 = P - B

    r1_mag = norm(r1)
    r2_mag = norm(r2)

    return (r1_mag + r2_mag) * numpy.cross(r1, r2) / \
           (r1_mag * r2_mag * (r1_mag * r2_mag + r1.dot(r2)))


def _assemble_AIC_mtx(mtx, params, surfaces, transient, skip=False, wake=False):
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

        # Obtain the lifting surface mesh in the form expected by the solver,
        # with shape [nx_, ny_, 3]
        mesh = params[name_+'def_mesh']
        if wake:
            bpts = params[name_+'wake_b_pts']
        else:
            bpts = params[name_+'b_pts']

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
            nx = surface['num_x']
            ny = surface['num_y']
            n = nx * ny
            n_bpts = nx * ny
            n_panels = (nx - 1) * (ny - 1)
            symmetry = surface['symmetry']

            # Obtain the collocation points used to compute the AIC mtx.
            # If setting up the AIC mtx, we use the collocation points (c_pts),
            # but if setting up the matrix to solve for drag, we use the
            # midpoints of the bound vortices.
            if skip:
                # Find the midpoints of the bound points, used in drag computations
                pts = (params[name+'b_pts'][:-1, 1:, :] + \
                    params[name+'b_pts'][:-1, :-1, :]) / 2
            else:
                pts = params[name+'c_pts']

            # Initialize sub-matrix to populate within full mtx
            small_mat = numpy.zeros((n_panels, n_panels_, 3), dtype='complex')

            # Fortran assembly for the AIC matrix
            if fortran_flag:
                small_mat[:, :, :] = OAS_API.oas_api.assembleaeromtx(alpha, pts, bpts,
                                                         skip, symmetry, transient)
            # Python matrix assembly
            else:
                # Spanwise loop through vortex rings
                for el_j in xrange(ny_ - 1):
                    el_loc_j = el_j * (nx_ - 1)

                    # Chordwise loop through horseshoe elements in
                    # reversed order, starting with the panel closest
                    # to the leading edge. This is done to sum the
                    # AIC contributions from the side vortex filaments
                    # as we loop through the elements
                    for el_i in xrange(nx_ - 1):
                        el_loc = el_i + el_loc_j

                        A = bpts[el_i, el_j + 0, :]
                        B = bpts[el_i, el_j + 1, :]

                        # Check if this is the last panel; if so, use
                        # the trailing edge mesh points for C & D, else
                        # use the directly aft panel's bound points
                        # for C & D
                        C = bpts[el_i + 1, el_j + 1, :]
                        D = bpts[el_i + 1, el_j + 0, :]
                        if el_i == nx_ - 2:
                            C_far = 1.e6 * u + C
                            D_far = 1.e6 * u + D

                        if symmetry:
                            sym = numpy.array([1., -1., 1.])
                            A_sym = A * sym
                            B_sym = B * sym
                            C_sym = C * sym
                            D_sym = D * sym
                            if el_i == nx_ - 2:
                                C_far_sym = C_far * sym
                                D_far_sym = D_far * sym

                        # Spanwise loop through control points
                        for cp_j in xrange(ny - 1):
                            cp_loc_j = cp_j * (nx - 1)

                            # Chordwise loop through control points
                            for cp_i in xrange(nx - 1):
                                cp_loc = cp_i + cp_loc_j

                                P = pts[cp_i, cp_j]

                                gamma = 0.
                                gamma += _calc_vorticity(B, C, P)
                                gamma += _calc_vorticity(D, A, P)
                                if symmetry:
                                    gamma += _calc_vorticity(C_sym, B_sym, P)
                                    gamma += _calc_vorticity(A_sym, D_sym, P)

                                if not skip:
                                    gamma += _calc_vorticity(A, B, P)
                                    if symmetry:
                                        gamma += _calc_vorticity(B_sym, A_sym, P)

                                    if not wake:
                                        if el_i < nx_ - 2:
                                            gamma += _calc_vorticity(C, D, P)
                                            if symmetry:
                                                gamma += _calc_vorticity(D_sym, C_sym, P)
                                    else:
                                        gamma += _calc_vorticity(C, D, P)
                                        if symmetry:
                                            gamma += _calc_vorticity(D_sym, C_sym, P)

                                else:
                                    if el_loc == cp_loc:
                                        pass
                                    else:
                                        gamma += _calc_vorticity(A, B, P)
                                        if symmetry:
                                            gamma += _calc_vorticity(B_sym, A_sym, P)

                                    if el_i + 1 == cp_i and el_j == cp_j:
                                        pass
                                    else:
                                        if el_i < nx_-2:
                                            gamma += _calc_vorticity(C, D, P)
                                            if symmetry:
                                                gamma += _calc_vorticity(D_sym, C_sym, P)

                                if el_i == nx_ - 2 and not transient:
                                    gamma += _calc_vorticity(C, C_far, P)
                                    gamma += _calc_vorticity(D_far, D, P)
                                    if symmetry:
                                        gamma += _calc_vorticity(C_far_sym, C_sym, P)
                                        gamma += _calc_vorticity(D_sym, D_far_sym, P)

                                # gamma += _calc_vorticity(B, C, P)
                                # gamma += _calc_vorticity(D, A, P)
                                # gamma += _calc_vorticity(C, D, P)
                                # gamma += _calc_vorticity(A, B, P)
                                # if symmetry:
                                #     gamma += _calc_vorticity(C_sym, B_sym, P)
                                #     gamma += _calc_vorticity(A_sym, D_sym, P)
                                #     gamma += _calc_vorticity(D_sym, C_sym, P)
                                #     gamma += _calc_vorticity(B_sym, A_sym, P)


                                # If skip, do not include the contributions
                                # from the panel's bound vortex filament, as
                                # this causes a singularity when we're taking
                                # the influence of a panel on its own
                                # collocation point. This true for the drag
                                # computation and false for circulation
                                # computation, due to the different collocation
                                # points.
                                small_mat[cp_loc, el_loc, :] = gamma

            # exit()
            # Populate the full-size matrix with these surface-surface AICs
            mtx[i_panels:i_panels+n_panels,
                i_panels_:i_panels_+n_panels_, :] = small_mat

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

    def __init__(self, surface, t, dt, transient):
        super(VLMGeometry, self).__init__()

        self.surface = surface
        self.transient = transient

        ny = surface['num_y']
        nx = surface['num_x']

        self.add_param('v', val=0.)
        self.add_param('alpha', val=0.)
        self.add_param('def_mesh', val=numpy.zeros((nx, ny, 3),
                       dtype="complex"))

        if t > 0:
            self.add_param('prev_wake_b_pts', val=numpy.zeros((t, ny, 3)))

        self.add_output('b_pts', val=numpy.zeros((nx, ny, 3),
                        dtype="complex"))
        self.add_output('c_pts', val=numpy.zeros((nx-1, ny-1, 3)))
        self.add_output('widths', val=numpy.zeros((nx-1, ny-1)))
        self.add_output('normals', val=numpy.zeros((nx-1, ny-1, 3)))
        self.add_output('S_ref', val=0.)

        self.add_output('wake_b_pts', val=numpy.zeros((t+1, ny, 3),
                        dtype="complex"))

        self.t = t
        self.dt = dt

        self.deriv_options['type'] = 'cs'
        self.deriv_options['form'] = 'central'

    def solve_nonlinear(self, params, unknowns, resids):
        mesh = params['def_mesh']

        b_pts = unknowns['b_pts']

        # Compute the bound points at 1/4 chord
        b_pts[:-1, :, :] = mesh[:-1, :, :] * .75 + mesh[1:, :, :] * .25

        # Populate the last row of b_pts with the extrapolated mesh
        # May need to incorporate the timestep and velocity here like Gio
        direction = mesh[-1, :, :] - mesh[-2, :, :]
        norm = numpy.linalg.norm(direction, axis=1)
        for ind in xrange(3):
            direction[:, ind] /= norm
        b_pts[-1, :, :] = .3 * params['v'] * self.dt * direction + mesh[-1, :, :]

        # Compute the collocation points at the midpoints of each
        # panel's 3/4 chord line
        c_pts = 0.5 * 0.25 * mesh[:-1, :-1, :] + \
                0.5 * 0.75 * mesh[1:, :-1, :] + \
                0.5 * 0.25 * mesh[:-1,  1:, :] + \
                0.5 * 0.75 * mesh[1:,  1:, :]

        # Compute the widths of each panel
        widths = numpy.sqrt(numpy.sum((b_pts[:-1, 1:, :] - b_pts[:-1, :-1, :])**2, axis=2))

        # Compute the normal of each panel by taking the cross-product of
        # its diagonals. Note that this could be a nonplanar surface
        normals = numpy.cross(
            mesh[:-1,  1:, :] - mesh[1:, :-1, :],
            mesh[:-1, :-1, :] - mesh[1:,  1:, :],
            axis=2)

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
        unknowns['c_pts'] = c_pts
        unknowns['widths'] = widths
        unknowns['normals'] = normals

        ### END OF PREVIOUS GEOMETRY CALCULATIONS

        ### START OF WAKE GEOMETRY CALCULATIONS

        if self.t == 0:
            old_wake = b_pts[-1, :, :]
        else:
            # change this for real
            old_wake = params['prev_wake_b_pts']

        alpha = params['alpha']*numpy.pi/180.
        cosa = numpy.cos(alpha)
        sina = numpy.sin(alpha)
        freestream_direction = numpy.array([cosa, 0., sina])

        distance = params['v'] * self.dt * freestream_direction

        # Wake bound points only matter after t > 0.
        # In the first timestep there is no wake.
        unknowns['wake_b_pts'][1:, :, :] = old_wake + distance
        unknowns['wake_b_pts'][0, :, :] = b_pts[-1, :, :]


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

        for surface in surfaces:
            self.surface = surface
            ny = surface['num_y']
            nx = surface['num_x']
            name = surface['name']

            self.add_param(name+'def_mesh', val=numpy.zeros((nx, ny, 3),
                           dtype="complex"))
            self.add_param(name+'b_pts', val=numpy.zeros((nx, ny, 3),
                           dtype="complex"))
            self.add_param(name+'c_pts', val=numpy.zeros((nx-1, ny-1, 3),
                           dtype="complex"))
            self.add_param(name+'normals', val=numpy.zeros((nx-1, ny-1, 3)))
            self.add_param(name+'wake_b_pts', val=numpy.zeros((t+1, ny, 3),
                            dtype="complex"))

            if t > 0:
                self.add_param(name+'prev_circ', val=numpy.zeros((t, ny-1), dtype='complex'))


        self.add_param('v', val=0.)
        self.add_param('alpha', val=0.)

        self.deriv_options['linearize'] = True  # only for circulations

        tot_panels = 0
        for surface in surfaces:
            tot_panels += (surface['num_x'] - 1) * (surface['num_y'] - 1)
        self.tot_panels = tot_panels

        self.add_state('circulations', val=numpy.zeros((tot_panels),
                       dtype="complex"))

        self.AIC_mtx = numpy.zeros((tot_panels, tot_panels, 3),
                                   dtype="complex")
        self.mtx = numpy.zeros((tot_panels, tot_panels), dtype="complex")
        self.rhs = numpy.zeros((tot_panels), dtype="complex")

    def _assemble_system(self, params):

        # Actually assemble the AIC matrix for the wing
        _assemble_AIC_mtx(self.AIC_mtx, params, self.surfaces, self.transient)

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

            if self.t > 0:
                # mult! change size of matrix
                wake_mtx = numpy.zeros((num_panels, self.t*(ny-1), 3), dtype='complex')

                # Actually assemble the AIC matrix for the wake on the wing
                _assemble_AIC_mtx(wake_mtx, params, self.surfaces, self.transient, wake=True)

                circ = params[name+'prev_circ'].reshape(-1, order='C')
                v_ind_wake = numpy.zeros((num_panels, 3))
                for ind in xrange(3):
                    v_ind_wake[:, ind] = wake_mtx[:, :, ind].dot(circ)

            i += num_panels

        # Construct a matrix that is the AIC_mtx dotted by the normals at each
        # collocation point. This is used to compute the circulations
        self.mtx[:, :] = 0.
        for ind in xrange(3):
            self.mtx[:, :] += (self.AIC_mtx[:, :, ind].T * flattened_normals[:, ind]).T

        # Obtain the freestream velocity direction and magnitude by taking
        # alpha into account
        alpha = params['alpha'] * numpy.pi / 180.
        cosa = numpy.cos(alpha)
        sina = numpy.sin(alpha)
        v = params['v'] * numpy.array([cosa, 0., sina], dtype="complex")

        # Populate the right-hand side of the linear system with the
        # expected velocities at each collocation point
        v_vec = numpy.ones((self.rhs.shape[0], 3)) * v
        self.rhs[:] = -flattened_normals.reshape(-1, 3, order='C').dot(v)
        if self.t > 0:
            v_vec += v_ind_wake
            self.rhs[:] = -numpy.sum(flattened_normals.reshape(-1, 3, order='C') * (v_vec), axis=1)
        else:
            self.rhs[:] = -flattened_normals.reshape(-1, 3, order='C').dot(v)


    def solve_nonlinear(self, params, unknowns, resids):
        """ Solve the linear system to obtain circulations. """
        self._assemble_system(params)
        unknowns['circulations'] = numpy.linalg.solve(self.mtx, self.rhs)
        print 'circ'
        print unknowns['circulations']

    def apply_nonlinear(self, params, unknowns, resids):
        """ Compute the residuals of the linear system. """
        self._assemble_system(params)

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
                                                        name+'def_mesh', name+'b_pts',
                                                        name+'c_pts'],
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

            self.add_param(name+'def_mesh', val=numpy.zeros((nx, ny, 3), dtype='complex'))
            self.add_param(name+'b_pts', val=numpy.zeros((nx, ny, 3), dtype='complex'))
            self.add_param(name+'widths', val=numpy.zeros((nx-1, ny-1), dtype='complex'))
            self.add_output(name+'sec_forces', val=numpy.zeros((nx-1, ny-1, 3), dtype='complex'))
            self.add_output(name+'wake_circ', val=numpy.zeros((t+1, ny-1), dtype='complex'))

            if t > 0:
                self.add_param(name+'prev_circ', val=numpy.zeros((t, ny-1), dtype='complex'))


        self.tot_panels = tot_panels
        self.add_param('circulations', val=numpy.zeros((tot_panels)))
        self.add_param('alpha', val=3.)
        self.add_param('v', val=10.)
        self.add_param('rho', val=3.)
        self.surfaces = surfaces
        self.transient = transient

        self.mtx = numpy.zeros((tot_panels, tot_panels, 3), dtype="complex")
        self.v = numpy.zeros((tot_panels, 3), dtype="complex")
        self.t = t

        # self.deriv_options['type'] = 'fd'
        # self.deriv_options['form'] = 'central'

    def solve_nonlinear(self, params, unknowns, resids):
        circ = params['circulations']
        alpha = params['alpha'] * numpy.pi / 180.
        cosa = numpy.cos(alpha)
        sina = numpy.sin(alpha)

        # Assemble a different matrix here than the AIC_mtx from above; Note
        # that the collocation points used here are the midpoints of each
        # bound vortex filament, not the collocation points from above
        _assemble_AIC_mtx(self.mtx, params, self.surfaces, self.transient, skip=True)

        # Compute the induced velocities at the midpoints of the
        # bound vortex filaments
        for ind in xrange(3):
            self.v[:, ind] = self.mtx[:, :, ind].dot(circ)

        # Add the freestream velocity to the induced velocity so that
        # self.v is the total velocity seen at the point
        self.v[:, 0] += cosa * params['v']
        self.v[:, 2] += sina * params['v']

        i = 0
        for surface in self.surfaces:
            name = surface['name']
            nx = surface['num_x']
            ny = surface['num_y']
            num_panels = (nx - 1) * (ny - 1)

            b_pts = params[name+'b_pts']

            bound = b_pts[:-1, 1:, :] - b_pts[:-1, :-1, :]

            # Cross the obtained velocities with the bound vortex filament
            # vectors
            cross = numpy.cross(self.v[i:i+num_panels],
                                bound.reshape(-1, bound.shape[-1], order='F'))

            sec_forces = numpy.zeros((num_panels, 3), dtype='complex')

            circ_slice = circ[i:i+num_panels].reshape(nx-1, ny-1, order='F')
            cross_slice = cross.reshape(nx-1, ny-1, 3, order='F')

            # Compute the sectional forces acting on each panel
            for ind in xrange(3):
                sec_forces[:ny-1, ind] = circ_slice[0, :] * cross_slice[0, :, ind]

                for j in xrange(1, nx - 1):
                    sec_forces[j*(ny-1):(j+1)*(ny-1), ind] = \
                        (circ_slice[j, :] - circ_slice[j-1, :]) * cross_slice[j, :, ind]


            unknowns[name+'sec_forces'] = params['rho'] * sec_forces.reshape((nx-1, ny-1, 3), order='C')

            print unknowns[name+'sec_forces']

            unknowns[name+'wake_circ'][0, :] = circ_slice[-1, :]
            if self.t > 0:
                unknowns[name+'wake_circ'][1:, :] = params[name+'prev_circ']

            i += num_panels

    def linearize(self, params, unknowns, resids):
        """ Jacobian for forces."""

        jac = self.alloc_jacobian()

        fd_jac = self.complex_step_jacobian(params, unknowns, resids,
                                         fd_params=['alpha', 'circulations', 'v'],
                                         fd_states=[])
        jac.update(fd_jac)

        rho = params['rho'].real

        for surface in self.surfaces:
            name = surface['name']

            fd_jac = self.complex_step_jacobian(params, unknowns, resids,
                                             fd_params=[name+'b_pts',
                                                name+'def_mesh'],
                                             fd_states=[])
            jac.update(fd_jac)

            sec_forces = unknowns[name+'sec_forces'].real
            jac[name+'sec_forces', 'rho'] = sec_forces.flatten() / rho

        return jac


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
