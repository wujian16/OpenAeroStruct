""" Manipulate geometry mesh based on high-level design parameters. """

from __future__ import division
import numpy
from numpy import cos, sin, tan

from openmdao.api import Component

from b_spline import get_bspline_mtx
from crm_data import crm_base_mesh

def rotate(mesh, thetas):
    """ Compute rotation matrices given mesh and rotation angles in degrees.

    Parameters
    ----------
    mesh : array_like
        Nodal mesh defining the initial aerodynamic surface.
    thetas : array_like
        1-D array of rotation angles for each wing slice in degrees.

    Returns
    -------
    mesh : array_like
        Nodal mesh defining the twisted aerodynamic surface.

    """

    te = mesh[-1]
    le = mesh[ 0]
    quarter_chord = 0.25 * te + 0.75 * le

    ny = mesh.shape[1]
    nx = mesh.shape[0]

    rad_thetas = thetas * numpy.pi / 180.

    mats = numpy.zeros((ny, 3, 3), dtype="complex")
    mats[:, 0, 0] = cos(rad_thetas)
    mats[:, 0, 2] = sin(rad_thetas)
    mats[:, 1, 1] = 1
    mats[:, 2, 0] = -sin(rad_thetas)
    mats[:, 2, 2] = cos(rad_thetas)
    for ix in range(nx):
        row = mesh[ix]
        row[:] = numpy.einsum("ikj, ij -> ik", mats, row - quarter_chord)
        row += quarter_chord
    return mesh


def sweep(mesh, angle, symmetry):
    """ Apply shearing sweep. Positive sweeps back.

    Parameters
    ----------
    mesh : array_like
        Nodal mesh defining the initial aerodynamic surface.
    angle : float
        Shearing sweep angle in degrees.
    symmetry : boolean
        Flag set to true if surface is reflected about y=0 plane.

    Returns
    -------
    mesh : array_like
        Nodal mesh defining the swept aerodynamic surface.

    """

    num_x, num_y, _ = mesh.shape
    le = mesh[0]
    p180 = numpy.pi / 180
    tan_theta = tan(p180*angle)

    if symmetry:
        y0 = le[-1, 1]
        dx = -(le[:, 1] - y0) * tan_theta

    else:
        ny2 = int((num_y - 1) / 2)
        y0 = le[ny2, 1]

        dx_right = (le[ny2:, 1] - y0) * tan_theta
        dx_left = -(le[:ny2, 1] - y0) * tan_theta
        dx = numpy.hstack((dx_left, dx_right))

    for i in xrange(num_x):
        mesh[i, :, 0] += dx

    return mesh


def dihedral(mesh, angle, symmetry):
    """ Apply dihedral angle. Positive angles up.

    Parameters
    ----------
    mesh : array_like
        Nodal mesh defining the initial aerodynamic surface.
    angle : float
        Dihedral angle in degrees.
    symmetry : boolean
        Flag set to true if surface is reflected about y=0 plane.

    Returns
    -------
    mesh : array_like
        Nodal mesh defining the aerodynamic surface with dihedral angle.

    """

    num_x, num_y, _ = mesh.shape
    le = mesh[0]
    p180 = numpy.pi / 180
    tan_theta = tan(p180*angle)

    if symmetry:
        y0 = le[-1, 1]
        dx = -(le[:, 1] - y0) * tan_theta

    else:
        ny2 = int((num_y-1) / 2)
        y0 = le[ny2, 1]
        dx_right = (le[ny2:, 1] - y0) * tan_theta
        dx_left = -(le[:ny2, 1] - y0) * tan_theta
        dx = numpy.hstack((dx_left, dx_right))

    for i in xrange(num_x):
        mesh[i, :, 2] += dx

    return mesh



def stretch(mesh, length):
    """ Stretch mesh in spanwise direction to reach specified length.

    Parameters
    ----------
    mesh : array_like
        Nodal mesh defining the initial aerodynamic surface.
    length : float
        Relative stetch ratio in the spanwise direction.

    Returns
    -------
    mesh : array_like
        Nodal mesh defining the stretched aerodynamic surface.

    """

    le = mesh[0]

    num_x, num_y, _ = mesh.shape

    span = le[-1, 1] - le[0, 1]
    dy = (length - span) / (num_y - 1) * numpy.arange(1, num_y)

    for i in xrange(num_x):
        mesh[i, 1:, 1] += dy

    return mesh


def taper(mesh, taper_ratio, symmetry):
    """ Alter the spanwise chord to produce a tapered wing.

    Parameters
    ----------
    mesh : array_like
        Nodal mesh defining the initial aerodynamic surface.
    taper_ratio : float
        Taper ratio for the wing; 1 is untapered, 0 goes to a point.
    symmetry : boolean
        Flag set to true if surface is reflected about y=0 plane.

    Returns
    -------
    mesh : array_like
        Nodal mesh defining the tapered aerodynamic surface.

    """

    le = mesh[0]
    te = mesh[-1]
    num_x, num_y, _ = mesh.shape
    center_chord = .5 * te + .5 * le

    if symmetry:
        taper = numpy.linspace(1, taper_ratio, num_y)[::-1]

        jac = get_bspline_mtx(num_y, num_y, order=2)
        taper = jac.dot(taper)

        for i in xrange(num_x):
            for ind in xrange(3):
                mesh[i, :, ind] = (mesh[i, :, ind] - center_chord[:, ind]) * \
                    taper + center_chord[:, ind]

    else:
        ny2 = int((num_y + 1) / 2)
        taper = numpy.linspace(1, taper_ratio, ny2)[::-1]

        jac = get_bspline_mtx(ny2, ny2, order=2)
        taper = jac.dot(taper)

        dx = numpy.hstack((taper, taper[::-1][1:]))

        for i in xrange(num_x):
            for ind in xrange(3):
                mesh[i, :, ind] = (mesh[i, :, ind] - center_chord[:, ind]) * \
                    dx + center_chord[:, ind]

    return mesh


def mirror(mesh, right_side=True):
    """
    Take a half geometry and mirror it across the symmetry plane.

    Parameters
    ----------
    mesh : array_like
        Nodal mesh defining half the initial aerodynamic surface.
    right_side : boolean
        If right_side==True, it mirrors from right to left,
        assuming that the first point is on the symmetry plane. Else
        it mirrors from left to right, assuming the last point is on the
        symmetry plane.

    Returns
    -------
    mesh : array_like
        Nodal mesh defining the mirrored aerodynamic surface.

    """

    num_x, num_y, _ = mesh.shape

    new_mesh = numpy.empty((num_x, 2 * num_y - 1, 3), dtype='complex')

    mirror_y = numpy.ones(mesh.shape, dtype='complex')
    mirror_y[:, :, 1] *= -1.0

    if right_side:
        new_mesh[:, :num_y, :] = mesh[:, ::-1, :] * mirror_y
        new_mesh[:, num_y:, :] = mesh[:,   1:, :]
    else:
        new_mesh[:, :num_y, :] = mesh[:, ::-1, :]
        new_mesh[:, num_y:, :] = mesh[:,   1:, :] * mirror_y[:, 1:, :]

    # shift so 0 is at the left wing tip (structures wants it that way)
    y0 = new_mesh[0, 0, 1]
    new_mesh[:, :, 1] -= y0

    return new_mesh


def gen_crm_mesh(n_points_inboard=2, n_points_outboard=2,
                 num_x=2, mesh=crm_base_mesh):
    """
    Build the right hand side of the CRM wing with specified number
    of inboard and outboard panels, mirror it, add a specified number
    of chordwise nodes, and output a final full CRM mesh.

    Parameters
    ----------
    n_points_inboard : int
        Number of spanwise points between the wing root and yehudi break per
        wing side.
    n_points_outboard : int
        Number of spanwise points between the yehudi break and wingtip per
        wing side.
    num_x : int
        Number of chordwise points.
    mesh : array_like
        Base mesh with the leading and trailing edges defined that we use
        to populate the final mesh

    Returns
    -------
    full_mesh : array_like
        Final aerodynamic mesh representing the CRM wing.

    """

    # LE pre-yehudi
    s1 = (mesh[0, 1, 0] - mesh[0, 0, 0]) / (mesh[0, 1, 1] - mesh[0, 0, 1])
    o1 = mesh[0, 0, 0]

    # TE pre-yehudi
    s2 = (mesh[1, 1, 0] - mesh[1, 0, 0]) / (mesh[1, 1, 1] - mesh[1, 0, 1])
    o2 = mesh[1, 0, 0]

    # LE post-yehudi
    s3 = (mesh[0, 2, 0] - mesh[0, 1, 0]) / (mesh[0, 2, 1] - mesh[0, 1, 1])
    o3 = mesh[0, 2, 0] - s3 * mesh[0, 2, 1]

    # TE post-yehudi
    s4 = (mesh[1, 2, 0] - mesh[1, 1, 0]) / (mesh[1, 2, 1] - mesh[1, 1, 1])
    o4 = mesh[1, 2, 0] - s4 * mesh[1, 2, 1]

    n_points_total = n_points_inboard + n_points_outboard - 1
    half_mesh = numpy.zeros((2, n_points_total, 3), dtype='complex')

    # generate inboard points
    dy = (mesh[0, 1, 1] - mesh[0, 0, 1]) / (n_points_inboard - 1)
    for i in xrange(n_points_inboard):
        y = half_mesh[0, i, 1] = i * dy
        half_mesh[0, i, 0] = s1 * y + o1  # le point
        half_mesh[1, i, 1] = y
        half_mesh[1, i, 0] = s2 * y + o2  # te point

    yehudi_break = mesh[0, 1, 1]
    # generate outboard points
    dy = (mesh[0, 2, 1] - mesh[0, 1, 1]) / (n_points_outboard - 1)
    for j in xrange(n_points_outboard):
        i = j + n_points_inboard - 1
        y = half_mesh[0, i, 1] = j * dy + yehudi_break
        half_mesh[0, i, 0] = s3 * y + o3  # le point
        half_mesh[1, i, 1] = y
        half_mesh[1, i, 0] = s4 * y + o4  # te point

    full_mesh = mirror(half_mesh)
    full_mesh = add_chordwise_panels(full_mesh, num_x)
    full_mesh[:, :, 1] -= numpy.mean(full_mesh[:, :, 1])
    return full_mesh


def add_chordwise_panels(mesh, num_x):
    """ Divide the wing into multiple chordwise panels.

    Parameters
    ----------
    mesh : array_like
        Nodal mesh defining the initial aerodynamic surface with only
        the leading and trailing edges defined.
    num_x : float
        Desired number of chordwise node points for the final mesh.

    Returns
    -------
    new_mesh : array_like
        Nodal mesh defining the final aerodynamic surface with the
        specified number of chordwise node points.

    """

    le = mesh[ 0, :, :]
    te = mesh[-1, :, :]

    new_mesh = numpy.zeros((num_x, mesh.shape[1], 3), dtype='complex')
    new_mesh[ 0, :, :] = le
    new_mesh[-1, :, :] = te

    for i in xrange(1, num_x-1):
        w = float(i) / (num_x - 1)
        new_mesh[i, :, :] = (1 - w) * le + w * te

    return new_mesh


def gen_mesh(num_x, num_y, span, chord, span_cos_spacing=0., chord_cos_spacing=0., wing_type='rect'):
    """ Generate simple rectangular wing mesh.

    Parameters
    ----------
    num_x : float
        Desired number of chordwise node points for the final mesh.
    num_y : float
        Desired number of chordwise node points for the final mesh.
    span : float
        Total wingspan.
    chord : float
        Root chord.
    span_cos_spacing : float (optional)
        Blending ratio of uniform and cosine spacing in the spanwise direction.
        A value of 0. corresponds to uniform spacing and a value of 1.
        corresponds to regular cosine spacing. This increases the number of
        spanwise node points near the wingtips.
    chord_cos_spacing : float (optional)
        Blending ratio of uniform and cosine spacing in the chordwise direction.
        A value of 0. corresponds to uniform spacing and a value of 1.
        corresponds to regular cosine spacing. This increases the number of
        chordwise node points near the wingtips.

    Returns
    -------
    mesh : array_like
        Rectangular nodal mesh defining the final aerodynamic surface with the
        specified parameters.

    """

    mesh = numpy.zeros((num_x, num_y, 3), dtype='complex')
    ny2 = (num_y + 1) / 2
    beta = numpy.linspace(0, numpy.pi/2, ny2)

    # mixed spacing with span_cos_spacing as a weighting factor
    # this is for the spanwise spacing
    cosine = .5 * numpy.cos(beta)  # cosine spacing
    uniform = numpy.linspace(0, .5, ny2)[::-1]  # uniform spacing
    half_wing = cosine * span_cos_spacing + (1 - span_cos_spacing) * uniform
    full_wing = numpy.hstack((-half_wing[:-1], half_wing[::-1])) * span

    nx2 = (num_x + 1) / 2
    beta = numpy.linspace(0, numpy.pi/2, nx2)

    if wing_type == 'rect':
        # mixed spacing with span_cos_spacing as a weighting factor
        # this is for the chordwise spacing
        cosine = .5 * numpy.cos(beta)  # cosine spacing
        uniform = numpy.linspace(0, .5, nx2)[::-1]  # uniform spacing
        half_wing = cosine * chord_cos_spacing + (1 - chord_cos_spacing) * uniform
        full_wing_x = numpy.hstack((-half_wing[:-1], half_wing[::-1])) * chord

        # Special case if there are only 2 chordwise nodes
        if num_x <= 2:
            full_wing_x = numpy.array([0., chord])

        for ind_x in xrange(num_x):
            for ind_y in xrange(num_y):
                mesh[ind_x, ind_y, :] = [full_wing_x[ind_x], full_wing[ind_y], 0]

    else:
        full_wing_x = numpy.zeros((num_x, num_y))
        full_wing_x[0, :] = -numpy.sqrt((chord*.25)**2 * (1 - full_wing**2/(span/2)**2))
        full_wing_x[1, :] = numpy.sqrt((chord*.75)**2 * (1 - full_wing**2/(span/2)**2))

        for ind_x in xrange(2):
            for ind_y in xrange(num_y):
                mesh[ind_x, ind_y, :] = [full_wing_x[ind_x, ind_y], full_wing[ind_y], 0]

        if num_x >= 2:
            print
            print 'WARNING: Only "num_x == 2" supported for elliptical wing for now.'
            print

    return mesh


class GeometryMesh(Component):
    """
    OpenMDAO component that performs mesh manipulation functions.

    """

    def __init__(self, surface):
        super(GeometryMesh, self).__init__()

        self.surface = surface

        self.ny = surface['num_y']
        self.nx = surface['num_x']
        self.n = self.nx * self.ny
        self.mesh = surface['mesh']
        name = surface['name']

        self.add_param('span', val=58.7630524)
        self.add_param('sweep', val=0.)
        self.add_param('dihedral', val=0.)
        self.add_param('twist', val=numpy.zeros(self.ny), dtype='complex')
        self.add_param('taper', val=1.)
        self.add_output('mesh', val=self.mesh)

        self.symmetry = surface['symmetry']

        self.deriv_options['type'] = 'cs'
        # self.deriv_options['form'] = 'central'

    def solve_nonlinear(self, params, unknowns, resids):
        name = self.surface['name']
        mesh = self.mesh.copy()
        # stretch(mesh, params['span'])
        sweep(mesh, params['sweep'], self.symmetry)
        rotate(mesh, params['twist'])
        dihedral(mesh, params['dihedral'], self.symmetry)
        taper(mesh, params['taper'], self.symmetry)

        unknowns['mesh'] = mesh

    def linearize(self, params, unknowns, resids):
        name = self.surface['name']

        jac = self.alloc_jacobian()

        fd_jac = self.complex_step_jacobian(params, unknowns, resids,
                                            fd_params=['span', 'sweep',
                                                       'dihedral', 'twist',
                                                       'taper'],
                                            fd_states=[])
        jac.update(fd_jac)
        return jac


class Bspline(Component):
    """
    General function to translate from control points to actual points
    using a b-spline representation.

    """

    def __init__(self, cpname, ptname, jac):
        super(Bspline, self).__init__()
        self.cpname = cpname
        self.ptname = ptname
        self.jac = jac
        self.add_param(cpname, val=numpy.zeros(jac.shape[1]))
        self.add_output(ptname, val=numpy.zeros(jac.shape[0]))

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns[self.ptname] = self.jac.dot(params[self.cpname])

    def linearize(self, params, unknowns, resids):
        return {(self.ptname, self.cpname): self.jac}


class LinearInterp(Component):
    """ Linear interpolation used to create linearly varying parameters. """

    def __init__(self, num_y, name):
        super(LinearInterp, self).__init__()

        self.add_param('linear_'+name, val=numpy.zeros(2))
        self.add_output(name, val=numpy.zeros(num_y))

        self.deriv_options['type'] = 'cs'
        self.deriv_options['form'] = 'central'
        #self.deriv_options['extra_check_partials_form'] = "central"

        self.num_y = num_y
        self.vname = name

    def solve_nonlinear(self, params, unknowns, resids):
        a, b = params['linear_'+self.vname]

        if self.num_y % 2 == 0:
            imax = int(self.num_y/2)
        else:
            imax = int((self.num_y+1)/2)
        for ind in xrange(imax):
            w = 1.0*ind/(imax-1)

            unknowns[self.vname][ind] = a*(1-w) + b*w
            unknowns[self.vname][-1-ind] = a*(1-w) + b*w


if __name__ == "__main__":
    """ Test mesh generation and view results in .html file. """

    import plotly.offline as plt
    import plotly.graph_objs as go

    from plot_tools import wire_mesh, build_layout

    thetas = numpy.zeros(20)
    thetas[10:] += 10

    mesh = gen_crm_mesh(3, 3)

    # new_mesh = rotate(mesh, thetas)

    # new_mesh = sweep(mesh, 20)

    new_mesh = stretch(mesh, 100)

    # wireframe_orig = wire_mesh(mesh)
    wireframe_new = wire_mesh(new_mesh)
    layout = build_layout()

    fig = go.Figure(data=wireframe_new, layout=layout)
    plt.plot(fig, filename="wing_3d.html")
