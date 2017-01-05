""" Script to plot results from aero, struct, or aerostruct optimization.

Usage is
`python plot_all.py a` for aero only,
`python plot_all.py s` for struct only,
`python plot_all.py as` for aerostruct, or
`python plot_all.py __name__` for user-named database.

The script automatically appends '.db' to the provided name.
Ex: `python plot_all.py example` opens 'example.db'.

You can select a certain zoom factor for the 3d view by adding a number as a
last keyword.
The larger the number, the closer the view. Floats or ints are accepted.

Ex: `python plot_all.py a 1` a wider view than `python plot_all.py a 5`.

"""


from __future__ import division
import tkFont
import Tkinter as Tk
import sys

import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['axes.edgecolor'] = 'gray'
matplotlib.rcParams['axes.linewidth'] = 0.5
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,\
    NavigationToolbar2TkAgg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.animation as manimation

import numpy
import sqlitedict
import aluminum

#####################
# User-set parameters
#####################

if sys.argv[1] == 'as':
    filename = 'aerostruct'
elif sys.argv[1] == 'a':
    filename = 'aero'
elif sys.argv[1] == 's':
    filename = 'struct'
else:
    filename = sys.argv[1]

try:
    zoom_scale = sys.argv[2]
except:
    zoom_scale = 2.8

db_name = filename + '.db'


class Display(object):
    def __init__(self, db_name):

        self.root = Tk.Tk()
        self.root.wm_title("Viewer")

        # Initialize figure objects
        self.f = plt.figure(dpi=100, figsize=(12, 6), facecolor='white')
        self.canvas = FigureCanvasTkAgg(self.f, master=self.root)
        self.canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

        # Create frame to store options buttons and objects
        self.options_frame = Tk.Frame(self.root)
        self.options_frame.pack()

        toolbar = NavigationToolbar2TkAgg(self.canvas, self.root)
        toolbar.update()
        self.canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        self.ax = plt.subplot2grid((30, 8), (0, 0), rowspan=30,
                                   colspan=4, projection='3d')

        self.num_iters = 0
        self.db_name = db_name
        self.show_wing = True
        self.show_tube = True
        self.curr_pos = 0
        self.old_n = 0

        self.load_db()

        if self.show_wing and not self.show_tube:
            self.ax2 = plt.subplot2grid((30, 8), (0, 4), rowspan=15, colspan=4)
            self.ax3 = plt.subplot2grid((30, 8), (15, 4), rowspan=15, colspan=4)
        if self.show_tube and not self.show_wing:
            self.ax4 = plt.subplot2grid((30, 8), (0, 4), rowspan=10, colspan=4)
            self.ax5 = plt.subplot2grid((30, 8), (10, 4), rowspan=10, colspan=4)
            self.ax6 = plt.subplot2grid((30, 8), (20, 4), rowspan=10, colspan=4)
        if self.show_wing and self.show_tube:
            self.ax2 = plt.subplot2grid((30, 8), (0, 4), rowspan=6, colspan=4)
            self.ax3 = plt.subplot2grid((30, 8), (6, 4), rowspan=6, colspan=4)
            self.ax4 = plt.subplot2grid((30, 8), (12, 4), rowspan=6, colspan=4)
            self.ax5 = plt.subplot2grid((30, 8), (18, 4), rowspan=6, colspan=4)
            self.ax6 = plt.subplot2grid((30, 8), (24, 4), rowspan=6, colspan=4)

    def load_db(self):
        # Change for future versions of OpenMDAO, still in progress
        # self.db_metadata = sqlitedict.SqliteDict(self.db_name, 'metadata')
        # self.db = sqlitedict.SqliteDict(self.db_name, 'iterations')
        self.db = sqlitedict.SqliteDict(self.db_name, 'openmdao')

        self.twist = []
        self.mesh = []
        self.def_mesh = []
        self.r = []
        self.t = []
        sec_forces = []
        normals = []
        widths = []
        self.lift = []
        self.lift_ell = []
        self.vonmises = []
        alpha = []
        rho = []
        v = []
        self.CL = []
        self.AR = []
        self.S_ref = []
        self.obj = []
        self.modes = []
        self.freqs = []

        # Change for future versions of OpenMDAO
        # for tag in self.db_metadata:
        #     try:
        #         for item in self.db_metadata[tag]:
        #             for flag in self.db_metadata[tag][item]:
        #                 if 'is_objective' in flag:
        #                     self.obj_key = item
                #
                # except:
                #     pass

        for tag in self.db['metadata']:
            for item in self.db['metadata'][tag]:
                for flag in self.db['metadata'][tag][item]:
                    if 'is_objective' in flag:
                        self.obj_key = item

        for case_name, case_data in self.db.iteritems():
            # Only continue for these cases if there are more than one iteration
            # which occurs when we run an optimization case.
            if len(self.db.keys()) > 2:
                if "derivs" in case_name or "Driver" in case_name:
                    continue  # don't plot these cases

            if "metadata" in case_name:
                continue  # don't plot these cases

            names = []
            for key in case_data['Unknowns'].keys():
                if 'coupled' in key and 'loads' in key:
                    self.aerostruct = True
                    names.append(key.split('_')[:-1][0])
                elif 'mesh' in key and 'def_mesh' not in key and 'coupled' not in key:
                    self.aerostruct = False
                    names.append(key.split('.')[0])

            self.names = names
            n_names = len(names)
            try:
                self.obj.append(case_data['Unknowns'][self.obj_key])
            except AttributeError:
                pass

            # Loop through each of the surfaces
            for name in names:

                # A mesh exists for all types of cases
                self.mesh.append(case_data['Unknowns'][name+'.mesh'])

                # Check if this is an aerostructual case; treat differently
                # due to the way the problem is organized
                if not self.aerostruct:
                    try:
                        self.r.append(case_data['Unknowns'][name+'.r'])
                        self.t.append(case_data['Unknowns'][name+'.thickness'])
                        self.vonmises.append(
                            numpy.max(case_data['Unknowns'][name+'.vonmises'], axis=1))
                        # pick off only the first four modes
                        self.modes.append(case_data['Unknowns'][name+'.modes'][:, :4])
                        self.freqs.append(case_data['Unknowns'][name+'.freqs'])
                        self.show_tube = True
                    except:
                        self.show_tube = False
                    try:
                        self.def_mesh.append(case_data['Unknowns'][name+'.def_mesh'])
                        self.twist.append(case_data['Unknowns'][name+'.twist'])
                        normals.append(case_data['Unknowns'][name+'_geom.normals'])
                        widths.append(case_data['Unknowns'][name+'_geom.widths'])
                        sec_forces.append(case_data['Unknowns']['aero_states.' + name + '_sec_forces'])
                        self.CL.append(case_data['Unknowns'][name+'_perf.CL'])
                        self.S_ref.append(case_data['Unknowns'][name+'_geom.S_ref'])
                        self.show_wing = True
                    except:
                        self.show_wing = False
                else:
                    self.show_wing, self.show_tube = True, True
                    short_name = name.split('.')[1:][0]
                    self.r.append(case_data['Unknowns'][short_name+'.r'])
                    self.t.append(case_data['Unknowns'][short_name+'.thickness'])
                    self.vonmises.append(
                        numpy.max(case_data['Unknowns'][short_name+'_perf.vonmises'], axis=1))
                    self.def_mesh.append(case_data['Unknowns'][name+'.def_mesh'])
                    self.twist.append(case_data['Unknowns'][short_name+'.twist'])
                    normals.append(case_data['Unknowns'][name+'.normals'])
                    widths.append(case_data['Unknowns'][name+'.widths'])
                    sec_forces.append(case_data['Unknowns']['coupled.aero_states.' + short_name + '_sec_forces'])
                    # pick off only the first six modes
                    self.modes.append(case_data['Unknowns']['coupled.' + short_name + '.modes'][:, :12])
                    self.freqs.append(case_data['Unknowns']['coupled.' + short_name + '.freqs'])
                    self.CL.append(case_data['Unknowns'][short_name+'_perf.CL1'])
                    self.S_ref.append(case_data['Unknowns'][name+'.S_ref'])

            if self.show_wing:
                alpha.append(case_data['Unknowns']['alpha'] * numpy.pi / 180.)
                rho.append(case_data['Unknowns']['rho'])
                v.append(case_data['Unknowns']['v'])

        self.num_iters = numpy.max([int(len(self.mesh) / n_names) - 1, 0])

        symm_count = 0
        for mesh in self.mesh:
            if numpy.all(mesh[:, :, 1] >= -1e-8) or numpy.all(mesh[:, :, 1] <= 1e-8):
                symm_count += 1
        if symm_count == len(self.mesh):
            self.symmetry = True
        else:
            self.symmetry = False

        if self.show_wing:
            for i in range(self.num_iters + 1):
                for j, name in enumerate(names):
                    m_vals = self.mesh[i*n_names+j].copy()
                    cvec = m_vals[0, :, :] - m_vals[-1, :, :]
                    chords = numpy.sqrt(numpy.sum(cvec**2, axis=1))
                    chords = 0.5 * (chords[1:] + chords[:-1])
                    a = alpha[i]
                    cosa = numpy.cos(a)
                    sina = numpy.sin(a)
                    forces = numpy.sum(sec_forces[i*n_names+j], axis=0)
                    widths_ = numpy.mean(widths[i*n_names+j], axis=0)

                    lift = (-forces[:, 0] * sina + forces[:, 2] * cosa) / \
                        widths_/0.5/rho[i]/v[i]**2
                    # lift = (-forces[:, 0] * sina + forces[:, 2] * cosa)/chords/0.5/rho[i]/v[i]**2
                    # lift = (-forces[:, 0] * sina + forces[:, 2] * cosa)*chords/0.5/rho[i]/v[i]**2

                    if self.symmetry:
                        span = (m_vals[0, :, 1] / (m_vals[0, -1, 1] - m_vals[0, 0, 1]))
                        span = numpy.hstack((span[:-1], -span[::-1]))

                        lift = numpy.hstack((lift, lift[::-1]))

                        lift_area = numpy.sum(lift * (span[1:] - span[:-1]))

                        lift_ell = 2 * lift_area / numpy.pi * \
                            numpy.sqrt(1 - span**2)

                    else:
                        span = (m_vals[0, :, 1] / (m_vals[0, -1, 1] - m_vals[0, 0, 1]))
                        span = span - (span[0] + .5)

                        lift_area = numpy.sum(lift * (span[1:] - span[:-1]))

                        lift_ell = 4 * lift_area / numpy.pi * \
                            numpy.sqrt(1 - (2*span)**2)

                    self.lift.append(lift)
                    self.lift_ell.append(lift_ell)

                    wingspan = numpy.abs(m_vals[0, -1, 1] - m_vals[0, 0, 1])
                    self.AR.append(wingspan**2 / self.S_ref[i*n_names+j])

            # recenter def_mesh points for better viewing
            for i in range(self.num_iters + 1):
                center = numpy.zeros((3))
                for j in range(n_names):
                    center += numpy.mean(self.def_mesh[i*n_names+j], axis=(0,1))
                for j in range(n_names):
                    self.def_mesh[i*n_names+j] -= center / n_names

        # recenter mesh points for better viewing
        for i in range(self.num_iters + 1):
            center = numpy.zeros((3))
            for j in range(n_names):
                center += numpy.mean(self.mesh[i*n_names+j], axis=(0,1))
            for j in range(n_names):
                self.mesh[i*n_names+j] -= center / n_names

        if self.show_wing:
            self.min_twist, self.max_twist = self.get_list_limits(self.twist)
            diff = (self.max_twist - self.min_twist) * 0.05
            self.min_twist -= diff
            self.max_twist += diff
            self.min_l, self.max_l = self.get_list_limits(self.lift)
            self.min_le, self.max_le = self.get_list_limits(self.lift_ell)
            self.min_l, self.max_l = min(self.min_l, self.min_le), max(self.max_l, self.max_le)
            diff = (self.max_l - self.min_l) * 0.05
            self.min_l -= diff
            self.max_l += diff
        if self.show_tube:
            self.min_t, self.max_t = self.get_list_limits(self.t)
            diff = (self.max_t - self.min_t) * 0.05
            self.min_t -= diff
            self.max_t += diff

            self.min_vm, self.max_vm = self.get_list_limits(self.vonmises)
            diff = (self.max_vm - self.min_vm) * 0.05
            self.min_vm -= diff
            self.max_vm += diff

            self.min_modes, self.max_modes = self.get_list_limits(self.modes)
            diff = (self.max_modes - self.min_modes) * 0.05
            self.min_modes -= diff
            self.max_modes += diff

    def plot_sides(self):

        if self.show_wing:

            self.ax2.cla()
            self.ax2.locator_params(axis='y',nbins=5)
            self.ax2.locator_params(axis='x',nbins=3)
            self.ax2.set_ylim([self.min_twist, self.max_twist])
            self.ax2.set_xlim([-1, 1])
            self.ax2.set_ylabel('twist', rotation="horizontal", ha="right")

            self.ax3.cla()
            self.ax3.text(0.05, 0.8, 'elliptical',
                transform=self.ax3.transAxes, color='g')
            self.ax3.locator_params(axis='y',nbins=4)
            self.ax3.locator_params(axis='x',nbins=3)
            self.ax3.set_ylim([self.min_l, self.max_l])
            self.ax3.set_xlim([-1, 1])
            self.ax3.set_ylabel('lift', rotation="horizontal", ha="right")

        if self.show_tube:

            self.ax4.cla()
            self.ax4.locator_params(axis='y',nbins=4)
            self.ax4.locator_params(axis='x',nbins=3)
            self.ax4.set_ylim([self.min_t, self.max_t])
            self.ax4.set_xlim([-1, 1])
            self.ax4.set_ylabel('thickness', rotation="horizontal", ha="right")

            self.ax5.cla()
            self.ax5.locator_params(axis='y',nbins=4)
            self.ax5.locator_params(axis='x',nbins=3)
            self.ax5.set_ylim([self.min_vm, self.max_vm])
            self.ax5.set_ylim([0, 25e6])
            self.ax5.set_xlim([-1, 1])
            self.ax5.set_ylabel('von mises', rotation="horizontal", ha="right")
            self.ax5.axhline(aluminum.stress, c='r', lw=2, ls='--')
            self.ax5.text(0.05, 0.85, 'failure limit',
                transform=self.ax5.transAxes, color='r')

            self.ax6.cla()
            self.ax6.locator_params(axis='y',nbins=4)
            self.ax6.locator_params(axis='x',nbins=3)
            self.ax6.set_ylim([self.min_modes, self.max_modes])
            self.ax6.set_xlim([-1, 1])
            self.ax6.set_ylabel('mode shapes', rotation="horizontal", ha="right")

        for j, name in enumerate(self.names):
            m_vals = self.mesh[self.curr_pos+j].copy()
            span = m_vals[0, -1, 1] - m_vals[0, 0, 1]
            if self.symmetry:
                rel_span = (m_vals[0, :, 1] - m_vals[0, 0, 1]) / span - 1
                rel_span = numpy.hstack((rel_span[:-1], -rel_span[::-1]))
                span_diff = ((m_vals[0, :-1, 1] + m_vals[0, 1:, 1]) / 2 - m_vals[0, 0, 1]) / span - 1
                span_diff = numpy.hstack((span_diff, -span_diff[::-1]))
            else:
                rel_span = (m_vals[0, :, 1] - m_vals[0, 0, 1]) * 2 / span - 1
                span_diff = ((m_vals[0, :-1, 1] + m_vals[0, 1:, 1]) / 2 - m_vals[0, 0, 1]) * 2 / span - 1

            if self.show_wing:
                t_vals = self.twist[self.curr_pos+j]
                l_vals = self.lift[self.curr_pos+j]
                le_vals = self.lift_ell[self.curr_pos+j]

                if self.symmetry:
                    t_vals = numpy.hstack((t_vals[:-1], t_vals[::-1]))

                self.ax2.plot(rel_span, t_vals, lw=2, c='b')
                self.ax3.plot(rel_span, le_vals, '--', lw=2, c='g')
                self.ax3.plot(span_diff, l_vals, lw=2, c='b')

            if self.show_tube:
                thick_vals = self.t[self.curr_pos+j]
                vm_vals = self.vonmises[self.curr_pos+j]

                if self.symmetry:
                    thick_vals = numpy.hstack((thick_vals, thick_vals[::-1]))
                    vm_vals = numpy.hstack((vm_vals, vm_vals[::-1]))

                self.ax4.plot(span_diff, thick_vals, lw=2, c='b')
                self.ax5.plot(span_diff, vm_vals, lw=2, c='b')

                # Pick off only the second dof mode
                dof = 1
                modes = self.modes[self.curr_pos+j]
                modes = modes[dof::6, :]

                color_cycle = ['b', 'g', 'r', 'k', 'c'] * 10

                # Loop over each mode to plot
                for i in range(modes.shape[1]):
                    half = int(modes.shape[0] / 2)

                    # Mirror the modes if it is symmetric
                    if self.symmetry:
                        modes_ = numpy.hstack((modes[:, i], numpy.array([0.]), modes[:, i][::-1]))
                        self.ax6.plot(rel_span, modes_, lw=2, c=color_cycle[i])
                    else:
                        modes_ = numpy.hstack((modes[:half, i], numpy.array([0.]), modes[half:, i]))
                        self.ax6.plot(rel_span, modes_, lw=2, c=color_cycle[i])

    def plot_wing(self):

        n_names = len(self.names)
        self.ax.cla()
        az = self.ax.azim
        el = self.ax.elev
        dist = self.ax.dist

        for j, name in enumerate(self.names):
            mesh0 = self.mesh[self.curr_pos+j].copy()

            self.ax.set_axis_off()

            if self.show_wing:
                def_mesh0 = self.def_mesh[self.curr_pos+j]
                x = mesh0[:, :, 0]
                y = mesh0[:, :, 1]
                z = mesh0[:, :, 2]

                try:  # show deformed mesh option may not be available
                    if self.show_def_mesh.get():
                        x_def = def_mesh0[:, :, 0]
                        y_def = def_mesh0[:, :, 1]
                        z_def = def_mesh0[:, :, 2]

                        self.c2.grid(row=0, column=3, padx=5, sticky=Tk.W)
                        if self.ex_def.get():
                            z_def = (z_def - z) * 10 + z_def
                            def_mesh0 = (def_mesh0 - mesh0) * 30 + def_mesh0
                        else:
                            def_mesh0 = (def_mesh0 - mesh0) * 2 + def_mesh0
                        self.ax.plot_wireframe(x_def, y_def, z_def, rstride=1, cstride=1, color='k')
                        self.ax.plot_wireframe(x, y, z, rstride=1, cstride=1, color='k', alpha=.3)
                    else:
                        self.ax.plot_wireframe(x, y, z, rstride=1, cstride=1, color='k')
                        self.c2.grid_forget()
                except:
                    self.ax.plot_wireframe(x, y, z, rstride=1, cstride=1, color='k')

            if self.show_tube:
                r0 = self.r[self.curr_pos+j]
                t0 = self.t[self.curr_pos+j]
                colors = t0
                colors = colors / numpy.max(colors)
                num_circ = 12
                fem_origin = 0.35
                n = mesh0.shape[1]
                p = numpy.linspace(0, 2*numpy.pi, num_circ)
                if self.show_wing:
                    if self.show_def_mesh.get():
                        mesh0[:, :, 2] = def_mesh0[:, :, 2]
                for i, thick in enumerate(t0):
                    r = numpy.array((r0[i], r0[i]))
                    R, P = numpy.meshgrid(r, p)
                    X, Z = R*numpy.cos(P), R*numpy.sin(P)
                    chords = mesh0[-1, :, 0] - mesh0[0, :, 0]
                    comp = fem_origin * chords + mesh0[0, :, 0]
                    X[:, 0] += comp[i]
                    X[:, 1] += comp[i+1]
                    Z[:, 0] += fem_origin * (mesh0[-1, i, 2] - mesh0[0, i, 2]) + mesh0[0, i, 2]
                    Z[:, 1] += fem_origin * (mesh0[-1, i+1, 2] - mesh0[0, i+1, 2]) + mesh0[0, i+1, 2]
                    Y = numpy.empty(X.shape)
                    Y[:] = numpy.linspace(mesh0[0, i, 1], mesh0[0, i+1, 1], 2)
                    col = numpy.zeros(X.shape)
                    col[:] = colors[i]
                    try:
                        self.ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                            facecolors=cm.viridis(col), linewidth=0)
                    except:
                        self.ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                            facecolors=cm.coolwarm(col), linewidth=0)

        lim = 0.
        for j in range(n_names):
            ma = numpy.max(self.mesh[self.curr_pos*n_names+j], axis=(0,1,2))
            if ma > lim:
                lim = ma
        lim /= float(zoom_scale)
        self.ax.auto_scale_xyz([-lim, lim], [-lim, lim], [-lim, lim])
        self.ax.set_title("Major Iteration: {}".format(self.curr_pos))

        try:
            round_to_n = lambda x, n: round(x, -int(numpy.floor(numpy.log10(abs(x)))) + (n - 1))
            obj_val = round_to_n(self.obj[self.curr_pos], 7)
            self.ax.text2D(.55, .05, self.obj_key + ': {}'.format(obj_val),
                transform=self.ax.transAxes, color='k')
        except IndexError:
            pass

        # TODO: fix this here only for dynamic structural analysis
        # freq_val = round_to_n(numpy.min(self.freqs[self.curr_pos])/(2*numpy.pi), 3)
        # self.ax.text2D(.55, .0, 'lowest freq: {} Hz'.format(freq_val),
        #     transform=self.ax.transAxes, color='k')

        self.ax.view_init(elev=el, azim=az)  # Reproduce view
        self.ax.dist = dist

    def save_video(self):
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Movie', artist='Matplotlib')
        writer = FFMpegWriter(fps=5, metadata=metadata, bitrate=3000)

        with writer.saving(self.f, "movie.mp4", 100):
            self.curr_pos = 0
            self.update_graphs()
            self.f.canvas.draw()
            plt.draw()
            for i in range(10):
                writer.grab_frame()

            for i in range(self.num_iters):
                self.curr_pos = i
                self.update_graphs()
                self.f.canvas.draw()
                plt.draw()
                writer.grab_frame()

            self.curr_pos = self.num_iters
            self.update_graphs()
            self.f.canvas.draw()
            plt.draw()
            for i in range(20):
                writer.grab_frame()

    def update_graphs(self, e=None):
        if e is not None:
            self.curr_pos = int(e)
            self.curr_pos = self.curr_pos % (self.num_iters + 1)

        self.plot_wing()
        self.plot_sides()
        self.canvas.show()

    def check_length(self):
        # Load the current sqlitedict
        db = sqlitedict.SqliteDict(self.db_name, 'openmdao')

        # Get the number of current iterations
        self.num_iters = max(db.keys()[-1].split('/'))

    def get_list_limits(self, input_list):
        list_min = 1.e20
        list_max = -1.e20
        for list_ in input_list:
            mi = numpy.min(list_)
            if mi < list_min:
                list_min = mi
            ma = numpy.max(list_)
            if ma > list_max:
                list_max = ma

        return list_min, list_max


    def auto_ref(self):
        """
        Automatically refreshes the history file, which is
        useful if examining a running optimization.
        """
        if self.var_ref.get():
            self.root.after(800, self.auto_ref)
            self.check_length()
            self.update_graphs()

            # Check if the sqlitedict file has change and if so, fully
            # load in the new file.
            if self.num_iters > self.old_n:
                self.load_db()
                self.old_n = self.num_iters
                self.draw_slider()

    def save_image(self):
        fname = 'fig' + '.png'
        plt.savefig(fname)

    def quit(self):
        """
        Destroy GUI window cleanly if quit button pressed.
        """
        self.root.quit()
        self.root.destroy()

    def draw_slider(self):
        # scale to choose iteration to view
        self.w = Tk.Scale(
            self.options_frame,
            from_=0, to=self.num_iters,
            orient=Tk.HORIZONTAL,
            resolution=1,
            font=tkFont.Font(family="Helvetica", size=10),
            command=self.update_graphs,
            length=200)

        if self.curr_pos == self.num_iters - 1 or self.curr_pos == 0:
            self.curr_pos = self.num_iters
        self.w.set(self.curr_pos)
        self.w.grid(row=0, column=1, padx=5, sticky=Tk.W)

    def draw_GUI(self):
        """
        Create the frames and widgets in the bottom section of the canvas.
        """
        font = tkFont.Font(family="Helvetica", size=10)

        lab_font = Tk.Label(
            self.options_frame,
            text="Iteration number:",
            font=font)
        lab_font.grid(row=0, column=0, sticky=Tk.S)

        self.draw_slider()

        if self.show_wing and self.show_tube:
            # checkbox to show deformed mesh
            self.show_def_mesh = Tk.IntVar()
            c1 = Tk.Checkbutton(
                self.options_frame,
                text="Show deformed mesh",
                variable=self.show_def_mesh,
                command=self.update_graphs,
                font=font)
            c1.grid(row=0, column=2, padx=5, sticky=Tk.W)

            # checkbox to exaggerate deformed mesh
            self.ex_def = Tk.IntVar()
            self.c2 = Tk.Checkbutton(
                self.options_frame,
                text="Exaggerate deformations",
                variable=self.ex_def,
                command=self.update_graphs,
                font=font)
            self.c2.grid(row=0, column=3, padx=5, sticky=Tk.W)

        # Option to automatically refresh history file
        # especially useful for currently running optimizations
        self.var_ref = Tk.IntVar()
        # self.var_ref.set(1)
        c11 = Tk.Checkbutton(
            self.options_frame,
            text="Automatically refresh",
            variable=self.var_ref,
            command=self.auto_ref,
            font=font)
        c11.grid(row=0, column=4, sticky=Tk.W, pady=6)

        button = Tk.Button(
            self.options_frame,
            text='Save video',
            command=self.save_video,
            font=font)
        button.grid(row=0, column=5, padx=5, sticky=Tk.W)

        button4 = Tk.Button(
            self.options_frame,
            text='Save image',
            command=self.save_image,
            font=font)
        button4.grid(row=0, column=6, padx=5, sticky=Tk.W)

        button5 = Tk.Button(
            self.options_frame,
            text='Quit',
            command=self.quit,
            font=font)
        button5.grid(row=0, column=7, padx=5, sticky=Tk.W)

        self.auto_ref()

def disp_plot(db_name):
    disp = Display(db_name)
    disp.draw_GUI()
    plt.tight_layout()
    disp.root.protocol("WM_DELETE_WINDOW", disp.quit)
    Tk.mainloop()

if __name__ == '__main__':
    disp_plot(db_name)
