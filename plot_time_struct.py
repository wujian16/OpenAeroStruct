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
zoom_scale = 2.8
db_name = 'struct.db'

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
        self.ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)

        self.num_iters = 0
        self.db_name = db_name
        self.show_wing = False
        self.show_tube = True
        self.curr_pos = 0
        self.old_n = 0

        self.load_db()

    def load_db(self):
        # Change for future versions of OpenMDAO, still in progress
        # self.db_metadata = sqlitedict.SqliteDict(self.db_name, 'metadata')
        # self.db = sqlitedict.SqliteDict(self.db_name, 'iterations')
        self.db = sqlitedict.SqliteDict(self.db_name, 'openmdao')

        self.disp = []
        self.mesh = []
        self.r = []
        self.t = []
        self.modes = []
        self.freqs = []

        for case_name, case_data in self.db.iteritems():
            if "metadata" in case_name or "derivs" in case_name:
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

            # Loop through each of the surfaces
            for name in names:

                # A mesh exists for all types of cases
                self.mesh.append(case_data['Unknowns'][name+'.mesh'])

                max_t = 1
                for key in case_data['Unknowns'].keys():
                    if 'disp_' in key:
                        num = int(key.split('_')[-1])
                        if num > max_t:
                            max_t = num
                self.num_iters = max_t-1

                # Check if this is an aerostructual case; treat differently
                # due to the way the problem is organized
                if not self.aerostruct:
                    self.r.append(case_data['Unknowns'][name+'.r'])
                    self.t.append(case_data['Unknowns'][name+'.thickness'])
                    for t in range(max_t):
                        self.disp.append(case_data['Unknowns']['wing.disp_'+str(t)])
                    # pick off only the first four modes
                    self.modes.append(case_data['Unknowns'][name+'.modes'][:, :4])
                    self.freqs.append(case_data['Unknowns'][name+'.freqs'])
                    self.show_tube = True


        symm_count = 0
        for mesh in self.mesh:
            if numpy.all(mesh[:, :, 1] >= -1e-8) or numpy.all(mesh[:, :, 1] <= 1e-8):
                symm_count += 1
        if symm_count == len(self.mesh):
            self.symmetry = True
        else:
            self.symmetry = False

        self.min_disp, self.max_disp = self.get_list_limits(self.disp)
        diff = (self.max_disp - self.min_disp) * 0.05
        self.min_disp -= diff
        self.max_disp += diff


    def plot_sides(self):
        for j, name in enumerate(self.names):
            m_vals = self.mesh[0].copy()
            span = m_vals[0, -1, 1] - m_vals[0, 0, 1]
            if self.symmetry:
                rel_span = (m_vals[0, :, 1] - m_vals[0, 0, 1]) / span - 1
                rel_span = numpy.hstack((rel_span[:-1], -rel_span[::-1]))
                span_diff = ((m_vals[0, :-1, 1] + m_vals[0, 1:, 1]) / 2 - m_vals[0, 0, 1]) / span - 1
                span_diff = numpy.hstack((span_diff, -span_diff[::-1]))
            else:
                rel_span = (m_vals[0, :, 1] - m_vals[0, 0, 1]) * 2 / span - 1
                span_diff = ((m_vals[0, :-1, 1] + m_vals[0, 1:, 1]) / 2 - m_vals[0, 0, 1]) * 2 / span - 1

            if self.show_tube:
                self.ax.set_ylim([self.min_disp, self.max_disp])
                disp = self.disp[self.curr_pos]
                half = int(len(rel_span) / 2)
                if self.symmetry:
                    rel_span = rel_span[half:]
                self.ax.plot(rel_span, disp[:, 2])
                self.ax.set_xlabel('Normalized span')
                self.ax.set_ylabel('Displacement (m)')


    def plot_wing(self):
        n_names = len(self.names)
        self.ax.cla()

        for j, name in enumerate(self.names):
            mesh0 = self.mesh[0]

    def save_video(self):
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Movie', artist='Matplotlib')
        writer = FFMpegWriter(fps=60, metadata=metadata, bitrate=3000)

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
