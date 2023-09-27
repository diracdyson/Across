"""
    Version 1.1.

    This program is used to calculate the flux motion of ReBCO stack/coil.
    The calculation theory can be found at: https://dx.doi.org/10.1088/1361-6668/abc567
    Copyright (C) 2020  Beijing Eastforce Superconducting Technology Co., Ltd.

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program.

    If you have any question about this program, please contact me.

    Dr. Lingfeng Lai
    Email:lailingfeng@eastfs.com
    2020-11-20

    Python Version: 3.6
    Modules used: numpy, matplotlib, scipy
"""
import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import time
import pickle


class YBCOtape:
    """
    class for YBCO tape
    """
    def __init__(self, a=4e-3, b=1e-6, nvalue=38, jc0=2.8e10, B0=0.04265, k=0.29515, alpha=0.7):
        """
        define the YBCO tape
        :param a: tape width
        :param b: tape thickness
        :param nvalue: n value
        :param jc0: ciricle current density
        :param B0: parameter for Kim model
        :param k:  parameter for Kim model
        :param alpha:  parameter for Kim model
        """
        self.a = a
        self.b = b
        self.nvalue = nvalue
        self.jc = np.array(jc0)
        self.ic = jc0 * a * b
        self.B0 = B0
        self.k = k
        self.alpha = alpha

    def jcB(self, Bpara, Bperp):
        """
        calculte the critical curent density in magnetic field parallel field Bpara and perpendicular field Bperp.
        :param Bpara: parallel field
        :param Bperp: perpendicular field
        :return: critical current density
        """
        return self.jc / (1 + np.sqrt((self.k * Bpara)**2 + Bperp**2) / self.B0)**self.alpha


class Coil:
    """
    class for the YBCO coil/stack
    """
    def __init__(self, rin=0.1, rout=0.2, zmin=0.1, count_turns=10):
        """
        define the geometry of the coil/stack
        :param rin:  inner diameter
        :param rout:  outter diameter
        :param zmin:  the bottom position of the coil
        :param count_turns:  total turn of the coil
        """
        self.rin = rin
        self.rout = rout
        self.zmin = zmin
        self.count_turns = count_turns


class AClossFT2D:
    """
    class to solve the flux penetration process
    """
    def __init__(self, tape_type=YBCOtape(4e-3, 20, 120), csys=0, sym=(0, 0)):
        """
        define the ReBCO tape used, the coordinate system, and symmetry condition.
        :param tape_type: (YBCOtape) YBCO tape defined
        :param csys: 0: Cartesian coordinate system (stack)  1: Cylindrical coordinates (coil)
        :param sym: (x-z, y-z) 1: 'symmetry', -1: 'anti- symmetry', 0: 'no-symmetry'
        """
        self.tape_type = tape_type
        self.csys = csys
        self.sym = sym
        self.coils = []
        self.save_time = []
        self.save_time_step = []
        self.j_limit = 1.1
        self.timestep_type = 0
        self.timestep_min = 1e-5
        self.timestep_max = 1e-4
        self.time_point_no = 0
        self.dta = []
        self.time_array = []
        self.power = []
        self.ny = 0
        self.dy = 0
        self.dx = 0
        self.count_turns = 0
        self.posi_x = np.empty(0)
        self.posi_y = np.empty(0)
        self.count_ele = 0
        self.Ba = 0
        self.Ba_fre = 0
        self.Ba_phi = 0
        self.Ia = 0
        self.Ia_fre = 0
        self.Ia_phi = 0
        self.jt = []
        self.jc = []
        self.L = None
        self.Qij_inv = None
        self.M = None
        self.LM_inv = None
        self.Mbx = None
        self.Mby = None
        self.step = 0

    def add_coil(self, rin=0.1, rout=0.2, zmin=0.0, count_turns=10):
        """
        adding a coil in the system
        :param rin: innter diameter
        :param rout:  outter diameter
        :param zmin:   the bottom position of the coil
        :param count_turns:  the turn number of the coil
        :return: None
        """
        self.coils.append(Coil(rin, rout, zmin, count_turns))

    def mesh(self, ny):
        """
        mesh, diving each tape into ny element evenly
        :param ny: number of elements in each tape
        :return: None
        """
        print(time.asctime(time.localtime(time.time())) + ': Start meshing.')
        self.ny = ny
        self.dy = self.tape_type.a / ny
        self.dx = self.tape_type.b
        self.count_turns = sum([c.count_turns for c in self.coils])
        self.posi_x = np.empty(0)
        self.posi_y = np.empty(0)
        for coil in self.coils:
            yk_new = (np.arange(0, ny) + 0.5) * self.tape_type.a / ny + coil.zmin
            if coil.count_turns == 1:
                dis_turn_x = 0
            else:
                dis_turn_x = (coil.rout - coil.rin) / (coil.count_turns - 1)
            for turn_num in range(coil.count_turns):
                posi_x_turn = coil.rin + turn_num * dis_turn_x
                self.posi_x = np.hstack((self.posi_x, np.ones(ny) * posi_x_turn))
                self.posi_y = np.hstack((self.posi_y, yk_new.copy()))
        self.count_ele = len(self.posi_x)
        print(time.asctime(time.localtime(time.time())) + ': Finish meshing.')

    def set_time(self, save_time, timestep_min, timestep_max, timestep_type, j_limit=1.1):
        """
        define the solver
        :param save_time: times to save results
        :param timestep_min: minimum steptime
        :param timestep_max:  maximum step time
        :param timestep_type: 1: auto steptime between timestep_min and timestep_max 0: dt=timestep_min
        :param j_limit: Time step criterion used when timestep_type=1
        :return: None
        """
        self.save_time = save_time
        self.timestep_min = timestep_min
        self.timestep_max = timestep_max
        self.timestep_type = timestep_type
        self.j_limit = j_limit

    def set_field(self, Ba, fre, phi=0):
        """
        define the appled field
        :param Ba:  field amplitude
        :param fre: field frequency
        :param phi: phase angle
        :return: None
        """
        self.Ba = Ba
        self.Ba_fre = fre
        self.Ba_phi = phi

    def set_current(self, Ia, fre):
        """
        define the applied current
        :param Ia: current amplitude
        :param fre: current frequency
        :return: None
        """
        self.Ia = Ia
        self.Ia_fre = fre
        self.Ia_phi = 0

    def _cal_b(self, time1):
        """
        calculate the applied field in different time
        :param time1: array of times
        :return: array of magnetic fields
        """
        return np.sin(2 * np.pi * self.Ba_fre * time1 + self.Ba_phi) * self.Ba

    def _cal_db(self, time1):
        """
        calculate the applied field gradient dB/dt in different time
        :param time1: array of times
        :return: array of magnetic fields gradient
        """
        return 2 * np.pi * self.Ba_fre * np.cos(2 * np.pi * self.Ba_fre * time1 + self.Ba_phi) * self.Ba

    def _cal_i(self, time1):
        """
        calculate the applied current in different time
        :param time1: array of times
        :return:  array of current
        """
        return np.sin(2 * np.pi * self.Ia_fre * time1 + self.Ia_phi) * self.Ia

    def _cal_di(self, time1):
        """
        calculate the applied current gradient dI/dt in different time
        :param time1: array of times
        :return:  array of current gradient
        """
        return 2 * np.pi * self.Ia_fre * np.cos(2 * np.pi * self.Ia_fre * time1 + self.Ia_phi) * self.Ia

    def run(self, build_matrix=True, cal_jcb=True):
        """
        solve the problem
        :param build_matrix: True: rebuild the matrixes (usually set as True)
        :param cal_jcb: True: Considering Jc(B) characteristics of the tape
        :return: None
        """
        print(time.asctime(time.localtime(time.time())) + ': Start solving.')
        self.jt = [None] * len(self.save_time)  # Array to current density distribution
        self.jc = [None] * len(self.save_time)  # Array to critical current density distribution
        dx = self.dx  # element size
        dy = self.dy  # element size

        Ec = 1e-4
        # Reshape the position arrays to N*1
        posi_x = np.reshape(self.posi_x, (self.count_ele, 1))
        posi_y = np.reshape(self.posi_y, (self.count_ele, 1))

        # Calculate the matrixes
        if isinstance(self.L, type(None)) or build_matrix:
            print(time.asctime(time.localtime(time.time())) + ': Start buliding matrices. ')
            if self.csys == 0:
                # Cartesian coordinate system
                print(time.asctime(time.localtime(time.time())) + ': Cartesian coordinate system.')
                self._build_matrix_car()
            elif self.csys == 1:
                # Cylindrical coordinate system
                print(time.asctime(time.localtime(time.time())) + ': Cylindrical coordinates system.')
                self._build_matrix_cyl()
            else:
                print(time.asctime(time.localtime(time.time())) + ': Wrong coordinates system.')
                return
        else:
            print(time.asctime(time.localtime(time.time())) + ': Using last matrices.')

        time_run = 0    # model time
        self.step = 0   # model step
        self.save_time_step.append(self.step)  # Array to record steps of saved results
        self.time_point_no = 1  # number of saved results
        self.dta = []  # Array to record each step time used in solving
        self.time_array = []  # Array to record the time for each step
        self.power = []  # Array to record the AC loss in each step
        lastjt = np.zeros((self.count_ele, 1))  # current density in last step
        self.jc[0] = np.ones((self.count_ele, 1)) * self.tape_type.jc  # Critical current density at step [0]
        self.jt[0] = lastjt.copy()  # current density at step [0]
        reach_save = False  # switch to record
        while self.time_point_no < len(self.save_time):
            # Main solving loop
            self.time_array.append(time_run)  # record the model time
            if cal_jcb:
                # Calculate the magnetic filed at each element and corresponding jc
                bx = np.matmul(self.Mbx, lastjt) + self._cal_b(time_run) * self.posi_y.reshape((self.count_ele, 1))
                by = np.matmul(self.Mby, lastjt)
                jc = self.tape_type.jcB(by, bx)
            else:
                jc = self.tape_type.jc
            Ee = np.abs(Ec * (lastjt / jc)**self.tape_type.nvalue) * np.sign(lastjt)
            if np.isnan(Ee[0]):
                # Record the Jt when the calculation does not converge
                self.breakj = lastjt
                break
            # power_temp = ((Ee * lastjt) * (abs(lastjt) > jc)+(Ec * lastjt**2 / jc)*(abs(lastjt) <= jc)).reshape((self.count_turns, self.ny)).sum(axis=1)
            power_temp = (Ee * lastjt).reshape((self.count_turns, self.ny)).sum(axis=1)  # calculte the AC loss
            self.power.append(power_temp)   # Record the Ac loss
            # Calculate the first part of dj
            jtdb = np.matmul(self.Qij_inv, Ee + posi_y * self._cal_db(time_run))
            # Calculate the Ea
            Ea = np.matmul(self.LM_inv, self._cal_di(time_run) / dx / dy * np.ones((self.count_turns, 1))
                             - np.matmul(self.L, jtdb))
            # Calculate the second part of dj
            jtdc = np.matmul(self.M, Ea)
            dj = jtdb + jtdc
            # Calculate dt
            if self.timestep_type == 1:
                # auto dt
                s1 = dj > 0
                s2 = dj < 0
                dts = self.timestep_max
                if s1.max():
                    dts = min(self.timestep_max, abs((self.j_limit * jc - lastjt)[s1] / dj[s1]).min())
                if s2.max():
                    dts = min(dts, abs((-self.j_limit * jc - lastjt)[s2] / dj[s2]).min())
                dts = max(dts, self.timestep_min)
            else:
                # fixed dt
                dts = self.timestep_min
            if dts >= self.save_time[self.time_point_no] - time_run:
                # Reach the save point
                reach_save = True
                dt = self.save_time[self.time_point_no] - time_run
            else:
                dt = dts
            self.dta.append(dt)  # Record dt
            time_run += dt  # update time
            lastjt += dj * dt  # update current
            self.step += 1  # update step
            if reach_save:
                # Record jt jc
                self.jt[self.time_point_no] = lastjt.copy()
                self.jc[self.time_point_no] = jc.copy()
                print(time.asctime(time.localtime(time.time())) + ': Result at time = %5.4f is recorded.' % time_run)
                self.time_point_no += 1
                self.save_time_step.append(self.step)
                reach_save = False

        print(time.asctime(time.localtime(time.time())) + ': Finish solving.')

    def _build_matrix_car(self):
        """
        Build the matrixes used in Cartesian coordinate system
        :return: None
        """
        miu0 = 4 * np.pi * 1e-7
        dx = self.dx
        dy = self.dy
        r1_x, r2_x = np.meshgrid(self.posi_x, self.posi_x)
        r1_y, r2_y = np.meshgrid(self.posi_y, self.posi_y)
        eps = 0.005 * dx * dy
        # Self.L is a matrix containing the relation between the element number and its turn number
        self.L = np.zeros((self.count_turns, self.count_ele))
        for i in range(self.count_turns):
            self.L[i, i * self.ny: (i + 1) * self.ny] = np.ones(self.ny)
        Qij = -0.5 * np.log((r1_x - r2_x) ** 2 + (r1_y - r2_y) ** 2 + eps) / 2 / np.pi
        self.Mbx = miu0 * dx * dy * (r2_y - r1_y) / ((r1_x - r2_x)**2 + (r1_y - r2_y)**2 + eps) / 2 / np.pi
        self.Mby = miu0 * dx * dy * (r1_x - r2_x) / ((r1_x - r2_x)**2 + (r1_y - r2_y)**2 + eps) / 2 / np.pi
        dic_wym = {1: 'symmetry', -1: 'anti- symmetry'}
        if abs(self.sym[0]) == 1:
            print(time.asctime(time.localtime(time.time())) + ': Using x-z plane ' + dic_wym[self.sym[0]] + '.')
            Qij += -self.sym[0] * 0.5 * np.log((r1_x - r2_x) ** 2 + (r1_y + r2_y) ** 2 + eps) / 2 / np.pi
            self.Mbx += self.sym[0] * miu0 * dx * dy * (r2_y + r1_y) \
                        / ((r1_x - r2_x) ** 2 + (r1_y + r2_y) ** 2 + eps) / 2 / np.pi
            self.Mby += self.sym[0] * miu0 * dx * dy * (r1_x - r2_x) \
                        / ((r1_x - r2_x) ** 2 + (r1_y + r2_y) ** 2 + eps) / 2 / np.pi
        if abs(self.sym[1]) == 1:
            print(time.asctime(time.localtime(time.time())) + ': Using y-z plane ' + dic_wym[self.sym[1]] + '.')
            Qij += -self.sym[1] * 0.5 * np.log((r1_x + r2_x) ** 2 + (r1_y - r2_y) ** 2 + eps) / 2 / np.pi
            self.Mbx += self.sym[1] * miu0 * dx * dy * (r2_y - r1_y) \
                        / ((r1_x + r2_x) ** 2 + (r1_y - r2_y) ** 2 + eps) / 2 / np.pi
            self.Mby += self.sym[1] * miu0 * dx * dy * (-r1_x - r2_x) \
                        / ((r1_x + r2_x) ** 2 + (r1_y - r2_y) ** 2 + eps) / 2 / np.pi
        if abs(self.sym[0] * self.sym[1]) == 1:
            Qij += -self.sym[0] * self.sym[1] * 0.5 * np.log((r1_x + r2_x) ** 2 + (r1_y + r2_y) ** 2 + eps) \
                   / 2 / np.pi
            self.Mbx += self.sym[0] * self.sym[1] * miu0 * dx * dy * (r2_y + r1_y) \
                        / ((r1_x + r2_x)**2 + (r1_y + r2_y)**2 + eps) / 2 / np.pi
            self.Mby += self.sym[0] * self.sym[1] * miu0 * dx * dy * (-r1_x - r2_x) \
                        / ((r1_x + r2_x)**2 + (r1_y + r2_y)**2 + eps) / 2 / np.pi
        del r1_x, r2_x, r1_y, r2_y
        self.Qij_inv = -1 / miu0 / dx / dy * np.linalg.inv(Qij)  # inverse kernel
        del Qij
        self.M = np.matmul(self.Qij_inv, self.L.transpose())  # To calculate the second part of dj
        self.LM_inv = np.linalg.inv(np.matmul(self.L, self.M))  # solution of the linear equations
        print(time.asctime(time.localtime(time.time())) + ': Finish buliding matrices.')

    def _build_matrix_cyl(self):
        """
         Build the matrixes used in Cylindrical coordinate system
        :return: None
        """
        miu0 = 4 * np.pi * 1e-7
        dx = self.dx
        dy = self.dy
        r1_x, r2_x = np.meshgrid(self.posi_x, self.posi_x)
        r1_y, r2_y = np.meshgrid(self.posi_y, self.posi_y)
        eps = 0.005 * dx * dy

        def calAphi(r_s, r_t, z_s, z_t):
            k2 = 4 * r_s * r_t / ((r_s + r_t) ** 2 + (z_s - z_t) ** 2 + eps)
            Aphi = 1 / np.pi / np.sqrt(k2) * np.sqrt(r_s / r_t) * (sps.ellipk(k2) * (1 - k2 / 2) - sps.ellipe(k2))
            del k2
            return Aphi

        self.L = np.zeros((self.count_turns, self.count_ele))
        for i in range(self.count_turns):
            self.L[i, i * self.ny: (i + 1) * self.ny] = np.ones(self.ny)
        Qij = -calAphi(r1_x, r2_x, r1_y, r2_y)
        dis = min(dx, dy) / 5
        self.Mbx = -miu0 * dx * dy * (calAphi(r1_x, r2_x, r1_y, r2_y + dis) - calAphi(r1_x, r2_x, r1_y, r2_y - dis)) / dis / 2
        self.Mby = miu0 * dx * dy * ((r2_x + dis) * calAphi(r1_x, r2_x + dis, r1_y, r2_y) -
                                      (r2_x - dis) * calAphi(r1_x, r2_x - dis, r1_y, r2_y)) / r2_x / dis / 2
        dic_wym = {1: 'symmetry', -1: 'anti- symmetry'}
        if abs(self.sym[0]) == 1:
            print(time.asctime(time.localtime(time.time())) + ': Using r plane ' + dic_wym[self.sym[0]] + '.')
            Qij += -self.sym[0] * calAphi(r1_x, r2_x, -r1_y, r2_y)
            self.Mbx += - self.sym[0] * miu0 * dx * dy * (calAphi(r1_x, r2_x, -r1_y, r2_y + dis) -
                                                          calAphi(r1_x, r2_x, -r1_y, r2_y - dis)) / dis / 2
            self.Mby += self.sym[0] * miu0 * dx * dy * ((r2_x + dis) * calAphi(r1_x, r2_x + dis, -r1_y, r2_y) -
                                      (r2_x - dis) * calAphi(r1_x, r2_x - dis, -r1_y, r2_y)) / r2_x / dis / 2
        del r1_x, r2_x, r1_y, r2_y
        self.Qij_inv = 1 / miu0 / dx / dy * np.linalg.inv(Qij)
        del Qij
        self.M = np.matmul(self.Qij_inv, self.L.transpose())
        self.LM_inv = np.linalg.inv(np.matmul(self.L, self.M))
        print(time.asctime(time.localtime(time.time())) + ': Finish buliding matrices.')

    def plot_resultj(self, timestep_n0, use_sym=False, plot3D=False):
        """
        plot the current density distribution at timestep_no
        :param timestep_n0: time step to show the result, which should be defind in function "set_time"
        :param use_sym: True: plot the symmetrical parts False: only plot calculated part
        :param plot3D: True: show 3D effect  False: 2D plot
        :return: None
        """
        if timestep_n0 >= len(self.save_time):
            print(time.asctime(time.localtime(time.time())) + ': Result at timestep %i is not saved.' % timestep_n0)
            return
        j = (self.jt[timestep_n0] / self.jc[timestep_n0]).reshape((self.count_turns, self.ny))
        x = self.posi_x.reshape((self.count_turns, self.ny))
        y = self.posi_y.reshape((self.count_turns, self.ny))
        f = plt.figure()
        if plot3D:
            ax = f.gca(projection='3d')
            ax.set_zlim([-1.1, 1.1])
            plotme = ax.plot_surface
        else:
            ax = f.add_subplot(111)
            plotme = ax.contourf
        k0 = 0
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title('J/Jc')
        for i in range(len(self.coils)):
            k1 = k0 + self.coils[i].count_turns
            surf = plotme(x[k0: k1, :], y[k0: k1, :], j[k0: k1, :], cmap=cm.coolwarm, levels=np.linspace(-1.2, 1.2, 10))
            k0 = k1
        cb = f.colorbar(surf, ticks=[-1, -0.5, 0, 0.5, 1])
        # cb.ax.set_ylabel('J/Jc')
        # cb.ax.annotate('J/Jc', (0.1,1.05))
        if use_sym:
            if abs(self.sym[0]) == 1:
                k0 = 0
                for i in range(len(self.coils)):
                    k1 = k0 + self.coils[i].count_turns
                    surf = plotme(x[k0: k1, :], -y[k0: k1, :], self.sym[0] * j[k0: k1, :], cmap=cm.coolwarm, vmin=-1
                                  , vmax=1)
                    k0 = k1
            if abs(self.sym[1]) == 1:
                k0 = 0
                for i in range(len(self.coils)):
                    k1 = k0 + self.coils[i].count_turns
                    surf = plotme(-x[k0: k1, :], y[k0: k1, :], self.sym[1] * j[k0: k1, :], cmap=cm.coolwarm, vmin=-1
                                  , vmax=1)
                    k0 = k1
            if abs(self.sym[0] * self.sym[1]) == 1:
                k0 = 0
                for i in range(len(self.coils)):
                    k1 = k0 + self.coils[i].count_turns
                    surf = plotme(-x[k0: k1, :], -y[k0: k1, :],  self.sym[0] * self.sym[1] * j[k0: k1, :],
                                  cmap=cm.coolwarm, vmin=-1, vmax=1)
                    k0 = k1

        f.savefig(f'{timestep_n0}CurrentDensityPlot.png')
            
    def plot_resultb(self, timestep_n0,  use_sym=False):
        """
        plot the normal magnetic field distribution at timestep_no
        :param timestep_n0: time step to show the result, which should be defind in function "set_time"
        :param use_sym: True: plot the symmetrical parts False: only plot calculated part
        :param plot3D: True: show 3D effect  False: 2D plot
        :return: None
        """
        if timestep_n0 >= len(self.save_time):
            print(time.asctime(time.localtime(time.time())) + ': Result at timestep %i is not saved.' % timestep_n0)
            return
        bx = np.matmul(self.Mbx, self.jt[timestep_n0])
        by = np.matmul(self.Mby, self.jt[timestep_n0])
        b = (np.sqrt(bx**2 + by**2)).reshape((self.count_turns, self.ny))
        x = self.posi_x.reshape((self.count_turns, self.ny))
        y = self.posi_y.reshape((self.count_turns, self.ny))
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title('B (T)')
        k0 = 0
        for i in range(len(self.coils)):
            k1 = k0 + self.coils[i].count_turns
            surf = ax.contourf(x[k0: k1, :], y[k0: k1, :], b[k0: k1, :], cmap=cm.coolwarm)
            k0 = k1
        cb = f.colorbar(surf)
        if use_sym:
            if abs(self.sym[0]) == 1:
                k0 = 0
                for i in range(len(self.coils)):
                    k1 = k0 + self.coils[i].count_turns
                    surf = ax.contourf(x[k0: k1, :], -y[k0: k1, :], b[k0: k1, :], cmap=cm.coolwarm)
                    k0 = k1
            if abs(self.sym[1]) == 1:
                k0 = 0
                for i in range(len(self.coils)):
                    k1 = k0 + self.coils[i].count_turns
                    surf = ax.contourf(-x[k0: k1, :], y[k0: k1, :], b[k0: k1, :], cmap=cm.coolwarm)
                    k0 = k1
            if abs(self.sym[0] * self.sym[1]) == 1:
                k0 = 0
                for i in range(len(self.coils)):
                    k1 = k0 + self.coils[i].count_turns
                    surf = ax.contourf(-x[k0: k1, :], -y[k0: k1, :], b[k0: k1, :], cmap=cm.coolwarm)
                    k0 = k1
        plt.savefig(f'{timestep_n0}BFieldPlot.png')
        plt.show()


    def post_cal_power(self, timestep0, timestep1, amp=1):
        """
        Calculate the AC loss between timestep0 and timestep1
        :param timestep0: step time to start
        :param timestep1:  step time to end
        :param amp: can set a amp paramter to compare different frequency
        :return: Average AC loss
        """
        time1 = self.save_time_step[timestep0]
        time2 = self.save_time_step[timestep1]
        power_avg = np.zeros(self.count_turns)
        for i in range(self.count_turns):
            power = np.array([j[i] for j in self.power][time1: time2]) * self.dx * self.dy
            dt = np.array(self.dta[time1:time2])
            power_avg[i] = (power * dt).sum() / dt.sum()
        f = plt.figure()
        axe = f.add_subplot(111)
        axe.semilogy(np.arange(1, self.count_turns + 1), power_avg*amp)
        axe.set_xlabel('Turn number')
        axe.set_ylabel('Average loss per cycle (W/m)')
        axe.grid(which='both')

        f = plt.figure()
        axe = f.add_subplot(111)
        startnum = 0
        for i in range(len(self.coils)):
            endnum = startnum + self.coils[i].count_turns
            axe.semilogy(np.arange(1, self.coils[i].count_turns + 1), power_avg[startnum: endnum] * amp)
            startnum = endnum
        axe.set_xlabel('Turn number')
        axe.set_ylabel('Average loss (W/m)')
        axe.grid(which='both')

        plt.show()
        print(time.asctime(time.localtime(time.time())) + ': Total average loss: %5.4f W/m.' % (power_avg.sum()*amp))
        return power_avg

    def save_result(self, filename='./ACloss_YBCO2D.pkl'):
        """
        save the calculated results
        :param filename: filename
        :return: None
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_result(filename):
        """
        load saved result
        :param filename: filename
        :return: (AClossFT2D) loaded results
        """
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data

    # def animate_resultj(self, filename, interval=100):
    #     x = self.model.posi_x.reshape((self.model.ny, self.model.nx))
    #     y = self.model.posi_y.reshape((self.model.ny, self.model.nx))
    #     times = sorted(self.jt.keys())
    #     fig = plt.figure()
    #     ax = fig.gca(projection='3d')
    #     def update(i):
    #         time = times[i]
    #         j = self.jt[time].reshape((self.model.ny, self.model.nx))
    #         ax.clear()
    #         ax.set_zlim([-1.5e8, 1.5e8])
    #         ax.plot_surface(x, y, j, cmap=cm.coolwarm)
    #     anim = FuncAnimation(fig, update, frames=np.arange(0, 66), interval=interval)
    #     anim.save(filename, dpi=80, writer='imagemagick')


if __name__ == '__main__':
    tape = YBCOtape()
    solver = AClossFT2D(tape, csys=1, sym=(1, 0))
    solver.add_coil(150e-6, 150e-6 + 300e-6*2, 0, 1)
    # solver.add_coil(150e-6, 150e-6 + 300e-6*10, 0.0044, 10)
    solver.mesh(15)
    save_time = np.linspace(0, 1, 13, endpoint=True)
    timestep_min = 1e-6
    timestep_max = 1e-4
    solver.set_time(save_time, timestep_min, timestep_max, 1)
    Imax = 20
    fre = 1
    solver.set_current(Imax, fre, 0)
    # Bmax = 40e-3
    # fre = 2
    # solver.set_field(Bmax, fre, 0)
    solver.run(cal_jcb=False)
    # plt.plot(solver.posi_y, solver.jt[12500].reshape(-1))
    use_sym = False
    solver.plot_resultj(9, use_sym=use_sym)
    solver.post_cal_power(6, 12)
    # solver.plot_resultb(2500, use_sym=use_sym)
    # solver.plot_resultj(5000, use_sym=use_sym)
    # solver.plot_resultj(7500, use_sym=use_sym)
    # solver.plot_resultj(10000, use_sym=use_sym)
    # solver.plot_resultj(15000)
    # solver.plot_resultj(20000)
    # solver.plot_resultj(25000)
    # solver.plot_resultj(30000)
    # solver.plot_resultj(35000)
    # solver.plot_resultj(40000)
    # solver.plot_resultj(45000)
    # solver.plot_resultj(50000)
    # solver.plot_resultj(55000)









