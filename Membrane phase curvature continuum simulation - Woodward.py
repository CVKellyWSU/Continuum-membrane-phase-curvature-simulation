# -*- coding: utf-8 -*-
"""
This simulation mimics the lipid phase separation over a static membrane shape as published in:

\title{Nanoscale membrane curvature sorts lipid phases and alters lipid diffusion}

\author[1]{Xinxin Woodward}
\author[2,3]{Matti Javanainen}
\author[2]{Balázs Fábián}
\author[1*]{Christopher V. Kelly}

\affil[1]{Department of Physics and Astronomy, Wayne State University, Detroit, MI, U.S.A. 48201}
\affil[2]{Institute of Organic Chemistry and Biochemistry, Czech Academy of Sciences, Prague, Czech Republic, 16000}
\affil[3]{Institute of Biotechnology, University of Helsinki, Helsinki, Finland, 00014}

\corr{cvkelly@wayne.edu}{CVK}

Created on Thu Apr  2 10:36:41 2020

@author: cvkelly
"""

import numpy as np
from os.path import isdir
from os import mkdir
from datetime import datetime
import matplotlib.pyplot as plt

# %% files to load to make this code work

fold = ''
file_to_load = 'membrane_shape.npz'
curvature_load = np.sqrt(np.loadtxt(fold+'Curvature vs Z.dat')[:,1])
import membrane_phase_fluctuation_functions as mpff

# %% parameters to customize
A = -1 # must be <0 for 2 stable phase. Equals Landau's A/2
B = 0.7 # how tight is line tension. Equals Landau's B/2   <----- MAJOR VARIABLE FOR DEGREE OF PHASE SEPARATION
C =  4 # 1*acfac # -2*acfac # must be > 0,
slow = 0.05


G_list = (0,0,0,0,0,0.1,0.1,0.2) # gamma = phase-curvature coupling
noise_list = np.array([0.065,0.07,0.08,0.1,0.065,0.07,0.065]) # scales with temperature

e_calc_freq = 1
plot_freq = 30000
max_time =  plot_freq*30

phase_norm_radius = 145 # not actually radius but x,y

# %% load files and prepare data

file = np.load(fold+file_to_load)
xyz = file['xyz']
closelist = file['closelist'].astype(np.int32)
file.close()
num_points = xyz.shape[0]

zarg = (xyz[:,2]*10+10).astype(np.int32)
curvature = curvature_load[zarg] # check this

plotting_resolution = 1
res= plotting_resolution

r = np.sqrt(xyz[:,0]**2+xyz[:,1]**2)
phase_norm_arg = (r>phase_norm_radius)


# %% run the simulation
simulation_cnt = 0
for repeat in range(1):  # how many times to repeat each T, gamma condition
    for con in range(len(G_list)): # which T, gamma condition
        g = G_list[con]
        noise = noise_list[con]

        plot_cnt = -1
        E_keep_cnt = -1
        simulation_cnt+=1

        if repeat == 0:
            # % initialize phase behavior
            phase = np.random.randn(xyz.shape[0])*0.2
            phase[xyz[:,0]<0] = -0.2
            phase[xyz[:,0]>0] = 0.2

            phase[phase_norm_arg] = phase[phase_norm_arg] - np.mean(phase[phase_norm_arg])
            tstart = 0

        tit_orig = 'A:'+str(A)[:5]+' B:'+str(B)[:5]+' C:'+str(C)[:5]+' G:'+str(g)[:5]+' S:'+str(slow)[:5] + ' N:'+str(noise)[:5] + ' R:'+str(repeat)
        fold_fig_save,madenew = mpff.fig_save_fold(fold+ 'figures' , tit_orig)
        if repeat > 0:
            fread = np.load(fold_fig_read+'data.npz')
            phase = fread['phase']
            tstart = fread['actual_max_time']
            fread.close()

        if madenew == -1:  # if this simulations has been previously performed, don't rerun it
            continue

        # % loop over time
        actual_max_time = np.max([int(max_time/60/noise**2+1),3*plot_freq])
        phase_all = np.zeros((int(actual_max_time/plot_freq+100),xyz.shape[0]))

        print('  Current Simulation Number:',simulation_cnt,' ', tit_orig,' Max Time:',actual_max_time, '  ',datetime.now().strftime("%H:%M:%S"))

        for t in range(actual_max_time): # loop over time
            phase = phase + np.random.randn(num_points)*noise - slow *(2*A*phase + 2*B*(phase-np.mean(phase[closelist],axis=1)) + 4*C*phase**3 + g*curvature)
            phase[phase_norm_arg] = phase[phase_norm_arg] - np.mean(phase[phase_norm_arg])
            if np.mod(t+1,plot_freq)==0 or t == actual_max_time-1:
                plot_cnt += 1
                phase_all[plot_cnt,:] = phase
                tit = tit_orig + ' T:'+str(t)
                plt.close()
                mpff.plot_colored_scatter3d_save(xyz, phase, res, tit , fold_fig_save, mpff.get_file_from_time(t))
                print('     '+ str(plot_cnt/(actual_max_time/plot_freq)*100)[:4] + '% completed')
                phase_all_now = phase_all[:plot_cnt,:]
                np.savez(fold_fig_save+'data.npz', phase=phase,
                    xyz=xyz, A=A, B=B, C=C, g=g, t=t,
                    repeat=repeat, noise=noise, slow=slow,
                    tit_orig=tit_orig, tit=tit, phase_norm_radius=phase_norm_radius,
                    plot_freq=plot_freq,
                    actual_max_time = (actual_max_time+tstart),
                    phase_all=phase_all_now)

