import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

#########################################################################################################################################
#########################################################################################################################################
###################################### Simple CLASS for SDOF system in free vibration ###################################################
#########################################################################################################################################
#########################################################################################################################################

class Simple_free_motion():
    def __init__(self, xo, xvo, w, to, dt, tf):
        self.xo = xo
        self.xvo = xvo
        self.w = w
        self.to = to
        self.dt = dt
        self.tf = tf
    
    def sim_free_sdof_nodamp(self):
        xo = self.xo
        xvo = self.xvo
        w = self.w
        to = self.to
        dt = self.dt
        tf = self.tf
        x = np.zeros((int((tf-to)/dt), 1))
        ti = np.zeros((len(x), 1))

        j = 0
        for t in np.arange(to,tf, dt):
            x[j] = xo * np.cos(w*t) + xvo / w * np.sin(w*t)
            ti[j] = t
            j = j+1

        x_df = pd.DataFrame(x, columns= ['Amplitude'])
        ti_df = pd.DataFrame(ti, columns= ['Time'])
        resul = pd.concat([ti_df, x_df], axis=1, ignore_index=False)
        
        return x, ti, resul

#########################################################################################################################################
#########################################################################################################################################
############################################# Simple CLASS for Plot Amplitude vs Time ###################################################
#########################################################################################################################################
#########################################################################################################################################

class plt_amp_tim():
    def __init__(self, x, ti, color = [0, 0, 0], title = 'Amplitude vs Time'):
        self.x = x
        self.ti = ti
        self.color = color
        self.title = title

    def plot_Avst(self):
        x = self.x
        ti = self.ti
        color = self.color
        title = self.title
        #------------------------- PLOTS -----------------------------#
        fig, ax = plt.subplots(1,1, figsize = (20,5))
        ax.plot(ti, x, color = color, lw = 1, ls = '-', marker = 'o', markersize = 2, label = 'Response Amplitude')
        ax.set_title(title, fontsize = 12, color = [0,0,1], fontweight = 'bold')
        ax.set_ylabel('Response Amplitude')
        ax.set_xlabel('Time [s]')
        ax.grid(visible= True, axis= 'x')
        ax.set_xlim(ti[0], ti[-1])
        plt.show()
        
