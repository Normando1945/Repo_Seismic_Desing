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

        return x, ti

#########################################################################################################################################
#########################################################################################################################################
############################################# Simple CLASS for Plot Amplitude vs Time ###################################################
#########################################################################################################################################
#########################################################################################################################################

class plt_amp_tim():
    def __init__(self, x = any, ti = any, color = [0, 0, 0], title = 'Amplitude vs Time',
                 ylabel = 'Amplitude', grl = 1):
        self.x = x
        self.ti = ti
        self.color = color
        self.title = title
        self.ylabel = ylabel
        self.grl = grl

    def plot_Avst(self):
        x = self.x
        ti = self.ti
        color = self.color
        title = self.title
        ylabel = self.ylabel
        grl = self.grl
        #------------------------- PLOTS -----------------------------#
        fig, ax = plt.subplots(1,1, figsize = (20,5))
        ax.plot(ti, x, color = color, alpha = 0.5 ,lw = grl, ls = '-', marker = 'o', markersize = 0, label = ylabel)
        ax.set_title(title, fontsize = 12, color = [0,0,1], fontweight = 'bold')
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Time [s]')
        ax.grid(visible= True, axis= 'x')
        ax.set_xlim(ti[0], ti[-1])
        plt.show()
        
#########################################################################################################################################
#########################################################################################################################################
################################# Simple CLASS for SDOF system in free vibration (Sub Damping) ##########################################
#########################################################################################################################################
#########################################################################################################################################

class Simple_free_motion_sub_damping():
    def __init__(self,T = 0.1, xo = 1.0, xvo = 0.0, zi = 0.05, to = 0, tf = 5.0, dt = 0.001):
        self.T = T
        self.xo = xo
        self.xvo = xvo
        self.zi = zi
        self.to = to
        self.tf = tf
        self.dt = dt
    
    def sim_free_sdof_SubDamping(self):
        T = self.T
        xo = self.xo
        xvo = self.xvo
        zi = self.zi
        to = self.to
        tf = self.tf
        dt = self.dt
        
        w = (2 * np.pi)/T
        xsub = np.zeros(int(tf/dt + 1))
        ti = np.zeros(int(tf/dt + 1))
        wsub = w * np.sqrt(1 - zi**2)
        j = 0
        for t in np.arange(to,tf + dt,dt):
            xsub[j] = np.exp(-zi*w*t)*(xo*np.cos(wsub*t) + (xvo + zi*w*xo)/(wsub)*np.sin(wsub*t))
            ti[j] = t
            j = j+1
        
        return xsub, ti
        
#########################################################################################################################################
#########################################################################################################################################
############################################ Response step by step (B-Newmark [Jr]) #####################################################
#########################################################################################################################################
#########################################################################################################################################

class Step_by_Step_BNewmark():
    
    def __init__(self, T= 1.0, M = 1.0, zi = 0.05, accel_record = any, vector_time = any,
                 colorSeism = (0.5,0.5,0.5), colorRaccel = (0,0,1),colorRTaccel = (1,0,0), title = 'Record'):
        self.T = T
        self.M = M
        self.zi = zi
        self.SG = accel_record 
        self.TI = vector_time
        self.colorSeism = colorSeism
        self.colorRaccel = colorRaccel
        self.colorRTaccel = colorRTaccel
        self.title = title
        
    
    def Bnewmark_Jr(self):
        T = self.T
        M = self.M
        zi = self.zi
        SG = self.SG
        TI = self.TI
        
        dt = TI[1] - TI[0]
        
        w = (2*np.pi)/T
        k = ((2*np.pi)/T)**(2)*M
        xo = 0
        xvo = 0

        xvn = xvo
        xn = xo
        xao = (-SG[0] * M - 2*zi*w*M*xvo -w**(2)*xo) * 1/M

        print(len(SG))

        xn1 = np.zeros(len(SG))
        xvn1 = np.zeros(len(SG))
        xan1 = np.zeros(len(SG))
        at = np.zeros(len(SG))

        xan = xao

        xan1[0] = xao
        xvn1[0] = xvo
        xn1[0] = xo  

        for i in np.arange(1, len(SG),1):
            xn1[i] = xn + dt*xvn + (dt**2)/2*xan
            xan1[i] = 1 / (M + (1/2)*(2*zi*w*M*dt)) * (-SG[i]*M - k*xn1[i] - 2*zi*w*M*(xvn + dt*(1 - 1/2)*xan))
            xvn1[i] = xvn + dt*((1-1/2)*xan + (1/2)*xan1[i])
            at[i] = SG[i] + xan1[i]
            
            xan = xan1[i]
            xvn = xvn1[i]
            xn = xn1[i]
        
        return at, xan1, xvn1, xn1  
    
    def plot_seis_Raccel(self, xan1, at):
        TI = self.TI
        SG = self.SG
        T = self.T
        zi = self.zi
        colorSeism = self.colorSeism
        colorRaccel = self.colorRaccel
        colorRTaccel = self.colorRTaccel
        title = self.title
        
        #----------Max Sg------------#
        maxSG = np.max(np.abs(SG))
        TI_maxSG = TI[np.argmax(np.abs(SG))]
        #----------Max AT------------#
        maxAT = np.max(np.abs(at))
        TI_maxAT = TI[np.argmax(np.abs(at))]
        #----------Eta------------#
        n = maxAT / maxSG
        #----------Plot------------#
        fig, ax = plt.subplots(2,1, figsize = (20,10))
        fig.suptitle(f"B-Newmark, Solver", fontsize=18, color = (0,0,1), y=0.98)
        
        ax[0].plot(TI, SG, color = colorSeism, alpha = 1.0 ,lw = 1.0, ls = '-', marker = 'o', 
                markersize = 0, label = 'Seismic Record')
        ax[0].plot(TI_maxSG, SG[np.argmax(np.abs(SG))], color = colorSeism, alpha = 1.0 ,lw = 1.0, ls = '-', marker = 'o', 
                markersize = 5, label = f'PGA = {maxSG:.4f} [g], t = {TI_maxSG:.4f} [s]')
        ax[0].set_title('REC =' + ' '+ title, fontweight = 'bold')
        ax[0].set_ylabel('Acceleration [g]')
        ax[0].set_xlabel('Time [s]')
        ax[0].grid(visible= True, axis= 'x')
        ax[0].set_xlim(TI[0], TI[-1])
        ax[0].legend(loc='best')
        
        ax[1].plot(TI, SG, color = colorSeism, alpha = 1.0 ,lw = 1.0, ls = '-', marker = 'o', 
                markersize = 0, label = 'Seismic Record')
        ax[1].plot(TI, xan1, color = colorRaccel, alpha = 1.0 ,lw = 1.0, ls = '-', marker = 'o', 
                markersize = 0, label = 'Acceleration Response')
        ax[1].plot(TI, at, color = colorRTaccel, alpha = 1.0 ,lw = 1.0, ls = '--', marker = 'o', 
                markersize = 0, label = 'Total Acceleration Response')
        ax[1].plot(TI_maxAT, at[np.argmax(np.abs(at))], color = colorRTaccel, alpha = 1.0 ,lw = 1.0, ls = '-', marker = 'o', 
                markersize = 5, label = f'maxAT = {maxAT:.4f} [g], t = {TI_maxAT:.4f} [s] / n = {n:.2f}')
        ax[1].set_title(f'Acceleration Response, T = {T:.2f} [s], zi = {zi * 100:.2f} [%]', fontweight = 'bold')
        ax[1].set_ylabel('Acceleration [g]')
        ax[1].set_xlabel('Time [s]')
        ax[1].grid(visible= True, axis= 'x')
        ax[1].set_xlim(TI[0], TI[-1])
        ax[1].legend(loc='best')
        
        plt.tight_layout()
        plt.show()        


#########################################################################################################################################
#########################################################################################################################################
########################################## Response Spectrum using (B-Newmark [Jr]) #####################################################
#########################################################################################################################################
#########################################################################################################################################

class SPEC_BNewmark():
    
    def __init__(self, To = 0.10, dT = 0.01, Tf = 4.0, M = 1.0, zi = 0.05, accel_record = any, vector_time = any,
                 colorSeism = (0.5,0.5,0.5), colorSpec = (0,0,1), title = 'Record'):
        self.To = To
        self.dT = dT
        self.Tf = Tf
        self.M = M
        self.zi = zi
        self.SG = accel_record 
        self.TI = vector_time
        self.colorSeism = colorSeism
        self.colorSpec = colorSpec
        self.title = title
        
    
    def Spec_Bnewmark_Jr(self):
        To = self.To
        dT = self.dT
        Tf = self.Tf
        M = self.M
        zi = self.zi
        SG = self.SG
        TI = self.TI
        
        dt = TI[1] - TI[0]
        PGA = np.max(np.abs(SG))
        
        Sa = np.zeros(int((Tf - To) / dT))
        Ti = np.zeros(int((Tf - To) / dT))
        n = np.zeros(int((Tf - To) / dT))
        d = 0
        
        for T in np.arange(To, Tf, dT):        
            w = (2*np.pi)/T
            k = ((2*np.pi)/T)**(2)*M
            xo = 0
            xvo = 0

            xvn = xvo
            xn = xo
            xao = (-SG[0] * M - 2*zi*w*M*xvo -w**(2)*xo) * 1/M

            xn1 = np.zeros(len(SG))
            xvn1 = np.zeros(len(SG))
            xan1 = np.zeros(len(SG))
            at = np.zeros(len(SG))

            xan = xao

            xan1[0] = xao
            xvn1[0] = xvo
            xn1[0] = xo  

            for i in np.arange(1, len(SG),1):
                xn1[i] = xn + dt*xvn + (dt**2)/2*xan
                xan1[i] = 1 / (M + (1/2)*(2*zi*w*M*dt)) * (-SG[i]*M - k*xn1[i] - 2*zi*w*M*(xvn + dt*(1 - 1/2)*xan))
                xvn1[i] = xvn + dt*((1-1/2)*xan + (1/2)*xan1[i])
                at[i] = SG[i] + xan1[i]
                
                xan = xan1[i]
                xvn = xvn1[i]
                xn = xn1[i]
            
            Sa[d] = np.max(np.abs(at))
            Ti[d] = T
            n[d] = Sa[d] / PGA
            d = d + 1
            
        return Sa, Ti, n
        
    def plot_SpecSa(self, Sa, Ti, n):
        TI = self.TI
        SG = self.SG
        zi = self.zi
        colorSeism = self.colorSeism
        colorSpec = self.colorSpec
        title = self.title
        
        #----------Max Sg------------#
        maxSG = np.max(np.abs(SG))
        TI_maxSG = TI[np.argmax(np.abs(SG))]
        
        #----------Max Sa------------#
        maxSa = np.max(np.abs(Sa))
        Ti_maxSa = Ti[np.argmax(np.abs(Sa))]
   
        #----------Plot------------#
        fig, ax = plt.subplots(2,1, figsize = (20,10))
        fig.suptitle(f"Spec B-Newmark, Solver", fontsize=18, color = (0,0,1), y=0.98)
        
        ax[0].plot(TI, SG, color = colorSeism, alpha = 1.0 ,lw = 1.0, ls = '-', marker = 'o', 
                markersize = 0, label = 'Seismic Record')
        ax[0].plot(TI_maxSG, SG[np.argmax(np.abs(SG))], color = colorSeism, alpha = 1.0 ,lw = 1.0, ls = '-', marker = 'o', 
                markersize = 5, label = f'PGA = {maxSG:.4f} [g], t = {TI_maxSG:.4f} [s]')
        ax[0].set_title('REC =' + ' '+ title, fontweight = 'bold')
        ax[0].set_ylabel('Acceleration [g]')
        ax[0].set_xlabel('Time [s]')
        ax[0].grid(visible= True, axis= 'x')
        ax[0].set_xlim(TI[0], TI[-1])
        ax[0].legend(loc='best')
        
        ax[1].plot(Ti, Sa, color = colorSpec, alpha = 1.0 ,lw = 1.0, ls = '-', marker = 'o', 
                markersize = 0, label = 'Acceleration Response Spectrum')
        ax[1].plot(Ti_maxSa, Sa[np.argmax(np.abs(Sa))], color = colorSpec, alpha = 1.0 ,lw = 1.0, ls = '-', marker = 'o', 
                markersize = 5, label = f'maxSa = {maxSa:.4f} [g], T = {Ti_maxSa:.4f} [s]')
        ax[1].set_title(f'Acceleration Response Spectrum, zi = {zi * 100:.2f} [%]', fontweight = 'bold')
        ax[1].set_ylabel('Acceleration [g]')
        ax[1].set_xlabel('Period [s]')
        ax[1].grid(visible= True, axis= 'x')
        ax[1].set_xlim(Ti[0], Ti[-1])
        ax[1].legend(loc='best')
        
        plt.tight_layout()
        plt.show()      
            

#########################################################################################################################################
#########################################################################################################################################
####################################################### UHS (NEC-2024) ##################################################################
#########################################################################################################################################
#########################################################################################################################################

class SPEC_NEC_2024():
    def __init__(self, z = 0.4, n = 2.4, fa = 1.2, fd = 1.0, fs = 1.0, dT = 0.001, Tf = 5.0, r = 1.0,
                 city = 'Ciudad', soild = 'soild', pga = '0.4', zone = 'II'):
        self.z = z
        self.n = n
        self.fa = fa
        self.fd = fd
        self.fs = fs
        self.dT = dT
        self.Tf = Tf
        self.r = r
        self.city = city
        self.soild = soild
        self.pga = pga
        self.zone = zone
        
    def spec(self):
        z = self.z
        n = self.n
        fa = self.fa
        fd = self.fd
        fs = self.fs
        dT = self.dT
        Tf = self.Tf 
        r = self.r       
        
        To = 0.1 * fs * fd / fa
        Tc = 0.45 * fs * fd / fa
        Tl = 2.4 * fd
        
        Sae = []
        Tie = []

        for T in np.arange(0, Tf, dT):
            if T <= To:
                Sae.append(z*fa*(1 + 1.4*(T/To)))
                Tie.append(T)
            else:
                if T <= Tc:
                    Sae.append(n*z*fa)
                    Tie.append(T)
                else:
                    if T <= Tl:
                        Sae.append(n*z*fa*(Tc/T)**(r))
                        Tie.append(T)
                    else:
                        Sae.append(n*z*fa*(Tc/T)**(r)*(Tl/T)**(2))
                        Tie.append(T)
        print("="*120)
        print(f'To = {To} [s], Tc = {Tc} [s], Tl = {Tl} [s], fa = {fa}, fd = {fd}, fs = {fs}')
        print("="*120)
        
        return Sae, Tie, To, Tc, Tl, fa, fd, fs
        
        
        
    def plotSPECNEC(self, Tie, Sae):
        city = self.city
        soild = self.soild
        pga = self.pga
        zone = self.zone
        #----------Plot------------#
        fig, ax = plt.subplots(1,1, figsize = (20,6))
        fig.suptitle(f"UHS NEC 2024, City = {city}, Soild = {soild}, PGA = {pga:.2f}, Zone = {zone}", fontsize=18, color = (0,0,1), y=0.98)
        
        ax.plot(Tie, Sae, color = (0,0,0), alpha = 1.0 ,lw = 1.0, ls = '-', marker = 'o', 
                markersize = 0, label = 'UHS NEC 2024')
        ax.set_title('UHS NEC 2024', fontweight = 'bold')
        ax.set_ylabel('Acceleration [g]')
        ax.set_xlabel('Period [s]')
        ax.grid(visible= True, axis= 'x')
        ax.set_xlim(Tie[0], Tie[-1])
        ax.legend(loc='best')
        
        plt.tight_layout()
        plt.show() 
        
        
            
        
          
    
