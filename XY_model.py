## IMPORT LIBRARY

import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab
import random
import matplotlib.image as mpimg
from matplotlib.legend_handler import HandlerLine2D
import math
from scipy.optimize import curve_fit
from numpy import pi
import os

## applying Metropolis algorithm
# input: T/temperature
#        S/spins configuration(in 1d list)
#        H/ecternal field.default value=0

class XYSystem():
    def __init__(self,temperature = 3,width=10):
        self.width = width
        self.num_spins = width**2
        L,N = self.width,self.num_spins
        self.nbr = {i : ((i // L) * L + (i + 1) % L, (i + L) % N,
                    (i // L) * L + (i - 1) % L, (i - L) % N) \
                                            for i in list(range(N))}
        self.spin_config = np.random.random(self.num_spins)*2*pi
        self.temperature = temperature
        self.energy = np.sum(self.get_energy())/self.num_spins
        self.M = []
        self.Cv = []


    def set_temperature(self,temperature):
        self.temperature = temperature;
    
    def sweep(self):
        beta = 1.0 / self.temperature
        spin_idx = list(range(self.num_spins))
        random.shuffle(spin_idx)
        for idx in spin_idx:#one sweep in defined as N attempts of flip
            #k = np.random.randint(0, N - 1)#randomly choose a spin
            energy_i = -sum(np.cos(self.spin_config[idx]-self.spin_config[n]) for n in self.nbr[idx]) 
            dtheta = np.random.uniform(-np.pi,np.pi)
            spin_temp = self.spin_config[idx] + dtheta
            energy_f = -sum(np.cos(spin_temp-self.spin_config[n]) for n in self.nbr[idx]) 
            delta_E = energy_f - energy_i
            if np.random.uniform(0.0, 1.0) < np.exp(-beta * delta_E):
                self.spin_config[idx] += dtheta



    ## calculate the energy of a given configuration  
    #  input: S/spin configuration in list
    #         H/external field, defult 0
    def get_energy(self):
        """Oblicza energię układu i normalizuje na spin."""
        energy = 0
        for i in range(self.num_spins):
            for j in self.nbr[i]:  # Sąsiedzi spinów
                if i < j:  # Licz tylko jedną stronę interakcji
                    energy -= np.cos(self.spin_config[i] - self.spin_config[j])
        return energy / self.num_spins  # Normalizuj na spin


        
    ## Let the system evolve to equilibrium state
    def equilibrate(self,max_nsweeps=int(1e4),temperature=None,H=None,show = False):
        if temperature != None:
            self.temperature = temperature
        dic_thermal_t = {}
        dic_thermal_t['energy']=[]
        beta = 1.0/self.temperature
        energy_temp = 0
        for k in list(range(max_nsweeps)):
            self.sweep()     
            #list_M.append(np.abs(np.sum(S)/N))
            energy = np.sum(self.get_energy())/self.num_spins/2
            dic_thermal_t['energy'] += [energy]
            #print( abs(energy-energy_temp)/abs(energy))
            if show  & (k%1e3 ==0):
                print('#sweeps=%i'% (k+1))
                print('energy=%.2f'%energy)
                self.show()
            if ((abs(energy-energy_temp)/abs(energy)<1e-4) & (k>500)) or k == max_nsweeps-1:
                print('\nequilibrium state is reached at T=%.1f'%self.temperature)
                print('#sweep=%i'%k)
                print('energy=%.2f'%energy)
                break
            energy_temp = energy
        nstates = len(dic_thermal_t['energy'])
        energy=np.average(dic_thermal_t['energy'][int(nstates/2):])
        self.energy = energy
        energy2=np.average(np.power(dic_thermal_t['energy'][int(nstates/2):],2))
        self.Cv=(energy2-energy**2)*beta**2

    ## To see thermoquantities evolve as we cooling the systems down
    # input: T_inital: initial tempreature
    #        T_final: final temperature
    #        sample/'log' or 'lin',mean linear sampled T or log sampled( centered at critical point)
    def annealing(self, T_init=2.5, T_final=0.1, nsteps=20, show_equi=False):
        dic_thermal = {}
        dic_thermal['temperature'] = list(np.linspace(T_init, T_final, nsteps))
        dic_thermal['energy'] = []
        dic_thermal['Cv'] = []
        all_vortices = []
        paired_vortices_list = []

        for T in dic_thermal['temperature']:
            self.equilibrate(temperature=T)
            if show_equi:
                self.show(colored=True)
            dic_thermal['energy'] += [self.energy]
            dic_thermal['Cv'] += [self.Cv]

            vortices, antivortices = self.find_vortices()
            all_vortices.append(len(vortices))

            vortex_coords = np.array(vortices)
            antivortex_coords = np.array(antivortices)
            if len(vortex_coords) > 0 and len(antivortex_coords) > 0:  # Sprawdzenie czy istnieją wiry i antywiry
                distances = np.linalg.norm(vortex_coords[:, np.newaxis, :] - antivortex_coords[np.newaxis, :, :],
                                           axis=2)
                threshold = 3  # Próg parowania

                paired_vortices = 0
                for i in range(len(vortices)):
                    if np.any(distances[i] < threshold):
                        paired_vortices += 1
                paired_vortices_list.append(paired_vortices)
            else:
                paired_vortices_list.append(0)  # Dodanie 0 jeśli nie ma wirów lub antywirów

        plt.plot(dic_thermal['temperature'], dic_thermal['Cv'], '.')
        plt.ylabel(r'$C_v$')  # Poprawka: Cv zamiast magnetyzacji
        plt.xlabel('Temperatura')
        plt.show()

        plt.plot(dic_thermal['temperature'], dic_thermal['energy'], '.')
        plt.ylabel(r'$\langle E \rangle$')  # Poprawka: energia zamiast wirowości
        plt.xlabel('Temperatura')
        plt.show()

        plt.plot(dic_thermal['temperature'], all_vortices, label="Wszystkie wiry")
        plt.plot(dic_thermal['temperature'], paired_vortices_list, label="Sparowane wiry")
        plt.xlabel("Temperatura")
        plt.ylabel("Liczba wirów")
        plt.legend()
        plt.title("Liczba wirów w funkcji temperatury")
        plt.show()

        return dic_thermal

    @staticmethod
    ## convert configuration inz list to matrix form
    def list2matrix(S):
        N=int(np.size(S))
        L = int(np.sqrt(N))
        S=np.reshape(S,(L,L))
        return S
    
    def find_vortices(self):
        """Znajduje sparowane wiry i antywiry w konfiguracji spinów."""
        vortices = []
        antivortices = []

        L = self.width
        for i in range(L):
            for j in range(L):
                index = i * L + j
                neighbors = self.nbr[index]
                angle_sum = 0
                for n in neighbors:
                    angle_diff = self.spin_config[n] - self.spin_config[index]
                    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi  # Zmiana zakresu na [-pi, pi]
                    angle_sum += angle_diff
                winding_number = int(round(angle_sum / (2 * np.pi)))
                if winding_number == 1:
                    vortices.append((i, j))
                elif winding_number == -1:
                    antivortices.append((i, j))

        # Znajdowanie par wirów i antywirów
        threshold = 1  # Maksymalna odległość między wirami i antywirami
        paired_vortices = []  # Lista przechowująca pary (wir, antywir)
        used_antivortices = set()  # Aby uniknąć wielokrotnego użycia antywirów

        for vortex in vortices:
            vx, vy = vortex
            for antivortex in antivortices:
                if antivortex in used_antivortices:
                    continue  # Jeśli antywir został już użyty, pomiń
                ax, ay = antivortex
                distance = np.sqrt((vx - ax)**2 + (vy - ay)**2)
                if distance <= threshold:  # Jeśli wir i antywir są wystarczająco blisko
                    paired_vortices.append((vortex, antivortex))
                    used_antivortices.add(antivortex)  # Oznacz antywir jako użyty
                    break

        # Przekształć sparowane wiry i antywiry w oddzielne listy
        vortices = [v[0] for v in paired_vortices]
        antivortices = [v[1] for v in paired_vortices]

        return vortices, antivortices


    ## visulize a configurtion
    #  input：S/ spin configuration in list form
    def show(self, colored=True):
        """Wizualizuje konfigurację spinów, wyróżniając wiry i antywiry."""
        config_matrix = self.list2matrix(self.spin_config)
        X, Y = np.meshgrid(np.arange(0, self.width), np.arange(0, self.width))
        U = np.cos(config_matrix)
        V = np.sin(config_matrix)

        vortices, antivortices = self.find_vortices()

        plt.figure(figsize=(4, 4), dpi=100)
        Q = plt.quiver(X, Y, U, V, units='width')
        qk = plt.quiverkey(Q, 0.1, 0.1, 1, r'$spin$', labelpos='E', coordinates='figure')

        if colored:
            for vortex in vortices:
                x, y = vortex
                plt.gca().add_patch(plt.Rectangle((y - 0.5, x - 0.5), 1, 1, color='red', alpha=0.5))

            for antivortex in antivortices:
                x, y = antivortex
                plt.gca().add_patch(plt.Rectangle((y - 0.5, x - 0.5), 1, 1, color='blue', alpha=0.5))

                vortex_coords = np.array(vortices)
                antivortex_coords = np.array(antivortices)

                distances = np.linalg.norm(vortex_coords[:, np.newaxis, :] - antivortex_coords[np.newaxis, :, :],
                                           axis=2)

                threshold = 1  # Przykładowy próg - dostosuj do swoich potrzeb

                paired_vortices = 0
                for i in range(len(vortices)):
                    if np.any(distances[i] < threshold):
                        paired_vortices += 1


        plt.title('T=%.2f' % self.temperature + ', #spins=' + str(self.width) + 'x' + str(self.width))
        plt.axis('off')
        plt.show()
        
        