import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    def single_sim(self, voltages, ns, times, time_max, 
                        title, folder, sim_name, show=False):
        fig, ax = plt.subplots(2, 1, figsize=(10,10))
        
        ax[0].plot(times, voltages)
        ax[0].set_xlabel('Time (ms)')
        ax[0].set_ylabel('Voltage (mV)')
        ax[0].set_xlim(0, time_max)

        ax[1].plot(times, ns)
        ax[1].set_xlabel('Time (ms)')
        ax[1].set_ylabel('n')
        ax[1].set_xlim(0, time_max)

        fig.suptitle(title)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        out_file = f'{folder}/single_sim_{sim_name}.png'
        plt.savefig(out_file, bbox_inches='tight')
        if show: plt.show()
        plt.close(fig)
        print(f'saved {out_file}')

        return None
    
    def two_signals_same_domain(self, signal_1, label_1, 
                                     signal_2, label_2, times, 
                                     time_max, y_label, title, 
                                     folder, sim_name, show=False):
        fig, ax = plt.subplots()
        ax.plot(times, signal_1, label=label_1)
        ax.plot(times, signal_2, label=label_2)
        ax.set_xlabel('Time (ms)')
        ax.set_xlim(0, time_max)
        ax.set_ylabel(y_label)
        ax.set_title(title, fontsize=15)
        ax.legend()

        fig.tight_layout()

        out_file = f'{folder}/two_signals_{sim_name}.png'
        fig.savefig(out_file, bbox_inches='tight')
        if show: plt.show()
        plt.close(fig)

        print(f'saved {out_file}')
        return None
    
    def plot_one_biff_util(self, ax, data, legend):
        equilibria_i_v = data['equilibria_i_v']
        asc_pass = data['asc_pass_iapps_vmin_vmax']
        des_pass = data['des_pass_iapps_vmin_vmax']

        i_eq = equilibria_i_v[0]
        v_eq = equilibria_i_v[1]

        ax.scatter(i_eq, v_eq, s=7, label=f'{legend} equilibria')
        
        i_app_asc = asc_pass[0]
        v_min_asc = asc_pass[1]
        v_max_asc = asc_pass[2]
        ax.plot(i_app_asc, v_min_asc, linestyle='-', label=f'{legend} min asceding')
        ax.plot(i_app_asc, v_max_asc, linestyle='-', label=f'{legend} max asceding')

        i_app_des = des_pass[0]
        v_min_des = des_pass[1]
        v_max_des = des_pass[2]
        ax.plot(i_app_des, v_min_des, linestyle=':', label=f'{legend} min descending')
        ax.plot(i_app_des, v_max_des, linestyle=':', label=f'{legend} max descending')

        return None
    
    def plot_one_biff(self, data, title, folder, sim_name, parameters_array, show):
        fig, ax = plt.subplots()

        self.plot_one_biff_util(ax, data, '')

        ax.set_xlabel('I_app')
        ax.set_ylabel('V (mV)')
        ax.set_title(title, fontsize=15)
        ax.grid(True)
        ax.legend(loc='best', fontsize='large')

        fig.tight_layout()

        out_file = f'{folder}/biff_{sim_name}'
        fig.savefig(out_file + '.png')
        if show: plt.show()
        plt.close(fig)
        print(f'Saved {out_file}')
        if parameters_array is not None:
            np.savetxt(out_file + '.txt', np.asarray(parameters_array))

        return None
    
    def plot_two_biff(self, data_1, data_2, legends, title, folder, sim_name, parameters_array, show):
        fig, ax = plt.subplots()

        self.plot_one_biff_util(ax, data_1, legends[0])
        self.plot_one_biff_util(ax, data_2, legends[1])

        ax.set_xlabel('I_app')
        ax.set_ylabel('V (mV)')
        ax.set_title(title, fontsize=12.5)
        ax.grid(True)
        #ax.legend(loc='best')

        fig.tight_layout()

        out_file = f'{folder}/biff_{sim_name}'
        fig.savefig(out_file + '.png')
        if show: plt.show()
        plt.close(fig)
        print(f'Saved {out_file}')
        np.savetxt(out_file + '.txt', np.asarray(parameters_array))

        return None
