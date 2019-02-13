"""Programme containing energy plotter for energy conservation data,
    produced in model runs for Hamiltonian energy conserving framework
    for rotating shallow water equations and dry Euler equations"""

# pylint: disable=invalid-name
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

def plot_nc_data(diag, paths, info,
                 start_t=None, end_t=None, step=None,
                 fontsize=10, lnwdth=1., legends=True,
                 clrs = ['darkcyan', 'purple', 'yellowgreen',
                         'crimson', 'saddlebrown']):
    """Given nc diagnostic, plot curve wrt time
       arg diag: string for the diagnostic
       arg paths: paths leading to nc files with energy data (data length
                  to be plotted defaults to file with shortest data set)
       arg info: 3-tuple containing resolution, time step,
                 and model name (for plot title)
       arg start_t : number of time steps from which to start from.
                     Default None implies start_t = 0
       arg end_t: number of times steps up to which to plot the
                  energy curve to. Default None sets end_t to length
                  of dataset
       arg step: Only plot every `step`th data point (for large
                 data sets to save plot compilation time); defaults to 1
       arg fontsize: fontsize for plot (labels, ticks); defaults to 10
       arg lnwdth: linewidth for plot; defaults to 1.
       arg legends: contain legend in plots; defaults to True
       arg clrs: list of plot curves' colours. Defaults list contains
                 i.a. darkcyan, purple."""

    res, dt, models = info
    
    nc_data = []
    for i in range(len(paths)):
        nc_data.append(Dataset(paths[i]))
    
    if start_t is None:
        s = 0
    else:
        s = start_t

    if end_t is None:
        end_t = min([len(nc_data[i].dimensions['time'])
                    for i in range(len(paths))])

    if step is None:
        rel_vals = np.zeros((end_t - s, len(nc_data)))
        times = np.zeros(end_t - s)
        step = 1
    else:
        rel_vals = np.zeros((int((end_t - s)/step), len(nc_data)))
        times = np.zeros(int((end_t - s)/step))

    nd = nc_data
    for j in range(len(nd)):
        _c = 0
        for i in range(s, end_t, step):
            rel_vals[_c][j] = (nd[j].groups[diag].variables['total'][i]
                               -nd[j].groups[diag].variables['total'][0])
            if diag[:8] != 'L2_error':
                rel_vals[_c][j] /= nd[j].groups[diag].variables['total'][0]

            times[_c] = _c*step
            _c += 1

        plt.plot(times, rel_vals[:, j], color=clrs[j], linewidth=lnwdth)

    plt.title('Relative {0} development, res {1}'\
              ', dt {2} \n'.format(diag, res, dt), loc='left',
              fontsize=fontsize)
    #plt.title('Golo Wimmer', loc='right')
    if legends:
        plt.legend(models)
    plt.ylabel('Relative {0} error'.format(diag), fontsize=fontsize)
    plt.xlabel('Time steps', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks (fontsize=fontsize)
    plt.gca().yaxis.get_offset_text().set_size(fontsize)
    plt.grid(which='both')
    plt.show()

    # Below code plots diag value averaged over [start_t, end_t], for
    # refinement levels 3, 4, 5 (and hence requires as input three nc
    # file paths to work).
    if False:
        intrvl = end_t - s
        av_vals = [sum(nd[j].groups[diag].variables['total'][s:end_t])/
                   intrvl for j in [0, 1, 2]]

        slopes = [1, 2, 3]
        lnst = ['dashed', 'dotted', 'dashdot']
        sphere = True
        if sphere:
            diam = 2*6371220*np.pi/1000. # cell side length in km
            cell_h = [diam/(4*2**j) for j in [3, 4, 5]]
        else:
            cell_h = [1/(16*2**j) for j in [0, 1, 2]]
        plt.plot(cell_h, av_vals, 'x', color=clrs[1])

        _c = 0
        for s in slopes:
            cs = 1.2*av_vals[0]/(cell_h[0]**(s))
            slope_vals = [cs*j**s for j in cell_h]
            plt.plot(cell_h, slope_vals, color=clrs[0], linestyle=lnst[_c])
            _c += 1

        s_legend = [r'$\propto \; \Delta x^{0}$'.format(abs(j))
                    for j in slopes]
        plt.legend(('{0} error'.format(diag[-1]), *s_legend))
        plt.title('Williamson 2 L2 error after 50 days,\n averaged over' \
                  'last {0} time steps)\n'.format(intrvl))
        plt.yscale('log')
        plt.xscale('log')
        plt.xlim([1.1*cell_h[0], 0.9*cell_h[-1]])
        plt.grid(True, which='both', linestyle=':')
        plt.ylabel('$L^2$ error', fontsize=fontsize)
        plt.xlabel('Cell side length $\Delta x$', fontsize=fontsize)
        plt.show()


if __name__ == "__main__":
    # Discretisation, diagnostic choices
    # Diagnostic to plot
    diag_choices = ['Energy', 'Mass', 'Enstrophy',
                    'L2_error_u', 'L2_error_D']
    diagnostic=diag_choices[0]
    # path to nc file
    sim_choices = ['W2', 'W5', 'Galewsky']
    discr_choices = ['nec', 'ec_D-ad', 'ec_flux']

    path_list, discr_list = [], []
    def path_list_extender(sim_c, d_c):
        new_discr = sim_choices[sim_c] + '_' + discr_choices[d_c]
        outp = '/Users/gaw16/ICL/PhD/HPC/CX1/Files/'
        new_path = outp + 'diagnostics_{0}.nc'.format(new_discr)
        path_list.append(new_path)
        discr_list.append(new_discr)

    # Use path list extender to add paths to diagnostic files to
    # path_list, each of which is then plotted in the same plot.
    path_list_extender(1, 1)
    path_list_extender(1, 2)

    # model info for plot: res, dt, name
    model_info = (5, 50, discr_list)

    plot_nc_data(diagnostic, path_list, model_info,
                 start_t=None, end_t=None, step=1,
                 fontsize=10, lnwdth=1., legends=True)
