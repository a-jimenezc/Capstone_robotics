import numpy as np

def add_noise_to_signal(signal, mean, std, rng):
    noise = rng.normal(loc=mean, scale=std, size=np.size(signal))
    signal_noisy = signal + noise
    return signal_noisy

def results_to_table(results):
    rows = ['$g_L$', '$g_K$', '$g_{Ca}$', '$\\phi$', '$V_1$', '$V_2$', '$V_3$', '$V_4$']
    h = results['hopf']
    s = results['snic']
    m = results['homo']
    lines = []

    for i in range(len(rows)):
        line = ' & '.join([
            rows[i],
            f"{h['actual'][i]:.3f}", 
            f"{h['init_guess_hopf'][i]:.3f}", 
            f"{h['init_guess_snic'][i]:.3f}", 
            f"{h['init_guess_homo'][i]:.3f}",
            f"{s['actual'][i]:.3f}", 
            f"{s['init_guess_hopf'][i]:.3f}", 
            f"{s['init_guess_snic'][i]:.3f}", 
            f"{s['init_guess_homo'][i]:.3f}",
            f"{m['actual'][i]:.3f}", 
            f"{m['init_guess_hopf'][i]:.3f}", 
            f"{m['init_guess_snic'][i]:.3f}", 
            f"{m['init_guess_homo'][i]:.3f}",
        ]) + '\\\\'
        lines.append(line)
    
    table = '\n'.join(lines)

    return table

def percentage_error(estimate, actual):
    return (abs(estimate - actual) / abs(actual)) * 100

def results_to_table_error(results):
    rows = ['$g_L$', '$g_K$', '$g_{Ca}$', '$\\phi$', '$V_1$', '$V_2$', '$V_3$', '$V_4$']
    h = results['hopf']
    s = results['snic']
    m = results['homo']
    lines = []

    for i in range(len(rows)):
        line = ' & '.join([
            rows[i],
            f"{percentage_error(h['init_guess_hopf'][i], h['actual'][i]):.2f}\\%",
            f"{percentage_error(h['init_guess_snic'][i], h['actual'][i]):.2f}\\%",
            f"{percentage_error(h['init_guess_homo'][i], h['actual'][i]):.2f}\\%",
            f"{percentage_error(s['init_guess_hopf'][i], s['actual'][i]):.2f}\\%",
            f"{percentage_error(s['init_guess_snic'][i], s['actual'][i]):.2f}\\%",
            f"{percentage_error(s['init_guess_homo'][i], s['actual'][i]):.2f}\\%",
            f"{percentage_error(m['init_guess_hopf'][i], m['actual'][i]):.2f}\\%",
            f"{percentage_error(m['init_guess_snic'][i], m['actual'][i]):.2f}\\%",
            f"{percentage_error(m['init_guess_homo'][i], m['actual'][i]):.2f}\\%",
        ]) + '\\\\'
        lines.append(line)
    
    table = '\n'.join(lines)

    return table
