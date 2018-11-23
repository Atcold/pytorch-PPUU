import matplotlib.pyplot as plt
import numpy as np
plt.ion()

time_steps = [1, 3, 5, 10, 20, 30, 40]

success = {
    'MPUR-s-drop-mean': np.array([
        0.09833333333, 
        0.162, 
        0.224, 
        0.5826666667, 
        0.5916666667, 
        0.7523333333, 
        0.7486666667
    ]),
    'MPUR-s-drop-std': np.array([
        0.03458805189, 
        0.011, 
        0.09180413934, 
        0.05651843357, 
        0.04960174728, 
        0.05316327053, 
        0.02967041175
    ]),
    'MPUR-s-mean': np.array([
        0.1353333333, 
        0.126, 
        0.146, 
        0.3626666667, 
        0.521, 
        0.663, 
        0.7183333333
    ]), 
    'MPUR-s-std': np.array([
        0.0150443788, 
        0.02066397832, 
        0.03629049462, 
        0.04705670338, 
        0.0238117618, 
        0.02364318084, 
        0.007505553499
    ]),
    'MPUR-d-mean': np.array([
        0.081, 
        0.08466666667, 
        0.07666666667, 
        0.2003333333, 
        0.5846666667, 
        0.691, 
        0.6266666667
    ]),
    'MPUR-d-std': np.array([
        0.02707397274, 
        0.009073771726, 
        0.03950105484, 
        0.1538191579, 
        0.04717343885, 
        0.01571623365, 
        0.01823001189
    ]),
    'MPER-d-mean': np.array([
        0.02233333333, 
        0.157, 
        0.4026666667,
        0.5843333333,
        0.6236666667,
        0.6373333333,
        0.6193333333
    ]),
    'MPER-d-std': np.array([
        0.03868246804, 
        0.01374772708, 
        0.03557152419, 
        0.04829423706, 
        0.05001333156, 
        0.005507570547, 
        0.02706165799
    ])
}


distance = {
    'MPUR-s-drop-mean': np.array([
        68.60416667, 
        86.64166667, 
        101.6986111, 
        146.4069444, 
        152.3680556, 
        171.1763889, 
        171.2277778
    ]),
    'MPUR-s-drop-std': np.array([
        16.60139511, 
        9.745479721, 
        10.79203372, 
        2.012057117, 
        6.278073524, 
        4.500857043, 
        3.876711838
    ]),
    'MPUR-s-mean': np.array([
        77.64861111, 
        74.41111111, 
        79.19027778, 
        119.4277778, 
        143.5805556, 
        160.1791667, 
        166.8597222
    ]),
    'MPUR-s-std': np.array([
        4.050605093, 
        2.093111774, 
        5.438246011, 
        5.494767608, 
        2.081725769, 
        2.634401155, 
        2.389997724
    ]),
    'MPUR-d-mean': np.array([
        67.26805556, 
        64.18472222, 
        64.85277778, 
        90.39305556, 
        152.0597222, 
        162.4402778, 
        155.0916667
    ]), 
    'MPUR-d-std': np.array([
        1.98230599, 
        2.653876663, 
        4.021160033, 
        28.82806742, 
        4.579381503, 
        2.763556652, 
        3.785725657
    ]),
    'MPER-d-mean': np.array([
        35.86944444, 
        80.32083333, 
        129.6027778, 
        146.4069444, 
        154.3722222, 
        157.3527778, 
        154.1666667
    ]),
    'MPER-d-std': np.array([
        13.39240193, 
        9.419318395, 
        6.236290984, 
        7.413370444, 
        6.327097167, 
        0.7285632424, 
        3.766844115
    ])
}
    

width = 3
metric = 'distance'
if metric == 'success':
    measure = success
elif metric == 'distance':
    measure = distance



color = 'red'
plt.plot(time_steps, measure['MPER-d-mean'], '-*', color=color, linewidth=width, label='MPER (deterministic)')
ub = measure['MPER-d-mean'] + measure['MPER-d-std']
lb = measure['MPER-d-mean'] - measure['MPER-d-std']
plt.fill_between(time_steps, ub, lb, color=color, alpha=0.2)

    
color = 'blue'
plt.plot(time_steps, measure['MPUR-s-drop-mean'], '-^', color=color, linewidth=width, label='MPUR (stochastic + z-dropout)')
ub = measure['MPUR-s-drop-mean'] + measure['MPUR-s-drop-std']
lb = measure['MPUR-s-drop-mean'] - measure['MPUR-s-drop-std']
plt.fill_between(time_steps, ub, lb, color=color, alpha=0.2)

color = 'magenta'
plt.plot(time_steps, measure['MPUR-s-mean'], '-o', color=color, linewidth=width, label='MPUR (stochastic)')
ub = measure['MPUR-s-mean'] + measure['MPUR-s-std']
lb = measure['MPUR-s-mean'] - measure['MPUR-s-std']
plt.fill_between(time_steps, ub, lb, color=color, alpha=0.2)

color = 'green'
plt.plot(time_steps, measure['MPUR-d-mean'], '-*', color=color, linewidth=width, label='MPUR (deterministic)')
ub = measure['MPUR-d-mean'] + measure['MPUR-d-std']
lb = measure['MPUR-d-mean'] - measure['MPUR-d-std']
plt.fill_between(time_steps, ub, lb, color=color, alpha=0.2)



handles, labels = plt.gca().get_legend_handles_labels()
order = [1, 2, 3, 0]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=12)

#plt.legend(['MPUR (stochastic+z-dropout)', 'MPUR (stochastic)', 'MPUR (deterministic)', 'MPER (deterministic)'], fontsize=12)
plt.xticks(time_steps)
plt.xlabel('Rollout length', fontsize=14)
if metric == 'distance':
    plt.ylabel('Mean distance (meters)', fontsize=14)
elif metric == 'success':
    plt.ylabel('Success %', fontsize=14)
plt.savefig('figures/driving/rollout_length_' + metric + '.pdf')
