import matplotlib.pyplot as plt

#nsteps = [1, 3, 5, 10, 20, 30, 40]
C=24.0/3.7
nsteps = [1, 3, 5, 10, 20, 30, 40]
distance = {
    'vg': [x/C for x in [46, 46, 45, 89, 101]], 
    'svg': [x/C for x in [89, 91, 87, 89, 88]],
    'mpplum': [x/C for x in [473, 465, 515, 718, 978, 1019, 1003]], 
    'smpplum': [x/C for x in [451, 436, 463, 608, 962, 1025, 1026]],
    'smpplum-i': [x/C for x in [173, 231, 235, 284, 333, 349, 386]], 
    'mpplum-i': [x/C for x in [330, 462, 795, 926, 1010, 1025, 972]], 
    '1step-il': [x/C for x in [322 for k in range(len(nsteps))]], 
    'no-action': [x/C for x in [566 for k in range(len(nsteps))]], 
    'human': [x/C for x in [1358 for k in range(len(nsteps))]]
}

success = {
    'mpplum': [10.0, 9.2, 11.4, 20.5, 56.4, 61.7, 64.4], 
    'smpplum': [9.2, 13.0, 15.0, 30.4, 59.1, 63.7, 63.2], 
    'vg': [0, 0, 0, 0, 0], 
    'svg': [0, 0, 0, 0, 0],
    'smpplum-i': [0.0, 0.0, 0.0, 0.7, 1.2], 
    'mpplum-i': [6.6, 15.4, 36.2, 60.5, 62.5, 63.7, 58.9],     
#    'smpplum-i': [0.0, 0.0, 0.0, 0.7, 1.2, 0.7, 0.8], 
#    'mpplum-i': [6.6, 15.4, 36.2, 60.5, 62.5, 63.7, 58.9], 
    '1step-il': [1.4 for k in range(len(nsteps))], 
    'no-action': [16.2 for k in range(len(nsteps))], 
    'human': [100 for k in range(len(nsteps))]
}

plot = 2
if plot == 1:
    plt.ion()
    s=10
    w=3
    plt.plot(nsteps, distance['human'], '--', markersize=s, c='black', linewidth=w)
    plt.plot(nsteps[:len(distance['mpplum'])], distance['mpplum'], '^-', markersize=s, linewidth=w, c='purple')
    plt.plot(nsteps[:len(distance['smpplum'])], distance['smpplum'], 'o-', markersize=s, linewidth=w, c='purple')
    plt.plot(nsteps[:len(distance['mpplum-i'])], distance['mpplum-i'], '^-', markersize=s, linewidth=w, c='blue')
    plt.plot(nsteps[:len(distance['smpplum-i'])], distance['smpplum-i'], 'o-', markersize=s, linewidth=w, c='blue')
    plt.plot(nsteps[:len(distance['vg'])], distance['vg'], '^-', markersize=s, linewidth=w, c='green')
    plt.plot(nsteps[:len(distance['svg'])], distance['svg'], 'o-', markersize=s, linewidth=w, c='green')
    plt.plot(nsteps, distance['no-action'], '--', markersize=s, linewidth=w, c='magenta')
    plt.plot(nsteps, distance['1step-il'], '--', markersize=s, linewidth=w, c='red')
    plt.legend(['Human', 'VG', 'SVG', 'MPPLUM-I', 'SMPPLUM-I', 'No action', '1-step IL'])
    plt.xlabel('Rollout length', fontsize=12)
    plt.ylabel('Mean Distance (meters)', fontsize=12)
    plt.xticks([1, 5, 10, 15, 20, 30, 40])
    plt.legend(['Human', 'MPUR', 'S-MPUR','MPIL', 'S-MPIL', 'VG', 'SVG', 'No action', '1-step IL'])
    plt.savefig('figures/driving/distance_perf.pdf')
elif plot == 2:
    plt.ion()
    s=10
    w=3
    plt.plot(nsteps, success['human'], '--', markersize=s, c='black', linewidth=w)
    plt.plot(nsteps[:len(success['mpplum'])], success['mpplum'], '^-', markersize=s, linewidth=w, c='purple')
    plt.plot(nsteps[:len(success['smpplum'])], success['smpplum'], 'o-', markersize=s, linewidth=w, c='purple')
    plt.plot(nsteps[:len(success['mpplum-i'])], success['mpplum-i'], '^-', markersize=s, linewidth=w, c='blue')
    plt.plot(nsteps[:len(success['smpplum-i'])], success['smpplum-i'], 'o-', markersize=s, linewidth=w, c='blue')
    plt.plot(nsteps[:len(success['vg'])], success['vg'], '^-', markersize=s, linewidth=w, c='green')
    plt.plot(nsteps[:len(success['svg'])], success['svg'], 'o-', markersize=s, linewidth=w, c='green')
    plt.plot(nsteps, success['no-action'], '--', markersize=s, linewidth=w, c='magenta')
    plt.plot(nsteps, success['1step-il'], '--', markersize=s, linewidth=w, c='red')
    plt.xlabel('Rollout length')
    plt.ylabel('Success %')
    plt.xticks([1, 5, 10, 15, 20, 30, 40])
    plt.legend(['Human', 'MPUR', 'S-MPUR', 'MPIL', 'SMPIL', 'VG', 'SVG', 'No action', '1-step IL'])
    plt.savefig('figures/driving/success_perf.pdf')
