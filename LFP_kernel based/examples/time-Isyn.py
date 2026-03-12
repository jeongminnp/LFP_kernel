from testfile_1021 import RadialNeuron 
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
from scipy.spatial import distance_matrix
import random as pyrandom
import json 
from scipy import signal

AP, ML, DV = 621.052*2, 465.905*2, 660.0483*2 
N_STN = 388
np.random.seed(74)
cx, cy, cz = AP/2, ML/2, DV/2 
noise_range = 5 
distance = 20

def in_ellipsoid(p, a=AP/2, b=ML/2, c=DV/2, center=(cx, cy, cz)): 
    x, y, z = p 
    cx, cy, cz = center 
    return ((x - cx)/a)**2 + ((y - cy)/b)**2 + ((z - cz)/c)**2 <= 1 

x = np.arange(0, AP, distance) 
y = np.arange(0, ML, distance) 
z = np.arange(0, DV, distance) 
X, Y, Z = np.meshgrid(x, y, z, indexing='ij') 
candidate_positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T 
candidate_positions = np.array([p for p in candidate_positions if in_ellipsoid(p)]) 
candidate_positions = np.array(candidate_positions)
np.random.shuffle(candidate_positions)

final_positions = []
for i, p in enumerate(candidate_positions):
    if len(final_positions) >= N_STN:
        break
    if i < N_STN // 2:
        noise = np.random.uniform(-noise_range, noise_range, 3)
    else:
        noise = np.zeros(3)
    candidate = p + noise
    if in_ellipsoid(candidate):
        final_positions.append(candidate)
        
final_positions = np.array(final_positions)
neurons = [] 
for pos in final_positions: 
    neuron = RadialNeuron() 
    neuron.position = pos 
    neuron.place_synapses(n_exc=100, n_inh=30)
    neurons.append(neuron) 

TARGET_STN = 118
print(f"\n Selected STN neuron index: {TARGET_STN}\n")
target_neuron = neurons[TARGET_STN]

exc_indices = [i for i, s in enumerate(target_neuron.syn_info) if s['type'] == 'exc']
inh_indices = [i for i, s in enumerate(target_neuron.syn_info) if s['type'] == 'inh']
print(f"  total synapses: {len(target_neuron.syn_info)}")
print(f"  exc: {len(exc_indices)}, inh: {len(inh_indices)}\n")

# Cortex→STN

data = np.loadtxt("Cortex_spike_times_normal.txt", skiprows=2)
N_CTX = 388

dt = 0.1
lfp_time = np.arange(2000, 5000+dt, dt)

cortex_spike_times = {}
for cx in range(N_CTX):
    mask = (data[:,0] == cx) & (data[:,1] >= 2000) & (data[:,1] <= 5000)
    cortex_spike_times[cx] = data[mask, 1].tolist()

stn_syn_events = {}
N_EXC_STN = 100

for stn_idx, neuron in enumerate(neurons):
    exc_indices_n = [i for i, s in enumerate(neuron.syn_info) if s['type'] == 'exc']
    chosen_stn_syns = np.random.choice(exc_indices_n, size=N_EXC_STN, replace=False)
    chosen_ctx_neurons = np.random.choice(N_CTX, size=N_EXC_STN, replace=False)
    for ctx_id, stn_syn_idx in zip(chosen_ctx_neurons, chosen_stn_syns):
        spikes = cortex_spike_times.get(ctx_id, [])
        if len(spikes) == 0:
            continue
        stn_syn_events[(stn_idx, stn_syn_idx)] = spikes.copy()

syn_index_map = {}
for n_idx, neuron in enumerate(neurons):
    base_pos = np.array(neuron.position)
    for s_idx, info in enumerate(neuron.syn_info):
        sec = info.get('section', None)
        typ = info.get('type', 'exc')
        pos_local = np.array([0.0, 0.0, 0.0])

        if sec is neuron.soma:
            theta = info.get('theta', None)
            phi = info.get('phi', None)
            if theta is not None and phi is not None:
                r = neuron.soma_radius
                pos_local = np.array([r * np.sin(phi) * np.cos(theta),
                                      r * np.sin(phi) * np.sin(theta),
                                      r * np.cos(phi)])
        elif sec in neuron.dendrites:
            idx = neuron.dendrites.index(sec)
            p0, p1 = neuron.dend_coords[idx]
            x_rel = info.get('x', 0.5)
            try: x_rel = float(x_rel)
            except:
                try: x_rel = float(x_rel.hocval())
                except: x_rel = 0.5
            p0 = np.array(p0); p1 = np.array(p1)
            pos_local = p0 + x_rel * (p1 - p0)
        elif sec in neuron.branches:
            idx = neuron.branches.index(sec)
            p0, p1 = neuron.branch_coords[idx]
            x_rel = info.get('x', 0.5)
            try: x_rel = float(x_rel)
            except:
                try: x_rel = float(x_rel.hocval())
                except: x_rel = 0.5
            p0 = np.array(p0); p1 = np.array(p1)
            pos_local = p0 + x_rel * (p1 - p0)
        else:
            if 'pos' in info:
                pos_local = np.array(info['pos'])

        pos_um = base_pos + pos_local
        syn_index_map[(int(n_idx), int(s_idx))] = {
            'pos_um': pos_um.tolist(),
            'type': str(typ)
        }

# GPe→STN

N_GPeT1 = 988
N_INH_STN = 30

gpe_data = np.loadtxt("GPeT1_spike_times_normal.txt", skiprows=2)
mask = (gpe_data[:,1] >= lfp_time[0]) & (gpe_data[:,1] <= lfp_time[-1])
gpe_window_data = gpe_data[mask]

gpe_spike_dict = {}
for g in range(N_GPeT1):
    g_mask = (gpe_window_data[:,0] == g)
    gpe_spike_dict[g] = gpe_window_data[g_mask, 1].tolist()

stn_gpe_syn_events = {}
for stn_idx, neuron in enumerate(neurons):
    inh_indices_n = [i for i, s in enumerate(neuron.syn_info) if s['type'] == 'inh']
    chosen_stn_syns = np.random.choice(inh_indices_n, size=N_INH_STN, replace=False)
    chosen_gpe_neurons = np.random.choice(N_GPeT1, size=N_INH_STN, replace=False)
    for gpe_id, stn_syn_idx in zip(chosen_gpe_neurons, chosen_stn_syns):
        spikes = gpe_spike_dict.get(int(gpe_id), [])
        if len(spikes) == 0:
            continue
        stn_gpe_syn_events[(stn_idx, stn_syn_idx)] = spikes.copy()

stn_data = np.loadtxt("STN_spike_times_normal.txt", skiprows=2)
mask_window = (stn_data[:,1] >= lfp_time[0]) & (stn_data[:,1] <= lfp_time[-1])
stn_window_data = stn_data[mask_window]

stn_spike_dict = {}
for n_idx in range(len(neurons)):
    mask = (stn_window_data[:,0] == n_idx)
    spikes = stn_window_data[mask, 1].tolist()
    if spikes:
        stn_spike_dict[n_idx] = spikes

def isyn_kernel(time_axis, spike_times, tau_rise, tau_decay, g_max):
    g = np.zeros_like(time_axis)
    for t_sp in spike_times:
        delta = time_axis - t_sp
        mask = delta >= 0
        g[mask] += g_max * (
            np.exp(-delta[mask] / tau_decay) -
            np.exp(-delta[mask] / tau_rise)
        )
    return g

# Task 1 : synaptic I_syn

V_rest = -65.0  # mV

syn_type_params = {

    'exc': {
        'tau_rise':  0.5,    # ms
        'tau_decay': 2.0,    # ms
        'g_max':     1.0,    
        'E_rev':     0.0     # mV
    },
    'inh': {
        'tau_rise':  0.5,    # ms
        'tau_decay': 6.0,    # ms
        'g_max':     1.0,   
        'E_rev':    -80.0    # mV
    }
}

isyn_per_syn = {}  # {syn_idx: I_syn_array}

# exc syn (CTX→STN)
for (stn_idx, syn_idx), spikes in stn_syn_events.items():
    if stn_idx != TARGET_STN:
        continue
    p = syn_type_params['exc']
    g = isyn_kernel(lfp_time, np.array(spikes),
                    p['tau_rise'], p['tau_decay'], p['g_max'])
    isyn_per_syn[syn_idx] = g * (V_rest - p['E_rev'])

# inh syn (GPe→STN)
for (stn_idx, syn_idx), spikes in stn_gpe_syn_events.items():
    if stn_idx != TARGET_STN:
        continue
    p = syn_type_params['inh']
    g = isyn_kernel(lfp_time, np.array(spikes),
                    p['tau_rise'], p['tau_decay'], p['g_max'])
    isyn_per_syn[syn_idx] = g * (V_rest - p['E_rev'])

for syn_idx in sorted(isyn_per_syn.keys())[:5]:
    stype = syn_index_map[(TARGET_STN, syn_idx)]['type']
    print(f"  syn_idx={syn_idx}, type={stype}, "
          f"max={isyn_per_syn[syn_idx].max():.3f}, "
          f"min={isyn_per_syn[syn_idx].min():.3f}")

all_synkeys = sorted(isyn_per_syn.keys())

exc_keys = [k for k in all_synkeys if syn_index_map[(TARGET_STN, k)]['type'] == 'exc']
inh_keys = [k for k in all_synkeys if syn_index_map[(TARGET_STN, k)]['type'] == 'inh']

chosen_exc = [exc_keys[i] for i in np.linspace(0, len(exc_keys)-1, 3, dtype=int)]
chosen_inh = [inh_keys[i] for i in np.linspace(0, len(inh_keys)-1, 2, dtype=int)]
chosen_syn_indices = sorted(chosen_exc + chosen_inh)

print(f"\n selected 5 syns : {chosen_syn_indices}")
for idx in chosen_syn_indices:
    stype = syn_index_map[(TARGET_STN, idx)]['type']
    print(f"  syn {idx}: {stype}")

fig, axes = plt.subplots(len(chosen_syn_indices), 1,
                          figsize=(14, 12), sharex=True)
fig.suptitle(f"STN Neuron {TARGET_STN}: I_syn per Synapse (5 examples)\n"
             f"Red=excitatory(CTX→STN), Blue=inhibitory(GPe→STN)",
             fontsize=13, fontweight='bold')

for ax, syn_idx in zip(axes, chosen_syn_indices):
    stype = syn_index_map[(TARGET_STN, syn_idx)]['type']
    color = 'tomato' if stype == 'exc' else 'steelblue'
    I = isyn_per_syn[syn_idx]

    ax.plot(lfp_time, I, color=color, lw=0.8, label=f'Syn {syn_idx} ({stype})')
    ax.axhline(y=0, color='gray', lw=0.5, ls='--', alpha=0.5)

    ax.set_ylabel('I_syn (a.u.)', fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (ms)', fontsize=12)
plt.tight_layout()
plt.savefig('task1_5synapses_isyn.png', dpi=150, bbox_inches='tight')
plt.show()

# I_syn_total

I_syn_total = np.zeros_like(lfp_time)
for I in isyn_per_syn.values():
    I_syn_total += I

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

for syn_idx in chosen_syn_indices:
    stype = syn_index_map[(TARGET_STN, syn_idx)]['type']
    color = 'tomato' if stype == 'exc' else 'steelblue'
    axes[0].plot(lfp_time, isyn_per_syn[syn_idx],
                 color=color, lw=0.7, alpha=0.8, label=f'syn{syn_idx}({stype})')
axes[0].axhline(y=0, color='gray', lw=0.5, ls='--', alpha=0.5)
axes[0].set_ylabel('I_syn (a.u.)', fontsize=11)
axes[0].set_title('5 example synapses (red=exc, blue=inh)', fontsize=11)
axes[0].legend(fontsize=8, loc='upper right', ncol=5)
axes[0].grid(True, alpha=0.3)

axes[1].plot(lfp_time, I_syn_total, color='purple', lw=1.0)
axes[1].axhline(y=0, color='gray', lw=0.5, ls='--', alpha=0.5)
axes[1].set_ylabel('I_syn_total (a.u.)', fontsize=11)
axes[1].set_xlabel('Time (ms)', fontsize=12)
axes[1].set_title(f'I_syn_total  ({len(isyn_per_syn)} synapses summed)', fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.suptitle(f'STN Neuron {TARGET_STN}: I_syn Summary', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('task1_isyn_total.png', dpi=150, bbox_inches='tight')
plt.show()
