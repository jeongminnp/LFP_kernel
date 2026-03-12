from testfile_1021 import RadialNeuron 
import numpy as np
from brian2 import *
from numba import nji
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt 

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

# STN grid placement, N_STN = 388
neurons = [] 
for pos in final_positions: 
    neuron = RadialNeuron() 
    neuron.position = pos
    neuron.place_synapses(n_exc=100, n_inh=30)
    neurons.append(neuron) 

syn_index_map = {}

for n_idx, neuron in enumerate(neurons):
    base_pos = np.array(neuron.position)

    for s_idx, info in enumerate(neuron.syn_info):
        sec = info.get('section', None)
        typ = info.get('type', 'exc')

        pos_local = np.array([0.0, 0.0, 0.0])

        # 1. Soma synapse
        if sec is neuron.soma:
            theta = info.get('theta', None)
            phi = info.get('phi', None)
            if theta is not None and phi is not None:
                r = neuron.soma_radius
                pos_local = np.array([
                    r * np.sin(phi) * np.cos(theta),
                    r * np.sin(phi) * np.sin(theta),
                    r * np.cos(phi)
                ])

        # 2. Dendrite synapse
        elif sec in neuron.dendrites:
            idx = neuron.dendrites.index(sec)
            p0, p1 = neuron.dend_coords[idx]
            x_rel = info.get('x', 0.5)
            try:
                x_rel = float(x_rel)
            except:
                try:
                    x_rel = float(x_rel.hocval())
                except:
                    x_rel = 0.5
            p0 = np.array(p0) 
            p1 = np.array(p1) 
            pos_local = p0 + x_rel * (p1 - p0) 

        # 3. Branch synapse
        elif sec in neuron.branches:
            idx = neuron.branches.index(sec)
            p0, p1 = neuron.branch_coords[idx]
            x_rel = info.get('x', 0.5)
            try:
                x_rel = float(x_rel)
            except:
                try:
                    x_rel = float(x_rel.hocval())
                except:
                    x_rel = 0.5
            p0 = np.array(p0)
            p1 = np.array(p1)
            pos_local = p0 + x_rel * (p1 - p0)

        pos_um = base_pos + pos_local

        syn_index_map[(int(n_idx), int(s_idx))] = {
            'pos_um': pos_um.tolist(),
            'type': str(typ)
        }
syn_index_map_strkey = {f"{k[0]},{k[1]}": v for k, v in syn_index_map.items()}

# CTX→STN
data = np.loadtxt("/path/to/Cortex_spike_times_normal.txt", skiprows=2)
N_CTX = 388

cortex_spike_times = {}
for cx in range(N_CTX):
    mask = (data[:,0] == cx) & (data[:,1] >= 2000) & (data[:,1] <= 5000)
    cortex_spike_times[cx] = data[mask, 1].tolist()

stn_syn_events = {}
N_EXC_STN = 100

for stn_idx, neuron in enumerate(neurons):
    exc_indices = [
        i for i, s in enumerate(neuron.syn_info)
        if s['type'] == 'exc'
    ]

    if len(exc_indices) < N_EXC_STN:
        raise RuntimeError(
            f"STN neuron {stn_idx} has only {len(exc_indices)} excitatory synapses"
        )

    chosen_stn_syns = np.random.choice(
        exc_indices,
        size=N_EXC_STN,
        replace=False
    )

    chosen_ctx_neurons = np.random.choice(
        N_CTX,
        size=N_EXC_STN,
        replace=False
    )

    #  exc syn mapping
    for ctx_id, stn_syn_idx in zip(chosen_ctx_neurons, chosen_stn_syns):
        spikes = cortex_spike_times.get(ctx_id, [])
        if len(spikes) == 0:
            continue
        stn_syn_events[(stn_idx, stn_syn_idx)] = spikes.copy()

stn_syn_events_strkey = {
    f"{k[0]},{k[1]}": v for k, v in stn_syn_events.items()
}

# GPe→STN
gpe_data = np.loadtxt("GPeT1_spike_times_normal.txt", skiprows=2)
N_GPeT1 = 988
N_INH_STN = 30

mask = (gpe_data[:,1] >= 2000) & (gpe_data[:,1] <= 5000)
gpe_window_data = gpe_data[mask]

gpe_spike_dict = {}
for g in range(N_GPeT1):
    g_mask = (gpe_window_data[:,0] == g)
    gpe_spike_dict[g] = gpe_window_data[g_mask, 1].tolist()

stn_gpe_syn_events = {}

for stn_idx, neuron in enumerate(neurons):
    inh_indices = [
        i for i, s in enumerate(neuron.syn_info)
        if s['type'] == 'inh'
    ]

    if len(inh_indices) < N_INH_STN:
        raise RuntimeError(
            f"STN neuron {stn_idx} has only {len(inh_indices)} inhibitory synapses"
        )

    chosen_stn_syns = np.random.choice(
        inh_indices,
        size=N_INH_STN,
        replace=False
    )

    chosen_gpe_neurons = np.random.choice(
        N_GPeT1,
        size=N_INH_STN,
        replace=False
    )

    # inh syn mapping
    for gpe_id, stn_syn_idx in zip(chosen_gpe_neurons, chosen_stn_syns):
        spikes = gpe_spike_dict.get(int(gpe_id), [])
        if len(spikes) == 0:
            continue
        stn_gpe_syn_events[(stn_idx, stn_syn_idx)] = spikes.copy()

stn_gpe_syn_events_strkey = {
    f"{k[0]},{k[1]}": v for k, v in stn_gpe_syn_events.items()
}

# STN
stn_data = np.loadtxt("STN_spike_times_normal.txt", skiprows=2) 
mask_window = (stn_data[:,1] >= 2000) & (stn_data[:,1] <= 5000)
stn_window_data = stn_data[mask_window]

N_STN_actual = len(neurons)
stn_spike_dict = {}
for n_idx in range(N_STN_actual):
    mask = (stn_window_data[:,0] == n_idx)
    spikes = stn_window_data[mask, 1].tolist()
    if spikes:
        stn_spike_dict[n_idx] = spikes

# 4 electrode
dt = 0.1
lfp_time = np.arange(2000, 5000+dt, dt)

xe0 = AP * 0.55
ye0 = ML * 0.68
ze0 = DV * 0.21

contact_spacing_um = 500  # μm

electrode_contacts = [
    np.array([xe0, ye0, ze0 + i * contact_spacing_um])
    for i in range(4)
]

n_contacts = len(electrode_contacts)

# Ke parameter
va = 0.2  # mm/ms (conduction velocity)
lambda_mm = 0.2  # mm (space constant)
amp_e = 0.48  # μV (excitatory amplitude)
gw_AMPA = 0.25 * 1.2 
gw_NMDA = 0.00625 * 1.2
total_gw = gw_AMPA + gw_NMDA

ratio_AMPA = gw_AMPA / total_gw
ratio_NMDA = gw_NMDA / total_gw

syn_params = {
    'AMPA': {'sigma': 1.1774, 'delay': 2.5, 'amplitude_ratio': ratio_AMPA},
    'NMDA': {'sigma': 5.096, 'delay': 2.5, 'amplitude_ratio': ratio_NMDA}
}

@njit
def kernel(lfp_time, spike_times, delay, sigma, amp):
    n_time = lfp_time.size
    n_spikes = spike_times.size
    out = np.zeros(n_time)
    
    for i in range(n_spikes):
        t0 = spike_times[i] + delay
        for j in range(n_time):
            out[j] += amp * np.exp(-((lfp_time[j] - t0)**2) / (2.0*sigma**2))
    
    return out

def compute_ke_contact(contact_pos, stn_syn_events_strkey, 
                       syn_index_map_strkey, lfp_time, syn_params):
    lfp_ke_contact = np.zeros_like(lfp_time)
    
    for key_str, spike_list in stn_syn_events_strkey.items():
        info = syn_index_map_strkey.get(key_str)
        if info is None or info['type'] != 'exc':
            continue

        syn_pos = np.array(info['pos_um'])
        dist_mm = np.linalg.norm(syn_pos - contact_pos) / 1000.0
        amp_spatial = np.exp(-dist_mm / lambda_mm)

        # AMPA + NMDA
        for rec in ('AMPA', 'NMDA'):
            rec_sigma = syn_params[rec]['sigma']
            rec_delay = syn_params[rec]['delay']
            rec_ampratio = syn_params[rec]['amplitude_ratio']
            
            delay_total = rec_delay + dist_mm / va
            A0_channel = amp_e * rec_ampratio
            amp_rec = A0_channel * amp_spatial
            
            lfp_ke_contact += kernel(
                lfp_time, 
                np.array(spike_list), 
                delay_total, 
                rec_sigma, 
                amp_rec
            )

    return lfp_ke_contact

# Ki parameter
sigma_GABA = 3.397
delay_GABA = 1.0
amp_i = 3

def compute_ki_contact(contact_pos, stn_gpe_syn_events_strkey, 
                       syn_index_map_strkey, lfp_time):
    lfp_ki_contact = np.zeros_like(lfp_time)

    for key_str, spike_list in stn_gpe_syn_events_strkey.items():
        info = syn_index_map_strkey.get(key_str)
        if info is None or info.get('type', '') != 'inh':
            continue

        syn_pos = np.array(info['pos_um'])
        dist_mm = np.linalg.norm(syn_pos - contact_pos) / 1000.0
        amp_spatial = np.exp(-dist_mm / lambda_mm)

        delay_total = delay_GABA + dist_mm / va
        amp_rec = amp_i * amp_spatial

        lfp_ki_contact += kernel(
            lfp_time, 
            np.array(spike_list), 
            delay_total, 
            sigma_GABA, 
            amp_rec
        )

    return lfp_ki_contact

# K_stn parameter
amp_stn = 3.0
sigma_stn = 2.0
delay_stn = 0.0

def compute_kstn_contact(contact_pos, neurons, stn_spike_dict, lfp_time):
    lfp_kstn_contact = np.zeros_like(lfp_time)

    for n_idx, neuron in enumerate(neurons):
        spk_times = np.array(stn_spike_dict.get(n_idx, []))
        if spk_times.size == 0:
            continue

        # soma position
        syn_pos = neuron.position
        dist_mm = np.linalg.norm(syn_pos - contact_pos) / 1000.0
        amp_spatial = np.exp(-dist_mm / lambda_mm)

        amp_rec = amp_stn * amp_spatial
        delay_total = delay_stn + dist_mm / va

        lfp_kstn_contact += kernel(
            lfp_time, 
            spk_times, 
            delay_total, 
            sigma_stn, 
            amp_rec
        )

    return lfp_kstn_contact

with ThreadPoolExecutor() as executor:
    lfp_ke_total = list(executor.map(
        lambda c_idx: compute_ke_contact(
            electrode_contacts[c_idx], 
            stn_syn_events_strkey, 
            syn_index_map_strkey, 
            lfp_time, 
            syn_params
        ),
        range(n_contacts)
    ))

with ThreadPoolExecutor() as executor:
    lfp_ki_total = list(executor.map(
        lambda c_idx: compute_ki_contact(
            electrode_contacts[c_idx], 
            stn_gpe_syn_events_strkey, 
            syn_index_map_strkey, 
            lfp_time
        ),
        range(n_contacts)
    ))

with ThreadPoolExecutor() as executor:
    lfp_stn_intrinsic = list(executor.map(
        lambda c_idx: compute_kstn_contact(
            electrode_contacts[c_idx], 
            neurons, 
            stn_spike_dict, 
            lfp_time
        ),
        range(n_contacts)
    ))

# total LFP
lfp_total = [
    lfp_ke_total[c_idx] + lfp_ki_total[c_idx] + lfp_stn_intrinsic[c_idx]
    for c_idx in range(n_contacts)
]
