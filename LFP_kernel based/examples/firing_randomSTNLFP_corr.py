from testfile_1021 import RadialNeuron 
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
from scipy.spatial import distance_matrix
import random as pyrandom
import json 
from scipy.ndimage import gaussian_filter1d
from scipy import signal
from numba import njit
from concurrent.futures import ThreadPoolExecutor

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
# random STN target
TARGET_STN = 118
print(f"\nSelected STN neuron index: {TARGET_STN}\n")

# Cortex→STN : exc syn 100

data = np.loadtxt("Cortex_spike_times_normal.txt", skiprows=2)
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

    # exc mapping
    for ctx_id, stn_syn_idx in zip(chosen_ctx_neurons, chosen_stn_syns):
        spikes = cortex_spike_times.get(ctx_id, [])
        if len(spikes) == 0:
            continue
        stn_syn_events[(stn_idx, stn_syn_idx)] = spikes.copy()

stn_syn_events_strkey = {
    f"{k[0]},{k[1]}": v for k, v in stn_syn_events.items()
}

print("CTX → STN mapping")
for i, (k, v) in enumerate(stn_syn_events_strkey.items()):
    print(f"{k}: {v[:5]} ...")
    if i >= 5:
        break

syn_index_map = {}  # (stn_neuron_idx, stn_syn_idx) -> {'pos_um': [x,y,z], 'type': 'exc'/'inh'}
for n_idx, neuron in enumerate(neurons):
    base_pos = np.array(neuron.position)

    for s_idx, info in enumerate(neuron.syn_info):
        sec = info.get('section', None)
        typ = info.get('type', 'exc')

        pos_local = np.array([0.0, 0.0, 0.0])

        # 1) soma-located synapse
        if sec is neuron.soma:
            theta = info.get('theta', None)
            phi = info.get('phi', None)
            if theta is not None and phi is not None:
                r = neuron.soma_radius
                pos_local = np.array([r * np.sin(phi) * np.cos(theta),
                                      r * np.sin(phi) * np.sin(theta),
                                      r * np.cos(phi)])
            else:
                pos_local = np.array([0.0, 0.0, 0.0])

        # 2) dendrite-located synapse
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
            p0 = np.array(p0); p1 = np.array(p1)
            pos_local = p0 + x_rel * (p1 - p0)

        # 3) branch-located synapse
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
            p0 = np.array(p0); p1 = np.array(p1)
            pos_local = p0 + x_rel * (p1 - p0)

        else:
            if 'pos' in info:
                pos_local = np.array(info['pos'])
            else:
                pos_local = np.array([0.0, 0.0, 0.0])

        # synapse_absolute_coordinates
        pos_um = base_pos + pos_local

        syn_index_map[(int(n_idx), int(s_idx))] = {
            'pos_um': pos_um.tolist(),
            'type': str(typ)
        }

syn_index_map_strkey = {f"{k[0]},{k[1]}": v for k, v in syn_index_map.items()}
with open("stn_synapses_13000.json", "w") as f:
    json.dump(syn_index_map_strkey, f, indent=2)

# Ke
dt = 0.1
lfp_time = np.arange(2000, 5000+dt, dt)
xe0 = AP * 0.55
ye0 = ML * 0.68   # lateral
ze0 = DV * 0.21   # dorsal

contact_spacing_mm = 0.5  # mm
contact_spacing_um = contact_spacing_mm * 1000  # μm

electrode_contacts = [
    np.array([xe0, ye0, ze0 + i * contact_spacing_um])
    for i in range(4)
] # contact 0 : Ventrolateral, contact 1,2 : Dorsolateral, contact 3: outside
n_contacts = len(electrode_contacts)

lfp_ke_AMPA = [np.zeros_like(lfp_time) for _ in range(n_contacts)]
lfp_ke_NMDA = [np.zeros_like(lfp_time) for _ in range(n_contacts)]
lfp_ke_total = [np.zeros_like(lfp_time) for _ in range(n_contacts)]

va = 0.2 # (mm/ms)
lambda_mm = 0.2 # (mm) 
amp_e = 0.48 # (μV)
gw_AMPA = 0.25 * 1.2 
gw_NMDA = 0.00625 * 1.2
total_gw = gw_AMPA + gw_NMDA

ratio_AMPA = gw_AMPA / total_gw
ratio_NMDA = gw_NMDA / total_gw
syn_params = {
    'AMPA': {'sigma': 1.1774,   'delay': 2.5,  'amplitude_ratio': ratio_AMPA, 'p': 1.0},
    'NMDA': {'sigma': 5.096,  'delay': 2.5,  'amplitude_ratio': ratio_NMDA, 'p': 1.0}
}
lambda_mm = float(lambda_mm)
va = float(va)

@njit
def kernel(lfp_time, spike_times, delay, sigma, amp):
    n_time = lfp_time.size
    n_spikes = spike_times.size
    out = np.zeros(n_time)
    for i in range(n_spikes):
        t0 = spike_times[i] + delay
        for j in range(n_time):
            out[j] += amp * np.exp(-((lfp_time[j] - t0)**2)/(2.0*sigma**2))
    return out

def compute_ke_contact(contact_pos, stn_syn_events_strkey, syn_index_map_strkey, lfp_time, syn_params):
    lfp_ke_contact = np.zeros_like(lfp_time)
    
    for key_str, spike_list in stn_syn_events_strkey.items():
        info = syn_index_map_strkey.get(key_str)
        if info is None or info['type'] != 'exc':
            continue
        stn_idx = int(key_str.split(',')[0])
        if stn_idx != TARGET_STN:
            continue
        syn_pos = np.array(info['pos_um'])
        dist_mm = np.linalg.norm(syn_pos - contact_pos) / 1000.0
        amp_spatial = np.exp(-dist_mm / lambda_mm)

        for rec in ('AMPA', 'NMDA'):
            rec_sigma = syn_params[rec]['sigma']
            rec_delay = syn_params[rec]['delay']
            rec_ampratio = syn_params[rec]['amplitude_ratio']
            delay_total = rec_delay + dist_mm / va
            A0_channel = amp_e * rec_ampratio
            amp_rec = A0_channel * amp_spatial

            lfp_ke_contact += kernel(lfp_time, np.array(spike_list), delay_total, rec_sigma, amp_rec)

    return lfp_ke_contact

with ThreadPoolExecutor() as executor:
    lfp_ke_total = list(executor.map(
        lambda c_idx: compute_ke_contact(electrode_contacts[c_idx], stn_syn_events_strkey, syn_index_map_strkey, lfp_time, syn_params),
        range(n_contacts)
    ))
  
# GPe→STN : inh syn 30

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

    # inh mapping
    for gpe_id, stn_syn_idx in zip(chosen_gpe_neurons, chosen_stn_syns):
        spikes = gpe_spike_dict.get(int(gpe_id), [])
        if len(spikes) == 0:
            continue
        stn_gpe_syn_events[(stn_idx, stn_syn_idx)] = spikes.copy()

stn_gpe_syn_events_strkey = {
    f"{k[0]},{k[1]}": v for k, v in stn_gpe_syn_events.items()
}

print("GPe→STN mapping")
for i, (k, v) in enumerate(stn_gpe_syn_events_strkey.items()):
    print(f"{k}: {v[:5]} ... (n={len(v)})")
    if i >= 5:
        break

# Ki
sigma_GABA = 3.397
delay_GABA = 1.0
gw_GABA = 1
amp_i = 3 # demo_lfp_kernel (soma layer)

lfp_ki_total = [np.zeros_like(lfp_time) for _ in range(n_contacts)]
def compute_ki_contact(contact_pos, stn_gpe_syn_events_strkey, syn_index_map_strkey, lfp_time):
    lfp_ki_contact = np.zeros_like(lfp_time)

    for key_str, spike_list in stn_gpe_syn_events_strkey.items():
        info = syn_index_map_strkey.get(key_str)
        if info is None:
            continue
        if info.get('type', '') != 'inh':
            continue
        stn_idx = int(key_str.split(',')[0])
        if stn_idx != TARGET_STN:
            continue

        syn_pos = np.array(info['pos_um'])
        dist_mm = np.linalg.norm(syn_pos - contact_pos) / 1000.0
        amp_spatial = np.exp(-dist_mm / lambda_mm)

        delay_total = delay_GABA + dist_mm / va
        amp_rec = amp_i * gw_GABA * amp_spatial

        lfp_ki_contact += kernel(lfp_time, np.array(spike_list), delay_total, sigma_GABA, amp_rec)

    return lfp_ki_contact

with ThreadPoolExecutor() as executor:
    lfp_ki_total = list(executor.map(
        lambda c_idx: compute_ki_contact(electrode_contacts[c_idx], stn_gpe_syn_events_strkey, syn_index_map_strkey, lfp_time),
        range(n_contacts)
    ))
  
# K_STN

stn_data = np.loadtxt("STN_spike_times_normal.txt", skiprows=2) 
mask_window = (stn_data[:,1] >= lfp_time[0]) & (stn_data[:,1] <= lfp_time[-1])
stn_window_data = stn_data[mask_window]
    
N_STN_actual = len(neurons)
stn_spike_dict = {}
for n_idx in range(N_STN_actual):
    mask = (stn_window_data[:,0] == n_idx)
    spikes = stn_window_data[mask, 1].tolist()
    if spikes:
        stn_spike_dict[n_idx] = spikes

amp_stn = 3.0    # μV
sigma_stn = 2.0  # ms 
delay_stn = 0.0  # ms
lfp_stn_intrinsic = [np.zeros_like(lfp_time) for _ in range(n_contacts)]

def compute_kstn_contact(contact_pos, neurons, stn_spike_dict, lfp_time):
    lfp_kstn_contact = np.zeros_like(lfp_time)

    for n_idx, neuron in enumerate(neurons):
        if n_idx != TARGET_STN:
                continue
        spk_times = np.array(stn_spike_dict.get(n_idx, []))
        if spk_times.size == 0:
            continue

        syn_pos = neuron.position
        dist_mm = np.linalg.norm(syn_pos - contact_pos) / 1000.0
        amp_spatial = np.exp(-dist_mm / lambda_mm)

        amp_rec = amp_stn * amp_spatial
        delay_total = delay_stn + dist_mm / va

        lfp_kstn_contact += kernel(lfp_time, spk_times, delay_total, sigma_stn, amp_rec)

    return lfp_kstn_contact

with ThreadPoolExecutor() as executor:
    lfp_stn_intrinsic = list(executor.map(
        lambda c_idx: compute_kstn_contact(electrode_contacts[c_idx], neurons, stn_spike_dict, lfp_time),
        range(n_contacts)
    ))
lfp_total = [
    lfp_ke_total[c_idx] + lfp_ki_total[c_idx] + lfp_stn_intrinsic[c_idx]
    for c_idx in range(n_contacts)
]

fig, ax = plt.subplots(figsize=(12,5))
for c_idx in range(n_contacts):
    ax.plot(lfp_time, lfp_total[c_idx], label=f'Contact {c_idx}')
ax.set_xlabel('time (ms)', fontsize=14)
ax.set_ylabel('LFP (µV)', fontsize=14)
ax.tick_params(axis='both', labelsize=12)
ax.legend()
plt.show()

# LFP
for c in range(n_contacts):
    ke_rms = np.sqrt(np.mean(lfp_ke_total[c]**2))
    ki_rms = np.sqrt(np.mean(lfp_ki_total[c]**2))
    stn_rms = np.sqrt(np.mean(lfp_stn_intrinsic[c]**2))
    total_rms = np.sqrt(np.mean(lfp_total[c]**2))
    R = ke_rms / total_rms if total_rms > 0 else np.nan

    print(f"Contact{c}")
    print("  Ke RMS:", ke_rms)
    print("  Ki RMS:", ki_rms)
    print("  STN RMS:", stn_rms)
    print("  Total RMS:", total_rms)
    print("  Contribution(RMS_Ke/RMS_total):", R)

# LFP ↔ Spike Correlation (Contact 0~3)

contact0_corr_all = np.zeros(len(neurons))

for n_idx in range(len(neurons)):

    spk_times_n = np.array(stn_spike_dict.get(n_idx, []))

    spike_rate = np.zeros_like(lfp_time)
    for t in spk_times_n:
        idx = int((t - lfp_time[0]) / dt)
        if 0 <= idx < len(spike_rate):
            spike_rate[idx] += 1

    sigma_ms = 5
    sigma = sigma_ms / dt
    spike_rate = gaussian_filter1d(spike_rate, sigma)

    lfp_sig = lfp_total[0]

    b, a = signal.butter(3, [13, 30], btype='bandpass', fs=1000/dt)
    lfp_filtered = signal.filtfilt(b, a, lfp_sig)

    if np.std(lfp_filtered) == 0 or np.std(spike_rate) == 0:
        contact0_corr_all[n_idx] = 0
        continue

    lfp_z = (lfp_filtered - np.mean(lfp_filtered)) / np.std(lfp_filtered)
    rate_z = (spike_rate - np.mean(spike_rate)) / np.std(spike_rate)

    corr_full = signal.correlate(lfp_z, rate_z, mode='same') / len(lfp_z)

    center = len(corr_full) // 2
    window = int(10 / dt)
    search = corr_full[center-window:center+window]

    contact0_corr_all[n_idx] = search[np.argmax(np.abs(search))]

print("\n================ Correlation Analysis ================")
spk_times = np.array(stn_spike_dict.get(TARGET_STN, []))
sigma_ms = 10
sigma = sigma_ms / dt

spike_rate = np.zeros_like(lfp_time)

for t in spk_times:
    idx = int((t - lfp_time[0]) / dt)
    if 0 <= idx < len(spike_rate):
        spike_rate[idx] += 1

spike_rate = gaussian_filter1d(spike_rate, sigma)
corr_values = []

b, a = signal.butter(3, [13, 30], btype='bandpass', fs=1000/dt)
for c_idx in range(n_contacts):

    lfp_sig = lfp_total[c_idx]
    lfp_filtered = signal.filtfilt(b, a, lfp_sig)

    lfp_z = (lfp_filtered - np.mean(lfp_filtered)) / np.std(lfp_filtered)
    rate_z = (spike_rate - np.mean(spike_rate)) / np.std(spike_rate)

    corr_full = signal.correlate(lfp_z, rate_z, mode='same') / len(lfp_z)

    center = len(corr_full)//2
    window = int(10/dt)

    search = corr_full[center-window:center+window]
    r_peak = search[np.argmax(np.abs(search))]

    corr_values.append(r_peak)

    print(f"Contact {c_idx} : Correlation = {r_peak:.4f}")

all_positions = np.array([n.position for n in neurons])
target_pos = np.array(neurons[TARGET_STN].position)

# 3D corr plot
fig = plt.figure(figsize=(14,10))
for c_idx in range(n_contacts):

    ax_3d = fig.add_subplot(2, 2, c_idx+1, projection='3d')

    contact_pos = electrode_contacts[c_idx]
    corr_value = corr_values[c_idx]

    ax_3d.scatter(
        all_positions[:,0], all_positions[:,1], all_positions[:,2],
        c='lightgray', s=16, alpha=0.35
    )
    sc = ax_3d.scatter(
        target_pos[0], target_pos[1], target_pos[2],
        c=[corr_value], cmap='coolwarm',
        s=170, edgecolor='black', linewidth=0.9,
        vmin=-0.3, vmax=0.3
    )
    ax_3d.scatter(
        contact_pos[0], contact_pos[1], contact_pos[2],
        c='darkgreen', s=240, marker='^',
        edgecolor='cyan', linewidths=1.2
    )
    ax_3d.plot(
        [contact_pos[0], target_pos[0]],
        [contact_pos[1], target_pos[1]],
        [contact_pos[2], target_pos[2]],
        linestyle='--', linewidth=1.4
    )
    dist = np.linalg.norm(target_pos - contact_pos)
    print(f"Contact {c_idx} : Distance = {dist:.2f} μm")
    ax_3d.set_box_aspect([1,1,1])
    ax_3d.set_xlabel('AP (μm)', fontsize=10)
    ax_3d.set_ylabel('ML (μm)', fontsize=10)
    ax_3d.set_zlabel('DV (μm)', fontsize=10)
    ax_3d.set_zlim(0, 1750)
    ax_3d.set_title(f'Contact {c_idx}  |  Corr = {corr_value:.4f} |  Dist = {dist:.1f} μm', fontsize=11)

    cbar = fig.colorbar(sc, ax=ax_3d, shrink=0.6, pad=0.05)
    cbar.set_label("Correlation", fontsize=9)

plt.suptitle("Single STN : LFP–Spike Correlation (per Contact)", fontsize=14)
plt.tight_layout()
plt.show()

# BEST neuron (Contact0)
best_neuron = int(np.argmax(contact0_corr_all))
print(f"\n[Best Neuron] neuron_id = {best_neuron}, corr(contact 0) = {contact0_corr_all[best_neuron]:.4f}")

spk_times_best = np.array(stn_spike_dict.get(best_neuron, []))

spike_rate = np.zeros_like(lfp_time)
for t in spk_times_best:
    idx = int((t - lfp_time[0]) / dt)
    if 0 <= idx < len(spike_rate):
        spike_rate[idx] += 1

sigma_ms = 5
sigma = sigma_ms / dt
spike_rate = gaussian_filter1d(spike_rate, sigma)

all_positions = np.array([n.position for n in neurons])
target_pos = np.array(neurons[best_neuron].position)

fig = plt.figure(figsize=(12,10))

for contact_id in range(4):
    lfp_sig = lfp_total[contact_id]
    b, a = signal.butter(3, [13, 30], btype='bandpass', fs=1000/dt)
    lfp_filtered = signal.filtfilt(b, a, lfp_sig)

    lfp_z = (lfp_filtered - np.mean(lfp_filtered)) / np.std(lfp_filtered)
    rate_z = (spike_rate - np.mean(spike_rate)) / np.std(spike_rate)
    corr_full = signal.correlate(lfp_z, rate_z, mode='same') / len(lfp_z)

    center = len(corr_full)//2
    window = int(10/dt)
    search = corr_full[center-window:center+window]
    corr_value = search[np.argmax(np.abs(search))]

    contact_pos = electrode_contacts[contact_id]
    target_pos = np.array(neurons[best_neuron].position)
    dist = np.linalg.norm(target_pos - contact_pos)

    ax_3d = fig.add_subplot(2,2,contact_id+1, projection='3d')
    contact_pos = electrode_contacts[contact_id]

    ax_3d.scatter(
        all_positions[:,0], all_positions[:,1], all_positions[:,2],
        c='lightgray', s=16, alpha=0.35
    )
    limit = 0.3
    sc = ax_3d.scatter(
        target_pos[0], target_pos[1], target_pos[2],
        c=[corr_value], cmap='coolwarm',
        s=180, edgecolor='black', linewidth=1,
        vmin=-limit, vmax=limit
    )
    ax_3d.scatter(
        contact_pos[0], contact_pos[1], contact_pos[2],
        c='darkgreen', s=260, marker='^',
        edgecolor='cyan', linewidths=1.3
    )
    ax_3d.plot(
        [contact_pos[0], target_pos[0]],
        [contact_pos[1], target_pos[1]],
        [contact_pos[2], target_pos[2]],
        linestyle='--', linewidth=1.4
    )
    ax_3d.set_box_aspect([1,1,1])
    ax_3d.set_zlim(0, 1750)

    ax_3d.set_title(f'Contact {contact_id}\nCorr={corr_value:.3f} | Dist={dist:.1f} μm')
    cbar = fig.colorbar(sc, ax=ax_3d, shrink=0.6, pad=0.02)
    cbar.set_label("Correlation", fontsize=11)

plt.tight_layout()
plt.show()
