from testfile_1021 import RadialNeuron
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from concurrent.futures import ThreadPoolExecutor
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
    return ((x-cx)/a)**2 + ((y-cy)/b)**2 + ((z-cz)/c)**2 <= 1

x = np.arange(0, AP, distance)
y = np.arange(0, ML, distance)
z = np.arange(0, DV, distance)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
candidate_positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
candidate_positions = np.array([p for p in candidate_positions if in_ellipsoid(p)])
np.random.shuffle(candidate_positions)

final_positions = []
for i, p in enumerate(candidate_positions):
    if len(final_positions) >= N_STN:
        break
    noise = np.random.uniform(-noise_range, noise_range, 3) if i < N_STN//2 else np.zeros(3)
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
print(f"\nTarget STN neuron: {TARGET_STN}\n")

dt = 0.1
lfp_time = np.arange(2000, 5000+dt, dt)

# CTX→STN (single STN LFP)

data = np.loadtxt("Cortex_spike_times_normal.txt", skiprows=2)
N_CTX = 388
cortex_spike_times = {}
for c in range(N_CTX):
    mask = (data[:,0] == c) & (data[:,1] >= 2000) & (data[:,1] <= 5000)
    cortex_spike_times[c] = data[mask, 1].tolist()

stn_syn_events = {}
N_EXC_STN = 100
for stn_idx, neuron in enumerate(neurons):
    exc_idx = [i for i, s in enumerate(neuron.syn_info) if s['type'] == 'exc']
    chosen_stn = np.random.choice(exc_idx, size=N_EXC_STN, replace=False)
    chosen_ctx = np.random.choice(N_CTX, size=N_EXC_STN, replace=False)
    for ctx_id, stn_syn_idx in zip(chosen_ctx, chosen_stn):
        spikes = cortex_spike_times.get(ctx_id, [])
        if spikes:
            stn_syn_events[(stn_idx, stn_syn_idx)] = spikes.copy()

stn_syn_events_strkey = {f"{k[0]},{k[1]}": v for k, v in stn_syn_events.items()}

syn_index_map = {}
for n_idx, neuron in enumerate(neurons):
    base_pos = np.array(neuron.position)
    for s_idx, info in enumerate(neuron.syn_info):
        sec = info.get('section', None)
        typ = info.get('type', 'exc')
        pos_local = np.array([0.0, 0.0, 0.0])
        if sec is neuron.soma:
            theta, phi = info.get('theta'), info.get('phi')
            if theta is not None and phi is not None:
                r = neuron.soma_radius
                pos_local = np.array([r*np.sin(phi)*np.cos(theta),
                                      r*np.sin(phi)*np.sin(theta),
                                      r*np.cos(phi)])
        elif sec in neuron.dendrites:
            idx = neuron.dendrites.index(sec)
            p0, p1 = np.array(neuron.dend_coords[idx][0]), np.array(neuron.dend_coords[idx][1])
            x_rel = info.get('x', 0.5)
            try: x_rel = float(x_rel)
            except: x_rel = 0.5
            pos_local = p0 + x_rel*(p1-p0)
        elif sec in neuron.branches:
            idx = neuron.branches.index(sec)
            p0, p1 = np.array(neuron.branch_coords[idx][0]), np.array(neuron.branch_coords[idx][1])
            x_rel = info.get('x', 0.5)
            try: x_rel = float(x_rel)
            except: x_rel = 0.5
            pos_local = p0 + x_rel*(p1-p0)
        else:
            pos_local = np.array(info['pos']) if 'pos' in info else np.array([0.,0.,0.])
        syn_index_map[(int(n_idx), int(s_idx))] = {
            'pos_um': (base_pos + pos_local).tolist(),
            'type': str(typ)
        }

syn_index_map_strkey = {f"{k[0]},{k[1]}": v for k, v in syn_index_map.items()}

# Gpe→STN (single STN LFP)

N_GPeT1 = 988
N_INH_STN = 30
gpe_data = np.loadtxt("GPeT1_spike_times_normal.txt", skiprows=2)
gpe_window = gpe_data[(gpe_data[:,1] >= lfp_time[0]) & (gpe_data[:,1] <= lfp_time[-1])]
gpe_spike_dict = {}
for g in range(N_GPeT1):
    gpe_spike_dict[g] = gpe_window[gpe_window[:,0]==g, 1].tolist()

stn_gpe_syn_events = {}
for stn_idx, neuron in enumerate(neurons):
    inh_idx = [i for i, s in enumerate(neuron.syn_info) if s['type'] == 'inh']
    chosen_stn = np.random.choice(inh_idx, size=N_INH_STN, replace=False)
    chosen_gpe = np.random.choice(N_GPeT1, size=N_INH_STN, replace=False)
    for gpe_id, stn_syn_idx in zip(chosen_gpe, chosen_stn):
        spikes = gpe_spike_dict.get(int(gpe_id), [])
        if spikes:
            stn_gpe_syn_events[(stn_idx, stn_syn_idx)] = spikes.copy()

stn_gpe_syn_events_strkey = {f"{k[0]},{k[1]}": v for k, v in stn_gpe_syn_events.items()}

# STN spike times

stn_data = np.loadtxt("STN_spike_times_normal.txt", skiprows=2)
stn_window = stn_data[(stn_data[:,1] >= lfp_time[0]) & (stn_data[:,1] <= lfp_time[-1])]
stn_spike_dict = {}
for n_idx in range(len(neurons)):
    spikes = stn_window[stn_window[:,0]==n_idx, 1].tolist()
    if spikes:
        stn_spike_dict[n_idx] = spikes

# 4 electrode_contacts

xe0 = AP * 0.55
ye0 = ML * 0.68
ze0 = DV * 0.21
contact_spacing_um = 500.0
electrode_contacts = [
    np.array([xe0, ye0, ze0 + i * contact_spacing_um])
    for i in range(4)
]
n_contacts = len(electrode_contacts)

# kernel_parameter

va        = 0.2
lambda_mm = 0.2
amp_e     = 0.48
gw_AMPA   = 0.25 * 1.2
gw_NMDA   = 0.00625 * 1.2
total_gw  = gw_AMPA + gw_NMDA
ratio_AMPA = gw_AMPA / total_gw
ratio_NMDA = gw_NMDA / total_gw

syn_params = {
    'AMPA': {'sigma': 1.1774, 'delay': 2.5, 'amplitude_ratio': ratio_AMPA},
    'NMDA': {'sigma': 5.096,  'delay': 2.5, 'amplitude_ratio': ratio_NMDA}
}
sigma_GABA = 3.397
delay_GABA = 1.0
amp_i      = 3.0
amp_stn    = 3.0
sigma_stn  = 2.0
delay_stn  = 0.0

@njit
def kernel(lfp_time, spike_times, delay, sigma, amp):
    n_time   = lfp_time.size
    n_spikes = spike_times.size
    out = np.zeros(n_time)
    for i in range(n_spikes):
        t0 = spike_times[i] + delay
        for j in range(n_time):
            out[j] += amp * np.exp(-((lfp_time[j] - t0)**2) / (2.0 * sigma**2))
    return out


# Task 1 : synaptic LFP

def compute_lfp_per_synapse(target_idx, contact_pos):
    lfp_per_syn = {}

    # exc (AMPA + NMDA)
    for key_str, spike_list in stn_syn_events_strkey.items():
        stn_idx = int(key_str.split(',')[0])
        syn_idx = int(key_str.split(',')[1])
        if stn_idx != target_idx:
            continue
        info = syn_index_map_strkey.get(key_str)
        if info is None or info['type'] != 'exc':
            continue
        syn_pos    = np.array(info['pos_um'])
        dist_mm    = np.linalg.norm(syn_pos - contact_pos) / 1000.0
        amp_spatial = np.exp(-dist_mm / lambda_mm)
        lfp_syn = np.zeros_like(lfp_time)
        for rec in ('AMPA', 'NMDA'):
            delay_total = syn_params[rec]['delay'] + dist_mm / va
            amp_rec     = amp_e * syn_params[rec]['amplitude_ratio'] * amp_spatial
            lfp_syn    += kernel(lfp_time, np.array(spike_list),
                                 delay_total, syn_params[rec]['sigma'], amp_rec)
        lfp_per_syn[syn_idx] = lfp_syn

    # inh (GABA)
    for key_str, spike_list in stn_gpe_syn_events_strkey.items():
        stn_idx = int(key_str.split(',')[0])
        syn_idx = int(key_str.split(',')[1])
        if stn_idx != target_idx:
            continue
        info = syn_index_map_strkey.get(key_str)
        if info is None or info['type'] != 'inh':
            continue
        syn_pos     = np.array(info['pos_um'])
        dist_mm     = np.linalg.norm(syn_pos - contact_pos) / 1000.0
        amp_spatial = np.exp(-dist_mm / lambda_mm)
        delay_total = delay_GABA + dist_mm / va
        amp_rec     = amp_i * amp_spatial
        lfp_per_syn[syn_idx] = kernel(lfp_time, np.array(spike_list),
                                      delay_total, sigma_GABA, amp_rec)
    return lfp_per_syn

CONTACT_IDX  = 0
contact_pos0 = electrode_contacts[CONTACT_IDX]

# same synapses (I_syn task): 0(inh), 6(exc), 55(exc), 105(exc), 129(inh)
chosen_5  = [0, 6, 55, 105, 129]
syn_types = {0: 'inh', 6: 'exc', 55: 'exc', 105: 'exc', 129: 'inh'}

fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
fig.suptitle(f'[LFP Task 1] STN Neuron {TARGET_STN}: LFP contribution per Synapse\n'
             f'Contact {CONTACT_IDX} | Red=exc, Blue=inh',
             fontsize=13, fontweight='bold')

for ax, syn_idx in zip(axes, chosen_5):
    stype = syn_types[syn_idx]
    color = 'tomato' if stype == 'exc' else 'steelblue'
    lfp_sig = lfp_per_syn.get(syn_idx, np.zeros_like(lfp_time))
    ax.plot(lfp_time, lfp_sig, color=color, lw=0.8,
            label=f'Syn {syn_idx} ({stype})')
    ax.axhline(y=0, color='gray', lw=0.5, ls='--', alpha=0.5)
    ax.set_ylabel('LFP (uV)', fontsize=9)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (ms)', fontsize=12)
plt.tight_layout()
plt.savefig('single LFP_task1_5synapses.png', dpi=150, bbox_inches='tight')
plt.show()

# Task 2 : total LFP (contact 0~3)

LFP_total = np.zeros_like(lfp_time)
for lfp_sig in lfp_per_syn.values():
    LFP_total += lfp_sig

spk_times_target = np.array(stn_spike_dict.get(TARGET_STN, []))
if spk_times_target.size > 0:
    syn_pos_stn  = np.array(neurons[TARGET_STN].position)
    dist_mm_stn  = np.linalg.norm(syn_pos_stn - contact_pos0) / 1000.0
    amp_spat_stn = np.exp(-dist_mm_stn / lambda_mm)
    delay_stn_total = delay_stn + dist_mm_stn / va
    LFP_total += kernel(lfp_time, spk_times_target,
                        delay_stn_total, sigma_stn, amp_stn * amp_spat_stn)

colors_t2 = ['darkgreen', 'royalblue', 'darkorange', 'purple']
fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
fig.suptitle(f'[LFP Task 2] STN Neuron {TARGET_STN}: LFP_total (Contact 0~3)',
             fontsize=13, fontweight='bold')

lfp_totals_by_contact = {}

for c_idx in range(n_contacts):
    cp = electrode_contacts[c_idx]
    lp = compute_lfp_per_synapse(TARGET_STN, cp)
    LFP_c = np.zeros_like(lfp_time)
    for sig in lp.values():
        LFP_c += sig
    if spk_times_target.size > 0:
        dist_c  = np.linalg.norm(np.array(neurons[TARGET_STN].position) - cp) / 1000.0
        amp_c   = amp_stn * np.exp(-dist_c / lambda_mm)
        delay_c = delay_stn + dist_c / va
        LFP_c  += kernel(lfp_time, spk_times_target, delay_c, sigma_stn, amp_c)
    lfp_totals_by_contact[c_idx] = LFP_c

    dist = np.linalg.norm(np.array(neurons[TARGET_STN].position) - cp)
    axes[c_idx].plot(lfp_time, LFP_c, color=colors_t2[c_idx], lw=0.8)
    axes[c_idx].set_ylabel('LFP_total (µV)', fontsize=10)
    axes[c_idx].set_title(f'Contact {c_idx} | dist={dist:.0f} µm', fontsize=10)
    axes[c_idx].grid(True, alpha=0.3)
    print(f"  Contact {c_idx}: mean={LFP_c.mean():.4f} µV  std={LFP_c.std():.4f} µV")

axes[-1].set_xlabel('Time (ms)', fontsize=12)
plt.tight_layout()
plt.savefig('single LFP_task2_total.png', dpi=150, bbox_inches='tight')
plt.show()

LFP_total = lfp_totals_by_contact[0]

TAU_FIRING = 10.0

def spike_to_continuous(spike_times, time_axis, tau_decay):
    continuous = np.zeros_like(time_axis)
    for t_sp in spike_times:
        delta = time_axis - t_sp
        mask  = delta >= 0
        continuous[mask] += np.exp(-delta[mask] / tau_decay)
    return continuous

lag_range_ms = 30.0
lags_ms      = np.arange(-lag_range_ms, lag_range_ms + dt, dt)
lag_steps    = (lags_ms / dt).astype(int)

def compute_lag_correlation(sig1, sig2, lag_steps):
    n = len(sig1)
    if np.std(sig1) == 0 or np.std(sig2) == 0:
        return np.zeros(len(lag_steps))
    z1 = (sig1 - sig1.mean()) / sig1.std()
    z2 = (sig2 - sig2.mean()) / sig2.std()
    corr_curve = np.zeros(len(lag_steps))
    for k, shift in enumerate(lag_steps):
        if shift >= 0:
            s1 = z1[:n-shift] if shift > 0 else z1
            s2 = z2[shift:]   if shift > 0 else z2
        else:
            s  = -shift
            s1 = z1[s:]
            s2 = z2[:n-s]
        if len(s1) == 0:
            continue
        corr_curve[k] = np.dot(s1, s2) / len(s1)
    return corr_curve

firing_cont = spike_to_continuous(spk_times_target, lfp_time, TAU_FIRING)
corr_curve  = compute_lag_correlation(LFP_total, firing_cont, lag_steps)

peak_idx  = np.argmax(np.abs(corr_curve))
peak_lag  = lags_ms[peak_idx]
peak_corr = corr_curve[peak_idx]

print(f"  peak lag τ  = {peak_lag:+.1f} ms")
print(f"  peak corr   = {peak_corr:+.4f}")
print(f"  lag=0 corr  = {corr_curve[len(lags_ms)//2]:+.4f}")

# Task 3-1 : firing_single LFP_correlation

colors_contact = ['steelblue', 'tomato', 'green', 'purple']

fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
fig.suptitle(f'[LFP Task 3] <LFP(t) · firing(t+τ)>  STN Neuron {TARGET_STN}\n'
             f'Contact 0~3',
             fontsize=13, fontweight='bold')

axes_flat = axes.flatten()
corr_curves_all = {}

for c_idx in range(n_contacts):
    cp = electrode_contacts[c_idx]
    lp = compute_lfp_per_synapse(TARGET_STN, cp)
    LFP_c = np.zeros_like(lfp_time)
    for sig in lp.values():
        LFP_c += sig
    if spk_times_target.size > 0:
        dist_c  = np.linalg.norm(np.array(neurons[TARGET_STN].position) - cp) / 1000.0
        amp_c   = amp_stn * np.exp(-dist_c / lambda_mm)
        delay_c = delay_stn + dist_c / va
        LFP_c  += kernel(lfp_time, spk_times_target, delay_c, sigma_stn, amp_c)
    cc = compute_lag_correlation(LFP_c, firing_cont, lag_steps)
    corr_curves_all[c_idx] = (cc, LFP_c)
    pi     = np.argmax(np.abs(cc))
    p_lag  = lags_ms[pi]
    p_corr = cc[pi]
    dist   = np.linalg.norm(np.array(neurons[TARGET_STN].position) - cp)
    ax = axes_flat[c_idx]
    ax.plot(lags_ms, cc, color=colors_contact[c_idx], lw=1.5)
    ax.axvline(x=0,     color='gray', lw=1.0, ls='--', alpha=0.5, label='τ=0')
    ax.axvline(x=p_lag, color='red',  lw=1.2, ls='--', alpha=0.8,
               label=f'peak τ={p_lag:+.1f}ms\nr={p_corr:+.4f}')
    ax.axhline(y=0, color='gray', lw=0.5, ls=':', alpha=0.4)
    ax.set_title(f'Contact {c_idx} | dist={dist:.0f}um', fontsize=10)
    ax.set_xlabel('lag τ (ms)', fontsize=10)
    ax.set_ylabel('Correlation', fontsize=10)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('single LFP_task3_lagcurve.png', dpi=150, bbox_inches='tight')
plt.show()

# Task 3-2 : firing_single LFP_correlation_3D

all_positions = np.array([n.position for n in neurons])
target_pos    = np.array(neurons[TARGET_STN].position)

peak_corrs_per_contact = []
peak_lags_per_contact  = []

for c_idx in range(n_contacts):
    cp = electrode_contacts[c_idx]

    lp = compute_lfp_per_synapse(TARGET_STN, cp)
    LFP_c = np.zeros_like(lfp_time)
    for sig in lp.values():
        LFP_c += sig
    if spk_times_target.size > 0:
        dist_c  = np.linalg.norm(np.array(neurons[TARGET_STN].position) - cp) / 1000.0
        amp_c   = amp_stn * np.exp(-dist_c / lambda_mm)
        delay_c = delay_stn + dist_c / va
        LFP_c  += kernel(lfp_time, spk_times_target, delay_c, sigma_stn, amp_c)

    # lag correlation
    cc = compute_lag_correlation(LFP_c, firing_cont, lag_steps)
    pi = np.argmax(np.abs(cc))
    peak_corrs_per_contact.append(cc[pi])
    peak_lags_per_contact.append(lags_ms[pi])
    print(f"  Contact {c_idx}: peak_lag={lags_ms[pi]:+.1f}ms  peak_corr={cc[pi]:+.4f}")

# 3D plot
limit = max(0.3, max(abs(v) for v in peak_corrs_per_contact) * 1.1)

fig = plt.figure(figsize=(14, 10))
fig.suptitle(f'[LFP Task 3] STN Neuron {TARGET_STN}: LFP ↔ Firing Lag Correlation\n'
             f'color = peak_corr (with delay)',
             fontsize=13, fontweight='bold')

for c_idx in range(n_contacts):
    ax3d = fig.add_subplot(2, 2, c_idx+1, projection='3d')
    cp   = electrode_contacts[c_idx]
    corr_val = peak_corrs_per_contact[c_idx]
    lag_val  = peak_lags_per_contact[c_idx]
    dist     = np.linalg.norm(target_pos - cp)

    ax3d.scatter(all_positions[:,0], all_positions[:,1], all_positions[:,2],
                 c='lightgray', s=16, alpha=0.35)

    sc = ax3d.scatter(target_pos[0], target_pos[1], target_pos[2],
                      c=[corr_val], cmap='coolwarm',
                      s=200, edgecolor='black', linewidth=1.0,
                      vmin=-limit, vmax=limit, zorder=5)

    ax3d.scatter(cp[0], cp[1], cp[2],
                 c='darkgreen', s=260, marker='^',
                 edgecolor='cyan', linewidths=1.3, zorder=5)

    ax3d.plot([cp[0], target_pos[0]],
              [cp[1], target_pos[1]],
              [cp[2], target_pos[2]],
              linestyle='--', linewidth=1.4, color='gray', alpha=0.7)

    ax3d.set_box_aspect([1, 1, 1])
    ax3d.set_xlabel('AP (um)', fontsize=9)
    ax3d.set_ylabel('ML (um)', fontsize=9)
    ax3d.set_zlabel('DV (um)', fontsize=9)
    ax3d.set_zlim(0, 1750)
    ax3d.set_title(f'Contact {c_idx}\n'
                   f'peak_corr={corr_val:+.4f} | peak_lag={lag_val:+.1f}ms\n'
                   f'dist={dist:.1f}um', fontsize=9)

    cbar = fig.colorbar(sc, ax=ax3d, shrink=0.55, pad=0.05)
    cbar.set_label('peak_corr (lag)', fontsize=8)

plt.tight_layout()
plt.savefig('single LFP_task3_corr 3D.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*55)
print(f" LFP Task 3 Summary: STN Neuron {TARGET_STN}")
print("="*55)
print(f"{'Contact':>8} | {'peak_lag(ms)':>13} | {'peak_corr':>10} | {'dist(um)':>10}")
print("-"*55)
for c_idx in range(n_contacts):
    cp   = electrode_contacts[c_idx]
    dist = np.linalg.norm(target_pos - cp)
    print(f"{c_idx:>8} | {peak_lags_per_contact[c_idx]:>+13.1f} | "
          f"{peak_corrs_per_contact[c_idx]:>+10.4f} | {dist:>10.1f}")
