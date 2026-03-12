from testfile_1021 import RadialNeuron
from brian2 import *
import numpy as np
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

dt = 0.1
lfp_time = np.arange(2000, 5000+dt, dt)

# CTX→STN (I_syn_total)

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

# GPe→STN (I_syn_total)

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

# I_syn calculate

V_rest = -65.0
syn_type_params = {
    'exc': {'tau_rise': 0.5, 'tau_decay': 2.0, 'g_max': 1.0, 'E_rev':   0.0},
    'inh': {'tau_rise': 0.5, 'tau_decay': 6.0, 'g_max': 1.0, 'E_rev': -80.0}
}

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

def compute_isyn_for_neuron(target_idx):
    isyn_per_syn = {}
    for (stn_idx, syn_idx), spikes in stn_syn_events.items():
        if stn_idx != target_idx:
            continue
        p = syn_type_params['exc']
        g = isyn_kernel(lfp_time, np.array(spikes),
                        p['tau_rise'], p['tau_decay'], p['g_max'])
        isyn_per_syn[syn_idx] = g * (V_rest - p['E_rev'])

    for (stn_idx, syn_idx), spikes in stn_gpe_syn_events.items():
        if stn_idx != target_idx:
            continue
        p = syn_type_params['inh']
        g = isyn_kernel(lfp_time, np.array(spikes),
                        p['tau_rise'], p['tau_decay'], p['g_max'])
        isyn_per_syn[syn_idx] = g * (V_rest - p['E_rev'])

    I_syn_total = np.zeros_like(lfp_time)
    for I in isyn_per_syn.values():
        I_syn_total += I
    return isyn_per_syn, I_syn_total

# AdEx → u(t)

tau_m   = 6.0    # ms
u_T     = -50.0  # mV
DeltaT  = 2.0    # mV
R_in    = 0.001
u_reset = -65.0  # mV
spike_thr = 20.0 # mV

def run_adex(I_syn_total):
    u = np.zeros_like(lfp_time)
    u[0] = V_rest
    for i in range(1, len(lfp_time)):
        exp_term = DeltaT * np.exp((u[i-1] - u_T) / DeltaT)
        exp_term = min(exp_term, 200.0)
        du = (-(u[i-1] - V_rest) + exp_term + R_in * I_syn_total[i-1]) / tau_m
        u[i] = u[i-1] + du * dt
        if u[i] >= spike_thr:
            u[i] = u_reset
    return u

# lag correlation: <I_syn(t) · u(t+τ)>, range : ±30ms

lag_range_ms = 30.0 
lag_step     = dt      
lags_ms      = np.arange(-lag_range_ms,
                          lag_range_ms + lag_step,
                          lag_step)
lag_steps    = (lags_ms / dt).astype(int)

def compute_lag_correlation(I_syn_total, u):
    n = len(I_syn_total)

    if np.std(I_syn_total) == 0 or np.std(u) == 0:
        return np.zeros_like(lags_ms)

    I_z = (I_syn_total - I_syn_total.mean()) / I_syn_total.std()
    u_z = (u - u.mean()) / u.std()

    corr_curve = np.zeros(len(lag_steps))

    for k, shift in enumerate(lag_steps):
        if shift >= 0:  # τ > 0
            I_seg = I_z[:n - shift] if shift > 0 else I_z
            u_seg = u_z[shift:]     if shift > 0 else u_z
        else:           # τ < 0
            s = -shift
            I_seg = I_z[s:]
            u_seg = u_z[:n - s]

        if len(I_seg) == 0 or len(u_seg) == 0:
            corr_curve[k] = 0.0
            continue

        corr_curve[k] = np.dot(I_seg, u_seg) / len(I_seg)

    return corr_curve

target_20 = np.linspace(0, len(neurons)-1, 20, dtype=int).tolist()
if 118 not in target_20:
    target_20[10] = 118

print(f"20 targets: {target_20}")
print(f"   lag range: ±{lag_range_ms}ms, step: {lag_step}ms\n")

results_A = {}

for n_idx in target_20:
    print(f"▶ Processing neuron {n_idx}...")
    _, I_syn_total = compute_isyn_for_neuron(n_idx)
    u = run_adex(I_syn_total)
    corr_curve = compute_lag_correlation(I_syn_total, u)

    # peak point_lag
    peak_idx   = np.argmax(np.abs(corr_curve))
    peak_lag   = lags_ms[peak_idx]
    peak_corr  = corr_curve[peak_idx]

    results_A[n_idx] = {
        'I_syn_total': I_syn_total,
        'u':           u,
        'corr_curve':  corr_curve,
        'peak_lag':    peak_lag,
        'peak_corr':   peak_corr
    }
    print(f"   peak lag: {peak_lag:+.1f}ms  |  peak corr: {peak_corr:+.4f}")

sample_neurons = [target_20[0], target_20[5], 118, target_20[-1]]
fig, axes = plt.subplots(len(sample_neurons), 3,
                          figsize=(18, 4*len(sample_neurons)))
fig.suptitle('[Task A] I_syn_total / u(t) / Lag-Correlation: Sample Neurons',
             fontsize=13, fontweight='bold')

for row, n_idx in enumerate(sample_neurons):
    r = results_A[n_idx]

    axes[row, 0].plot(lfp_time, r['I_syn_total'], color='purple', lw=0.7)
    axes[row, 0].set_ylabel(f'N{n_idx}\nI_syn_total', fontsize=9)
    axes[row, 0].set_xlabel('Time (ms)', fontsize=8)
    axes[row, 0].grid(True, alpha=0.3)
    if row == 0:
        axes[row, 0].set_title('I_syn_total (a.u.)', fontsize=10)

    axes[row, 1].plot(lfp_time, r['u'], color='black', lw=0.7)
    axes[row, 1].axhline(u_T, color='gray', lw=0.8, ls=':', alpha=0.5,
                          label=f'threshold ({u_T}mV)')
    axes[row, 1].set_ylabel('u (mV)', fontsize=9)
    axes[row, 1].set_xlabel('Time (ms)', fontsize=8)
    axes[row, 1].legend(fontsize=7)
    axes[row, 1].grid(True, alpha=0.3)
    if row == 0:
        axes[row, 1].set_title('AdEx u(t) (mV)', fontsize=10)

    axes[row, 2].plot(lags_ms, r['corr_curve'], color='steelblue', lw=1.2)
    axes[row, 2].axvline(x=0, color='gray', lw=0.8, ls='--', alpha=0.5)
    axes[row, 2].axvline(x=r['peak_lag'], color='red', lw=1.0, ls='--', alpha=0.7,
                          label=f"peak tau={r['peak_lag']:+.1f}ms\nr={r['peak_corr']:+.4f}")
    axes[row, 2].axhline(y=0, color='gray', lw=0.5, ls=':', alpha=0.4)
    axes[row, 2].set_ylabel('Correlation', fontsize=9)
    axes[row, 2].set_xlabel('lag tau (ms)', fontsize=8)
    axes[row, 2].legend(fontsize=8)
    axes[row, 2].grid(True, alpha=0.3)
    if row == 0:
        axes[row, 2].set_title('<I_syn(t) · u(t+tau)>', fontsize=10)

plt.tight_layout()
plt.savefig('sample_u(t)_Isyn_corr.png', dpi=150, bbox_inches='tight')
plt.show()

# 20 neurons lag-correlation plot

fig, axes = plt.subplots(4, 5, figsize=(18, 14), sharex=True, sharey=True)
fig.suptitle('[Task A] 20 Neurons: <I_syn(t) · u(t+tau)> Lag Correlation',
             fontsize=13, fontweight='bold')

axes_flat = axes.flatten()
for i, n_idx in enumerate(target_20):
    r = results_A[n_idx]
    ax = axes_flat[i]
    ax.plot(lags_ms, r['corr_curve'], color='steelblue', lw=1.0)
    ax.axvline(x=0,              color='gray', lw=0.8, ls='--', alpha=0.4)
    ax.axvline(x=r['peak_lag'],  color='red',  lw=1.0, ls='--', alpha=0.7)
    ax.axhline(y=0,              color='gray', lw=0.5, ls=':',  alpha=0.4)
    ax.set_title(f"N{n_idx} | tau={r['peak_lag']:+.1f}ms\nr={r['peak_corr']:+.4f}",
                 fontsize=8)
    ax.grid(True, alpha=0.2)

for ax in axes_flat:
    ax.set_xlabel('lag tau (ms)', fontsize=7)
    ax.set_ylabel('Corr', fontsize=7)

plt.tight_layout()
plt.savefig('20neurons_u(t)_Isyn_lagcorr.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*55)
print("<I_syn(t) · u(t+tau)>")
print("="*55)
print(f"{'Neuron':>8} | {'peak_lag(ms)':>13} | {'peak_corr':>10}")
print("-"*55)
for n_idx in target_20:
    r = results_A[n_idx]
    print(f"{n_idx:>8} | {r['peak_lag']:>+13.1f} | {r['peak_corr']:>+10.4f}")

peak_lags  = [results_A[n]['peak_lag']  for n in target_20]
peak_corrs = [results_A[n]['peak_corr'] for n in target_20]
print(f"\n  mean peak lag : {np.mean(peak_lags):+.2f}ms  std: {np.std(peak_lags):.2f}ms")
print(f"  mean peak corr : {np.mean(np.abs(peak_corrs)):.4f}")
