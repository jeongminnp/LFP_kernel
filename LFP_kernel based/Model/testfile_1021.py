from neuron import h, gui
import numpy as np
import matplotlib.pyplot as plt
import LFPy

class RadialNeuron:
    def __init__(self, soma_radius=10, num_dend=5, dend_length=100, dend_radius=2, branch_length=50, branch_radius=1):
        self.soma_radius = soma_radius
        self.num_dend = num_dend
        self.dend_length = dend_length
        self.dend_radius = dend_radius
        self.branch_length = branch_length 
        self.branch_radius = branch_radius
        self._setup_morphology()
        self._setup_biophysics()

    def _setup_morphology(self):
        self.soma = h.Section(name='soma')
        self.soma.L = self.soma.diam = 2*self.soma_radius
        self.soma.nseg = 1
        h.pt3dclear(sec=self.soma)
        h.pt3dadd(0,0,0, self.soma.diam, sec=self.soma)
        h.pt3dadd(0,0,2*self.soma_radius, self.soma.diam, sec=self.soma)
        self.dendrites = []
        self.branches = []
        self.dend_coords = []
        self.branch_coords = []
        phi_dend = np.linspace(0, np.pi, self.num_dend+2)[1:-1]
        theta_dend = np.linspace(0, 2*np.pi, self.num_dend, endpoint=False)

        for i,(phi,theta) in enumerate(zip(phi_dend,theta_dend)):
            dend = h.Section(name=f'dend_{i}')
            dend.L = self.dend_length
            dend.diam = self.dend_radius*2
            dend.nseg = 5
            dend.connect(self.soma(1))
            self.dendrites.append(dend)

            x0 = self.soma_radius*np.sin(phi)*np.cos(theta)
            y0 = self.soma_radius*np.sin(phi)*np.sin(theta)
            z0 = self.soma_radius*np.cos(phi)
            x1 = (self.soma_radius+self.dend_length)*np.sin(phi)*np.cos(theta)
            y1 = (self.soma_radius+self.dend_length)*np.sin(phi)*np.sin(theta)
            z1 = (self.soma_radius+self.dend_length)*np.cos(phi)
            p0 = np.array([x0,y0,z0])
            p1 = np.array([x1,y1,z1])
            self.dend_coords.append( (p0, p1) )
            h.pt3dclear(sec=dend)
            h.pt3dadd(p0[0], p0[1], p0[2], dend.diam, sec=dend)
            h.pt3dadd(p1[0], p1[1], p1[2], dend.diam, sec=dend)

            mid = 0.5*(p0 + p1)
            dend_axis = p1 - p0
            dend_axis = dend_axis / np.linalg.norm(dend_axis)
            rot_axis = np.cross(dend_axis, [0.0,0.0,1.0])
            if np.linalg.norm(rot_axis) == 0:
                rot_axis = np.array([1.0, 0.0, 0.0])
            else:
                rot_axis /= np.linalg.norm(rot_axis)
            for sign in [-1, 1]:
                angle = np.radians(30) * sign
                k = rot_axis
                v = dend_axis * self.branch_length
                v_rot = v*np.cos(angle) + np.cross(k,v)*np.sin(angle) + k*np.dot(k,v)*(1-np.cos(angle))
                p1_branch = mid + v_rot
                br = h.Section(name=f'branch_d{i}_{sign}')
                br.L = self.branch_length
                br.diam = self.branch_radius*2
                br.nseg = 3
                br.connect(dend, 0.5)
                h.pt3dclear(sec=br)
                h.pt3dadd(mid[0], mid[1], mid[2], br.diam, sec=br)
                h.pt3dadd(p1_branch[0], p1_branch[1], p1_branch[2], br.diam, sec=br)
                self.branches.append(br)
                self.branch_coords.append((mid, p1_branch))

    def _setup_biophysics(self):
        for sec in [self.soma] + self.dendrites + self.branches:
            sec.Ra = 100
            sec.cm = 1
        self.soma.insert('hh')
        for dend in self.dendrites:
            dend.insert('pas')
        for br in self.branches:
            br.insert('pas')

    def place_synapses(self, n_exc=100, n_inh=30):
        self.syns = []
        self.syn_info = []
        # soma : 6 inh syn
        for _ in range(6):
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            
            syn = h.Exp2Syn(self.soma(np.random.uniform(0.05, 0.95)))
            syn.tau1 = 0.5
            syn.tau2 = 8.0
            syn.e = -75.0
            self.syns.append(syn)
            self.syn_info.append({'type': 'inh', 'section': self.soma, 'theta': theta, 'phi': phi, 'x': syn})

        # divide section
        exc_counts = [int(n_exc*0.2), int(n_exc*0.3), n_exc-int(n_exc*0.2)-int(n_exc*0.3)]
        inh_counts = [int((n_inh-6)*0.5), int((n_inh-6)*0.3), (n_inh-6)-int((n_inh-6)*0.5)-int((n_inh-6)*0.3)]

        regions = [(0.0,0.2), (0.2,0.7), (0.7,1.0)]
        # exc syn
        exc_targets = self.dendrites + self.branches
        for count, (start, end) in zip(exc_counts, regions):
            for _ in range(count):
                sec = np.random.choice(exc_targets)
                if sec in self.dendrites:
                    x = np.random.uniform(start, end)
                else:
                    x = np.random.uniform(0.0, 1.0)
                syn = h.Exp2Syn(sec(x))
                syn.tau1 = 0.5
                syn.tau2 = 3.0
                syn.e = 0.0
                self.syns.append(syn)
                self.syn_info.append({'type': 'exc', 'section': sec, 'x': x})
        # inh syn
        for count, (start, end) in zip(inh_counts, regions):
            for _ in range(count):
                sec = np.random.choice(self.dendrites)
                x = np.random.uniform(start, end)
                syn = h.Exp2Syn(sec(x))
                syn.tau1 = 0.5
                syn.tau2 = 8.0
                syn.e = -75.0
                self.syns.append(syn)
                self.syn_info.append({'type': 'inh', 'section': sec, 'x': x})
    
    def place_electrodes(self):
        electrode_positions = []
        for info in self.syn_info:
            sec = info['section']
            if sec in self.dendrites:
                idx = self.dendrites.index(sec)
                p0, p1 = self.dend_coords[idx]
                x = info.get('x', None)
                if x is None:
                    continue
                pos = p0 + x*(p1 - p0)
            elif sec in self.branches:
                idx = self.branches.index(sec)
                p0, p1 = self.branch_coords[idx]
                x = info.get('x', None)
                if x is None:
                    continue
                pos = p0 + x*(p1 - p0)
            elif sec == self.soma:
                # coordinate transform
                r = self.soma_radius
                theta = info.get('theta', 0)
                phi = info.get('phi', 0)
                pos = np.array([
                    r * np.sin(phi) * np.cos(theta),
                    r * np.sin(phi) * np.sin(theta),
                    r * np.cos(phi)
                ])
            else:
                continue
            electrode_positions.append(pos)
        electrode_positions = np.array(electrode_positions).T
        self.electrode_positions = electrode_positions
        self.electrodes = LFPy.RecExtElectrode(cell=self.soma, sigma=0.3,
                                           x=electrode_positions[0], y=electrode_positions[1], z=electrode_positions[2],
                                           method='linesource')

    def simulate_with_lfp(self):
        h.finitialize(-65)
        h.continuerun(100)
        
    def visualize(self):
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        u,v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
        xs = self.soma_radius*np.sin(v)*np.cos(u)
        ys = self.soma_radius*np.sin(v)*np.sin(u)
        zs = self.soma_radius*np.cos(v)
        ax.plot_surface(xs,ys,zs,color='lightgray',alpha=0.2,edgecolor='k')
        for p0,p1 in self.dend_coords:
            X,Y,Z = self._cylinder_3d(p0,p1,self.dend_radius)
            ax.plot_surface(X,Y,Z,alpha=0.1,edgecolor='k')
        for p0,p1 in self.branch_coords:
            X,Y,Z = self._cylinder_3d(p0,p1,self.branch_radius)
            ax.plot_surface(X,Y,Z,alpha=0.08,edgecolor='k')
        # section display (0.2, 0.7)
        for frac, color, label in zip([0.2, 0.7], ['purple','orange'], ['proximal','middle']):
            r = self.soma_radius + frac*self.dend_length
            u_b, v_b = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
            x_b = r * np.sin(v_b) * np.cos(u_b)
            y_b = r * np.sin(v_b) * np.sin(u_b)
            z_b = r * np.cos(v_b)
            ax.plot_wireframe(x_b, y_b, z_b, color=color, alpha=0.3, linewidth=1)
            ax.text(r*1.05, 0, 0, f'{label} boundary', color=color, fontsize=9)
        for info in self.syn_info:
            sec = info['section']
            typ = info['type']
            color = 'blue' if typ=='exc' else 'red'
            if sec in self.dendrites:
                x = info['x']
                idx = self.dendrites.index(sec)
                p0, p1 = self.dend_coords[idx]
                pos = p0 + x*(p1-p0)
            elif sec in self.branches:
                x = info['x']
                idx = self.branches.index(sec)
                p0, p1 = self.branch_coords[idx]
                pos = p0 + x*(p1-p0)
            elif sec == self.soma:
                r = self.soma_radius
                theta = info['theta']
                phi = info['phi']
                pos = np.array([
                    r * np.sin(phi) * np.cos(theta),
                    r * np.sin(phi) * np.sin(theta),
                    r * np.cos(phi)
                ])
            else:
                continue
            ax.scatter(pos[0], pos[1], pos[2], c=color, s=18)
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_zlabel('Z (μm)')
        ax.set_title('Radial neuron: structure + synapse locations')
        plt.show()

    def _cylinder_3d(self, p0, p1, radius, n_theta=20, n_z=2):
        v = p1 - p0
        length = np.linalg.norm(v)
        if length == 0:
            vz = np.array([0,0,1.0])
        else:
            vz = v/length
        if vz[0]==0 and vz[1]==0:
            vx = np.array([1,0,0])
        else:
            vx = np.cross(vz,[0,0,1])
            vx /= np.linalg.norm(vx)
        vy = np.cross(vz,vx)
        theta = np.linspace(0,2*np.pi,n_theta)
        z = np.linspace(0,length,n_z)
        theta_grid, z_grid = np.meshgrid(theta,z)
        X = p0[0] + vz[0]*z_grid + radius*np.cos(theta_grid)*vx[0] + radius*np.sin(theta_grid)*vy[0]
        Y = p0[1] + vz[1]*z_grid + radius*np.cos(theta_grid)*vx[1] + radius*np.sin(theta_grid)*vy[1]
        Z = p0[2] + vz[2]*z_grid + radius*np.cos(theta_grid)*vx[2] + radius*np.sin(theta_grid)*vy[2]
        return X,Y,Z

if __name__ == "__main__":
    neuron = RadialNeuron()
    neuron.place_synapses(n_exc=100, n_inh=30)
    neuron.place_electrodes()
    neuron.simulate_with_lfp()
    print(f"Created {len(neuron.syns)} synapses ({sum(1 for s in neuron.syn_info if s['type']=='inh')} inh, {sum(1 for s in neuron.syn_info if s['type']=='exc')} exc).")
    neuron.visualize()
    
    print(f"Number of electrodes placed: {neuron.electrode_positions.shape[1]}")
    print("Electrode positions:")
    print(neuron.electrode_positions)
