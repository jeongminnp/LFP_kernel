# Decoding Neural spikes from LFPs
<img width="641" height="351.4832483" alt="image" src="https://github.com/user-attachments/assets/4430666f-b2a1-492f-89e9-c8b054cf8236" />

## Single STN
<img width="250" height="270" alt="Pasted Graphic" src="https://github.com/user-attachments/assets/2d877a02-73f2-433e-b121-a5cd64b3d636" />
<img width="300" height="306.5934" alt="Radial neuron structure + synapse locations" src="https://github.com/user-attachments/assets/d4c0767d-9a1e-4805-b53f-064466cedd95" />

- soma + dendrite(5) + branch(10) raidal 형태
- 시냅스 분포 시
  <br>
  억제성 시냅스 > soma & proximal dendrite에 주로 분포
  <br>
  흥분성 시냅스 > distal dendrite & branch에 주로 분포

- [soma : dendrite] 구간 내 유형별 시냅스 분포
  <br>
  proximal boundary : 0 ~ 20% = 30 * 0.5(억제성), 100 * 0.2(흥분성)
  <br>
  middle boundary : 20 ~ 70% = 30 * 0.3(억제), 100 * 0.3(흥분)
  <br>
  distal boundary : 70 ~ 100% = 30 * 0.2(억제), 100 * 0.5(흥분)

## References
[Ball-and-stick](https://nrn.readthedocs.io/en/latest/tutorials/ball-and-stick-1.html)

Neuron; 각 기관 형상, 좌표, 유형 정보 클래스 객체로 나열

<https://pmc.ncbi.nlm.nih.gov/articles/PMC2701041/>

  ”In general, the pyramidal cell's soma and axon initial segment receive only symmetric (inhibitory) synapses from axon terminals of GABAergic interneurons, whereas its dendrites receive synaptic inputs from axon terminals forming both symmetric and asymmetric (excitatory) synapses. ”

<https://pmc.ncbi.nlm.nih.gov/articles/PMC2712268/>

”Six terminals established symmetrical synaptic contacts with the soma and proximal dendrite of a single STN neuron.”
