# Decoding Neural spikes from LFPs
<img width="616" height="328" alt="image" src="https://github.com/user-attachments/assets/a9bf84c1-95f0-4acb-98a3-cd4881c1bedc" />

## Single STN
<img width="250" height="270" alt="Pasted Graphic" src="https://github.com/user-attachments/assets/a8dddb21-5668-4c7a-aaa7-91102a158194" />
<img width="300" height="306.5934" alt="Radial neuron structure + synapse locations" src="https://github.com/user-attachments/assets/391a953e-e6bb-4c47-9d4b-542482fdd384" />

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
