# Auditory-Neuromodulation
Exploring closed loop auditory neuromodulation approach to modulate theta oscillations and study the entrainment effects  using different auditory stimuli types.

![Cover](https://github.com/Sruthi-sk/Auditory-Neuromodulation/blob/main/India%20-%20EMBO%20poster.png)

## Comments
The polar charts showed that the majority of stimulations were presented at 0 phase, however, some of the detections seemed to be at different phases. This could be explained by the Gibbs phenomenon during the Hilbert phase extraction in real time - which led to distortions in the phase values of the last few samples. 
We have corrected for the hilbert phase code using ECHT: Endpoint Corrected Hilbert Transform mentioned in Schreglmann, Sebastian R., et al. “Non-Invasive Amelioration of Essential Tremor via Phase-Locked Disruption of Its Temporal Coherence.” 2020, doi:10.1101/2020.06.23.165498. 

A basic exploration of the echt code was done [here](https://github.com/rahulvenugopal/Phases-fading-away)
