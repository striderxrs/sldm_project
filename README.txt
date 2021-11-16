Projekt: Neuroeconomics – Sensorimotor Learning and Decision-Making

Code information:
python version 3.6.7
tensorflow 1.13.1


serial_gradient.py:
Execute this code to run the gradient descent optimization on the noise-correlation paramter "rho" of the distribution p(x|w).

Agent.py:
This class represent an bounded rational agent that implements the "Serial Information Processing Hierarchies" 
according to
 "Bounded Rationality, Abstraction, and Hierarchical Decision-Making: An Information-Theoretic Optimality Principle" by Genewein et al."
applied to
 "Selective Changes in Noise Correlations Contribute to an Enhanced Representation of Saccadic Targets in Prefrontal Neuronal Ensembles by Dehaqani et al."


Util:
DataStorage.py:
This class creates a storage with some important parameters of the Gradient Descent update of the Agent.

helper.py: 
Some helper functions to calculate the DKL between two probability distributions, a gaussian utility function, the population vector 
and to plot the p(x|w)-distribution and p(x|w)-means

results:
(empty) directory to store the update results.


You can use tensorboardX to observe the update runs online in the browser. 
The directory "runs" will be automatically created and stores the observed variables written by the SummaryWriter (see Agent.py).