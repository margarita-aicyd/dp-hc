# dp-hc

Codes for implementing the algorithm in On the Price of Differential Privacy for Hierarchical Clustering

### Libraries required
- Numpy
- Pandas
- Scipy
- Networkx
- Matpplotlib
- Scikit-learn
- Pulp
- Scikit-learn

### Implementation of Sparsest Cuts

Three approximation algorithms are implemented in this repository. An approximation algorithm based on cheegar inequality, a greedy algorithm and the Leighton-Rao algorithm. We recommend the first one for efficiency. The Leighton-Rao algorithm easily break down when the size of graph is larger than 10.

### Run the codes

Parameters:
- d: dataset name
- s: sparsest cut algorithm('cheegar','greedy','leightonrao')
- The privacy parameter epsilon can be changed directly in run.py

An example of running on the beedance dataset:

```
python run.py -d 'sbm' -s 'greedy'
```