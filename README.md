# CausalInference
Project for Math 522 at BYU. We are interested in understanding causal inference in deep learning.

# Previous Research

This field essentially exists because of the work of Judea Pearl. Pearl's main idea is that of Pearl's causality hierarchy. This hierarchy is as follows:

1. Association
* Seeing. (What if I see...?)
2. Intervention
* Doing. (What if I do...? How?)
3. Counterfactuals
* Imagining. (What if I had done...?)



<h2><a href="https://arxiv.org/abs/2107.00793" target="_blank" rel="noopener noreferrer">The Causal-Neural Connection: Expressiveness, Learnability, and Inference</a> </h2>

Talks about how neural networks and structural causal models are connected. Currently, the only way to identify causal effects is through $\text{do}()$ calculus. An interesting question, now, is if it is possible to estimate causal effects with neural models. Pearl's causality hierarchy is the basis of this paper.

One key finding of this paper is given a bunch of data that only represents observational data, a neural network is unable to identify causal relationships or effects of intervention. The neural causal model (ncm) is proposed. NCM is a structural causal model (scm) that is based on neural nets and can be learned using gradient descent. Uses feed-forward neural networks. The idea is to approximate the connections between variables by performing the other aspects of Pearl's causality heirarchy.

<h2><a href="https://arxiv.org/abs/2109.04173" target="_blank" rel="noopener noreferrer">Relating Graph Neural Networks to Structural Causal Models</a></h2>

The basic idea is to introduce the concept of interventions in GNNs to jointly learn embeddings and causal effects. This is implemented through so-called intervential GNN layers. 

<h2><a href="https://www.youtube.com/watch?v=-UjytpbqX4A" target="_blank" rel="noopener noreferrer">Pytorch Geometric Tutorial (Video, start at 33:33)</a></h2>

A nice introductory tutorial on Pytorch Geometric. Pytorch Geometric is a library used for graph neural networks.


<h2><a href="https://pytorch-geometric.readthedocs.io" target="_blank" rel="noopener noreferrer">PyTorch Geometric Documentation</a></h2>

Maybe we can find a way to take a super basic dataset and implement a GNN on it. Then we can remove connections to nodes and see how well/poor the prediction is.

<h1>Potential Datasets</h1>

<h2><a href="https://www.causality.inf.ethz.ch/data/LUCAS.html" target="_blank" rel="noopener noreferrer">LUCAS0: Medical Diagnosis</a></h2>

This is a synthetically generated dataset by causal Bayesian networks with binary variables. It has 11 different variables and a ton of samples. The variables interact with each other in a DAG format which is perfect for using SCMs and GNNs. Maybe we could take the data, format it in a way the PyTorch Geometric likes it, and then run a GNN on it, removing different connections? Maybe that can simulate a $\text{do}()$ operation?
