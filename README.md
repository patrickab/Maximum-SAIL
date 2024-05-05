# Maximum Surrogate Assisted Illimunation (Maximum-SAIL)

A new evolutionary approach for Data-Efficient design exploration and optimization of expensive black-box domains.

Performed using Computational Fluid Dynamic (CFD) simulation the example of optimizing wing shapes for an High Altitude Long Endurance Unarmed Aerial Vehicles (HALE UAV)

Built upon the idea of SAIL [(1)](https://arxiv.org/abs/1702.03713)
Implemented using [pyribs](https://arxiv.org/abs/2303.00191) & [botorch](https://proceedings.neurips.cc/paper/2020/hash/f5b1b89d98b7286673128a5fb112cb9a-Abstract.html)

<p align="left">
  <img src="https://github.com/patrickab/Maximum-SAIL/assets/82589835/c2818ed9-2de8-4d9b-bc94-c0bb790b79d7" width="65%" alt="Image">
</p>

First, an initial Gaussian Process (GP) regression model is constructed by drawing an initial space-filling design (sobol sequence) from the design space and evaluating it on the black box function. This model can then be used to select the most informative point for model improvement by maximizing a given acquisition function. This is the general procedure for any Bayesian Optimization (BO) method, however "optimizing acquisition function is itself a challenging problem" [(2)](https://proceedings.neurips.cc/paper/2020/hash/f5b1b89d98b7286673128a5fb112cb9a-Abstract.html), and the optimal way to do so is still an open problem. Acquisition landscapes are often "highly non-convex, multi-modal, and [...] often flat" [(2)](https://proceedings.neurips.cc/paper/2020/hash/f5b1b89d98b7286673128a5fb112cb9a-Abstract.html), making the application of gradient-based optimizers difficult. Multi-start gradient optimization can be seen as an attempt to solve this problem but still comes with a high risk of omitting acquisition-optimal solutions by converging towards local optima. The core idea of SAIL is to use an evolutionary algorithm called MAP-Elites [(3)](https://arxiv.org/abs/1504.04909) as an optimizer for acquisition functions. This algorithm belongs to the framework of Quality Diversity Optimization (QDO) [(4)](https://arxiv.org/abs/2012.04322), meaning that we are interested in high-quality solutions, that also show diverse behavior. Algorithms belonging to this subclass of evolutionary optimizers have several advantageous properties, however, most notable for the specific use case within SAIL is the resilience against convergence towards local optima. New query points are selected by this algorithm, and then evaluated on the objective function. Subsequently, the resulting information is incorporated into the GP model, refining the understanding of the targeted function. This acquisition-feedback loop can then be repeated until the budget is exhausted. In a final step, the mean predictions of the Gaussian Process are optimized using MAP-Elites, reportedly having shown its capability of yielding high-fidelity predictions, while consuming far fewer function evaluations compared to MAP-Elites without surrogate assistance. 

## Maximum SAIL

This new approach replaces the Upper Confidence Bound (UCB) acquisition function used in the original paper with Max-Value Entropy Search (MES) [(4)](https://arxiv.org/abs/1703.01968). By combining MES with further computationally cheap operations, this new algorithm has proven capable of rapidly exploring and optimizing the complicated search space (11 dimensions) of PARSEC encoded airfoils. Further benchmarks yet remain to be done, however, all relevant parameters of the algorithm dynamically adapt to the specified search-space. This is done inside MAP-Elites by generating new solutions using standard deviations, that perform Gaussian mutantions scaled to each dimension of the search-space ([interactive example](https://github.com/patrickab/Maximum-SAIL/blob/master/plots/mutants.py)). Therefore, the algorithm should at least in theory perform just as good in other benchmarks.

An 11-dimensional vector defines parameters, that can be translated into an upper and lower polynomial, which can be translated into xy-coordinates that encode a 2d cross-section of an airplane wing. In the image below we see a 3d model of a wing shape optimized by Maximum-SAIL

![image](https://github.com/patrickab/Maximum-SAIL/assets/82589835/acd294a0-dbfe-4f97-b215-58d6286ecb99)

After each acquisition loop, a batch of `BATCH_SIZE` solution vectors is expressed as 2d coordinates and stored inside a file. This file can then be evaluated using CFD Simulation, in our case [XFOIL](https://web.mit.edu/drela/Public/web/xfoil/), developed by the Massachusets Institute of Technology (MIT) and widely used in the aerodynamic engineering industry.

Using this software we can calculate both the aerodynamic resistance (drag) and the upward force (lift) generated at a given angle and velocity and set of coordinates. We can then define our objective function as the ratio $\frac{\text{lift}}{\text{drag}}$, which represents the goal of generating airfoils capable of carrying lots of cargo, while simultaneously consuming low amounts of fuel. 

As mentioned, the goal is not only to calculate high-performing solutions but also to find diverse solutions. Considering SAIL in the QDO setting, we measure behavior/diversity by considering two dimensions of the input vector. We can then define a grid upon this subspace, where we store the best solutions found during the optimization process by discretizing the corresponding dimensions.

In this example the x,y location of the highest point of the upper polynomial on a [25,25] Grid.

In comparison to Vanilla-SAIL [(1)](https://github.com/agaier/sail?tab=readme-ov-file), Maximum-SAIL has shown its capability to dramatically reduce the required black-box function evaluations by 75%, while also discovering higher-performing solutions, producing higher-fidelity predictions, and achieving higher coverage.

### Vanilla-SAIL with 1280 function evaluations / Batch Size 10 [Source Code](https://github.com/patrickab/Maximum-SAIL/commit/96c87ad7581f62046bff20d3ad32a419b5bdf504)

True function evaluations &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Predictions &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Evaluated Predictions &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Prediction Error in %

<p align="left">
  <img src="https://github.com/patrickab/Maximum-SAIL/assets/82589835/0a8648bc-45e8-4374-bc42-9bcc955065d4" width="100%" alt="Image">
</p>

### Maximum-SAIL with 320 function evaluations / Batch Size 20 [Source Code](https://github.com/patrickab/Maximum-SAIL/commit/e3b1528a526835d7eb27c476775ff2fc6372513b)

True function evaluations &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Predictions &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Evaluated Predictions &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Prediction Error in %

<p align="left">
  <img src="https://github.com/patrickab/Maximum-SAIL/assets/82589835/b12f0a06-d469-4c8a-8a71-67b4385e5dad" width="100%" alt="Image">
</p>

