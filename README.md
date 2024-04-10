# Maximum Surrogate Assisted Illimunation (Maximum-SAIL)

A new approach for Data-Efficient design exploration, optimization.
Built upon the idea of the [SAIL-Algorithm](https://arxiv.org/abs/1702.03713)

<p align="center">
  <img src="https://github.com/patrickab/Maximum-SAIL/assets/82589835/ab50e" width="70%" alt="Image">
</p>

[Source](https://arxiv.org/abs/1702.03713)


By replacing the Upper Confidence Bound (UCB) acquisition function with MES (Max-Value Entropy Search),
and combining this acquisition function with further computationally cheap operations, this new algorithm
has proven capable of rapidly exploring and optimizing the complicated searchspace (11 dimensions) of
PARSEC encoded airfoils. Further benchmarks yet remain to be done, however, all relevant parameters of the
algorithm dynamically adapt to the specified search-space. All subsequent operations for calculating new 
solutions are bound to standard deviations, that perform Gaussian Mutantions scaled to each dimension of the
search-space. Therefore, the algorithm should at least in theory perform just as good in other benchmarks.

An 11-dimensional vector defines parameters, that can be translated into
an upper and lower polynomial, which can be translated into xy-coordinates
that encode a wing for an airplane. These coordinates are stored inside a file, 
that can subsequently be evaluated using Computational Fluid Dynamic (CFD) Simulation, eg with [XFOIL](https://web.mit.edu/drela/Public/web/xfoil/).
This program has been developed by the Massachusets Institute of Technology (MIT).

![airfoil](https://github.com/patrickab/Maximum-SAIL/assets/82589835/841673fc-d407-4d91-824f-9068293c0722)

This software can be used to calculate both the aerodynamic resistance (drag) and
the upward force (lift) generated at a given angle and velocity.

We can then define our objective function as the ratio \frac{\text{lift}}{\text{drag}}, which represents the goal of
generating airfoils capable of carrying lots of cargo, while simultaneously consuming low amounts of fuel

Now, considering Illumination Algorithms such as SAIL, we do not only want to calculate
high-performing solutions, we also want diverse solutions. Diversity can be measured for
example by considering two dimensions of the input vector. We can then define a grid upon
this subspace, where we store the best solutions found during the optimization process.

In this example, were considering the x,y location of the highest point of the upper polynomial on a [25,25] Grid.

In comparison to [Vanilla-SAIL](https://github.com/agaier/sail?tab=readme-ov-file), Maximum-SAIL has shown capable to 
dramatically reduce the required black-box function evaluations, while also achieving better results than Vanilla-SAIL
We can observe how this new algorithm finds higher-performing solutions, produces higher-fidelity predictions,
and achieves higher coverage, while simultaneously requiring only 25% of the budget, that Vanilla-SAIL needed
to produce weaker results.

### Vanilla-SAIL with 1280 function evaluations / Batch Size 10
![vanilla-1280](https://github.com/patrickab/Maximum-SAIL/assets/82589835/4e3833bd-bb5f-4035-b070-4885f742996f)


### Maximum-SAIL with 320 function evaluations / Batch Size 20
![final_heatmaps_0](https://github.com/patrickab/Maximum-SAIL/assets/82589835/f96c5f67-60bd-47af-9198-18b140fc3bc0)

We can observe how this new algorithm finds higher-performing solutions, produces higher-fidelity predictions,
and achieves higher coverage, while simultaneously requiring only 25% of the budget, that Vanilla-SAIL needed
to produce weaker results.



