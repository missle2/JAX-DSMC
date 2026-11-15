Top level:
6 pages, not including citations and appendices.
focused on the machine learning aspects of the project; this write up is to submit to a machine learning class project rather than a draft for a paper or similar. 
We include all the mathematical details of the construction of DSMC with NTC/BGK/FP & ES, and an explanation of the AD DSMC for training, in appendices rather than the core article to enable preserving focus on the machine learning aspects.


Class requirements:
The six requirements
Requirement
R1	A decision, not a dataset. Who decides, what they decide, cost of each error type. “Predict house prices” is not a decision. “The assessor's office can inspect 300 of 40,000 properties a year and wants the ones most likely mis-valued” is.
R2	A baseline declared before you modeled. Report it beside your headline number every time. If the majority class gets 0.94 and you got 0.95, that is a failure, and say so.
R3	Two models across the interpretability axis. One readable by a non-expert (regularized linear or logistic, tree depth 4 or less). One flexible (random forest, GBM, small net). Same split, same metric, honestly tuned. Report both even when the simple one wins.
R4	Unsupervised structure where it changes the decision. Use it when it changes what you do, and if you do not use it, say in one sentence what you checked and why it did not help. A component that never reappears in the decision scores zero either way.

When it does earn its place: PCA or another dimensionality reduction, k-means, Gaussian mixtures, hierarchical clustering, or matrix completion for missingness, feeding a downstream decision. A PCA plot that never reappears is decoration and scores zero. A third of this course is unsupervised learning. Use it.
R5	A failure analysis with a named worst slice. Slice test performance by at least two variables you did not predict. Report the worst slice beside the headline. Explain why it fails there: scarcity, shift, label noise, an unintended proxy. Name the slices you intend to examine in the proposal where you can, report each slice's sample size and an interval, and mark any slice you found by searching as exploratory. A worst slice of 11 observations with no interval is noise, not a finding.
R6	A recommendation, and the fact that would reverse it. Which model you would deploy and why. Then one concrete thing, a regulation, a cost change, a new data source, a population shift, that flips your answer. If nothing could, you have not understood the tradeoff.
AI appendix (required, not counted in the page limit): what you asked a model to do, what it got wrong, how you caught it. Generative AI is fully permitted on this project, in code, prose and ideas. It is not permitted in the oral, and the oral is where the grade is.

Overall outline:
 **Introduction**
 P1: Rarefied Flow, applications, the boltzmann equation vs navier stokes, the popularity of DSMC as the reference numerical solution for it, general construction of DSMC.

 P2: High cost of DSMC due to grid and timestep requirements. Prompt linearized approaches to solve more cheaply, but introduce inaccuracies. Neural net models to correct - brief lit review of previous work here.

 P3: Brief overview of the approach we take. Correcting ES-FP operator, two models of varying complexity in input and output space.

 **Modelling**
 The set up of the problem: Table of inputs and the justification for why we want to consider each of them, with citations. Objective function, why we chose that measure of error, and the AD of DSMC enabling fast evaluation of it's gradient. Reference DSMC data, the set of cases considered, carefulness to configure the cases in terms of steps etc., why we chose those cases, and how it related to generalizing to real engineering problems. The per seed cross evaluation fold set up, and why we picked it. The reality of monte carlo estimates of the objective function and it's gradients, implications for training and what makes a model actually better (2 sigma test), and measures taken to try to minimize noise. The reference models - nearest neighbor NTC, BGK, ES-FP. How a model has to beat them in order to be preferable.

 ^ Not necessarily in this order, we just need to introduce all the common threads.

 **The Lasso**

 The Linear Regression Model. The hope of a sparse regression enabling extremely interpritable results. Running both single regression for just the rate - hoping to get an ok and extremely interpritable model - and over the rich outputs.

 The actual details of the lasso constrution. Gauss Newton, linearized objective function for AD, selection of alpha based on CV.

 Results - model fails

 Figure: magnitude of each coefficient for the model with a single output vs alpha.
 
 Interpritability of the model; what coefficients drop out first, etc.

 **The MLP**
 The details of the MLP set up. Cross validation to pick learning rate, weight decay, depth, and width.

 Figure - bias/variance curve vs size of the model. (i.e. x axis --> # parameters, log scale). Learning rate/weight picked on optimal for particular model size. Labels for each point to illustrate this. Independent sets of lines for 8 width and 32 width models (i.e. 4 lines in total, 2x for 8, 2x for 32 - a bias and a variance curve for each)

 Interprit figure. Overfitting with 32x3; all 8 width dramatically underfit, etc.

 Performance of model 

 **Why the Models Fail and Succeed**
 PCA on the features informing strong covariance between parameters, leads to failure of the lasso even with analyzing over many runs. Nonlinearity in MLP saves it.

 A figure or table on PCA things - correlation between PCs and our features & % of variance each explains.

 Failure of Linear model to generalize explained from the nearest neighbor model. Much better for the MLP.


 **The Decision**
 Engineers evaluate properties of vehicle based on simulations. Correctness of simulations controls engineering accuracy - failure results in satellites crashing, astronauts burning up on entry, or a microfluidics medical device failing and killing a patient. Need to validate and compare to full fidelity.

 Paragraph I had before for reference on the choices engineers would have to make:

 When taken beyond prototypes and deployed to accelerate engineering analysis, the decision based on the results of a simulation must be the responsibility of a qualified engineer. If such a neural network model is used to predict the drag on a satellite, but produces an incorrect result due to failure to generalize to a particular geometry, it can lead to premature reentry of billions of dollars of spacecraft. Worse, if such a tool is used to predict the entry heating of a spacecraft, but was optimized for prediction of aerodynamic forces, it may lead to unexpectedly high heating. In the worst case, the vehicle could burn up and astronauts aboard would be killed. On the flip side, a model that may be overly conservative in engineering estimates would result in needlessly expensive craft or incorrectly demonstrate some projects may be unfeasible. Thus, this tool (if deployed) should be treated similarly to how engineers currently use lower fidelity tools. It should only be trusted in domains it has been specifically validated in against higher fidelity models, and when used to produce analysis for the performance of a vehicle be used in conjunction with full, more expensive, simulations at key data points to anchor the model results.  Such an approach would minimize the chance of a large error while maintaining a large part of the massive acceleration in analysis such a model would enable.

 Here - recommendation --> deploy MLP. Use nearest neighbor model to cheaply analyze if linear model would work better for an engineer's use case, and switch to it if so. Example -- linear works decently in low speed shear flows, likely show up in biological microfluidics applications, where the MLP's better performance for shocks is much less useful than 

 Appendecies - code source & general reproducibility notes (where to find core code, how to run, build prerequesites), mathematical details, thanks to CURC for compute resources, citations