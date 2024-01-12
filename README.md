# Equilibrium Reaction Predictive Model

Develop and evaluate an algorithm to discover the mathematical model of molecular interactions. Authors: Sheldon Zhu, Eudora Fong, Joseph Perez

## General 
This model is a machine learning algorithm for accurately and efficiently predicting the species of equilibrium chemical reactions with very limited input data. It is designed to take in as input only the initial and added concentrations and volumes of each species {`c0`, `cadded`, `v0`, `vadded`}, initial species names, and maximum number of species to generate. A key feature of # is that the algorithm can process various scales of reactions due to the nature in which chemical models are generated and then optimized. 

Chemical model species names are generated through combinatorics applied to the initial species names, namely the process of combinations with replacement, which generates all possible arrangements of names, outputting a specific number of species names. The length of each new name generated is dependent on the number of initial species names, and this number increases with every iteration until the given maximum number of species is reached. Each new name is added to the list of initial species names with each iteration to create new chemical models to be optimized. 

To further complete each chemical model, beta values for each species in a given model must be optimized. To do this, Newton-Gauss-Levenberg-Marquart (NGLM) optimization is applied using an initial set of beta starting values 0, `c0`, `v0`, and `sig_k` of best initialization values. Within the NGLM algorithm, the residuals between the given chemical reaction’s observed pH data and the pH data simulated for each generated model. To simulate this pH data, we utilize the Newton-Raphson method, which for our purposes takes in a model, 0 in this case, and gives us a concentration matrix to calculate simulated pH data, which we compare with the observed pH data to get our residuals.

Further developments toward the algorithm have been made by extending the amount of parameters to be optimized. Instead of just optimizing the initial betas and models, we also optimized initial concentration by implementing grid search combined with our original nglm algorithm for beta and model optimization. This exhaustive search would determine which initial concentration value gives the lowest loss and thus offers the model and betas most closely fitted with our measured data.

[Algorithm Mechanics][https://docs.google.com/document/d/1q5KUX-xLURCOdjE8wkjHY9qv6dRMphpyP_P_VNnsyuw/edit?usp=sharing]


## Run `python3 main.py`


