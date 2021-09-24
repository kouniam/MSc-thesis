# MSc-thesis

This repository contains a simplified archive of the code/scripts used to implement the model I designed for my master thesis project. It makes extensive use of the DICE (Dynamic Integrated Climate-Economy) model, an Integrated Assessment Model (IMA) developed by economist William Nordhaus.

* DICE_optimizer.py - An adapted version of the DICE IMA (v2016), which runs on an optimization routine that outputs optimal path projections. [updated to DICEv2018]
* coupled_DICE.py - This version of DICE incorporates the coupling mechanism into the opinion dynamics model and is the main structure of the project.
* generator.py - Custom script that generates the 2D grid required to simulate dynamic network behaviour. [makes use of NetworkX]
* opinion_dynamics.py - A variation of the statistical physical Ising model that emulates opinion formation in a society. 
* pack_unpack.py - Auxiliary algebraic functions to increase performance.
* presentation.pdf - Short pdf version of the final presentation, with selected visuals.
* thesis_final_v2.pdf - The full text of my thesis, with detailed explanations of theory and results. [includes list of references]
