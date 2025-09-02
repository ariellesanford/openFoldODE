Protein Folding with Neural Ordinary Differential Equations
==============

openFoldODE implements a Neural Ordinary Differential Equation (Neural ODE) formulation of the Evoformer. Instead of fixed discrete blocks, we use a continuous-depth parameterization that preserves the Evoformer’s attention-based operations while enabling efficient and adaptive computation.


References
----------
- Gianluca Ceruti, Christian Lubich, Dominik Sulz  
  Rank-adaptive time integration of tree tensor networks  
  [SIAM J. Numer. Anal. 61, 194-222 (2023)](https://doi.org/10.1137/22M1473790)  
  [Code repository](https://github.com/DominikSulz/rank_adaptive_integrator_for_TTN)
- Gianluca Ceruti, Jonas Kusch, Christian Lubich  
  A rank-adaptive robust integrator for dynamical low-rank approximation  
  [BIT Numerical Mathematics 62, 1149-1174 (2022)](https://doi.org/10.1007/s10543-021-00907-7)  
  [Code repository](https://github.com/JonasKu/publication-A-rank-adaptive-robust-integrator-for-dynamical-low-rank-approximation)
- Christian Lubich, Bart Vandereycken, Hanna Walach  
  Time integration of rank-constrained Tucker tensors  
  [SIAM J. Numer. Anal. 56, 1273-1290 (2018)](https://doi.org/10.1137/17M1146889)


References
----------

- Gustaf Ahdritz, Nazim Bouatta, Christina Floristean, Sachin Kadyan, Qinghui Xia, William Gerecke, Timothy J. O'Donnell, Daniel Berenberg, Ian Fisk, Niccolò Zanichelli, Bo Zhang, Arkadiusz Nowaczynski, Bei Wang, Marta M. Stepniewska-Dziubinska, Shang Zhang, Adegoke Ojewole, Murat Efe Guney, Stella Biderman, Andrew M. Watkins, Stephen Ra, Pablo Ribalta Lorenzo, Lucas Nivon, Brian Weitzner, Yih-En Andrew Ban, Shiyang Chen, Minjia Zhang, Conglong Li, Shuaiwen Leon Song, Yuxiong He, Peter K. Sorger, Emad Mostaque, Zhao Zhang, Richard Bonneau, Mohammed AlQuraishi
  OpenFold: retraining AlphaFold2 yields new insights into its learning mechanisms and capacity for generalization
  [Nat. Methods 21, 1514–1524 (2024)](https://doi.org/10.1038/s41592-024-02272-z)

- John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger, Kathryn Tunyasuvunakool, Russ Bates, Augustin Žídek, Anna Potapenko, Alex Bridgland, Clemens Meyer, Simon A. A. Kohl, Andrew J. Ballard, Andrew Cowie, Bernardino Romera-Paredes, Stanislav Nikolov, Rishub Jain, Jonas Adler, Trevor Back, Stig Petersen, David Reiman, Ellen Clancy, Michal Zielinski, Martin Steinegger, Michalina Pacholska, Tamas Berghammer, Sebastian Bodenstein, David Silver, Oriol Vinyals, Andrew W. Senior, Koray Kavukcuoglu, Pushmeet Kohli, Demis Hassabis
  Highly accurate protein structure prediction with AlphaFold**
  [Nature 596, 583–589 (2021)](https://doi.org/10.1038/s41586-021-03819-2)

- Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David K. Duvenaud
  Neural ordinary differential equations
  [Adv. NeurIPS 31 (2018)](https://proceedings.neurips.cc/paper/2018/file/69386f6bb1dfed68692a24c8686939b9-Paper.pdf)

- Schrödinger LLC
  The PyMOL molecular graphics system
  [PyMOL website](http://www.pymol.org)

- Helen M. Berman, John Westbrook, Zukang Feng, Gary Gilliland, T. N. Bhat, Helge Weissig, Ilya N. Shindyalov, Philip E. Bourne
  The protein data bank
  [Nucleic Acids Res. 28, 235–242 (2000)](https://doi.org/10.1093/nar/28.1.235)

- OpenFold
  OpenProteinSet (dataset registry)
  [AWS Open Data Registry](https://registry.opendata.aws/openfold/) (Accessed: 2025-08-05)

- Gustaf Ahdritz, Nazim Bouatta, Sachin Kadyan, Lukas Jarosch, Daniel Berenberg, Ian Fisk, Andrew M. Watkins, Stephen Ra, Richard Bonneau, Mohammed AlQuraishi
  OpenProteinSet: training data for structural biology at scale
  [NeurIPS 37, Article 204 (2023)](https://openreview.net/forum?id=gO0kS0eE0F&noteId=ly7X3fS4uJ)

