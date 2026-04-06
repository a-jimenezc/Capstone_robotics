
# Capstone Project

## Overview
This repository contains the code for the capstone project title **Model-based Extended Kalman Filter vs Learned KalmanNet for Parameter Estimation in the Morris-Lecar Neuron Model**

## Running the project from a Conda environment

```bash
conda create -n capstone_env python=3.11
conda activate capstone_env
pip install -r requirements.txt
```

Then, install PyTorch. Instructions for specific systems are available at [pytorch.org/](https://pytorch.org/).

## How to repeat the analysis

Run the main scripts from the project root directory.

#### Generate ground-truth plots

```bash
python main_gt_plot.py
```

#### Run EKF analysis

```bash
python main_plot_generation_ekf.py
```

#### Run KalmanNet analysis

```bash
python main_plot_generation_kalmannet.py
```

#### Run UKF analysis

```bash
python main_plot_generation_ukf.py
```


After running the scripts, check the output folders `results/` which contains the estimation results and comparison plots.

## References

[1] Matthew J. Moye and Casey O. Diekman. Data assimilation methods for neuronal state and
parameter estimation. Journal of Mathematical Neuroscience, 8(1):11, August 2018. doi:
10.1186/s13408-018-0066-8.

[2] Eugene M. Izhikevich. Dynamical Systems in Neuroscience: The Geometry of Excitability
and Bursting. The MIT Press, 07 2006. ISBN 9780262276078. doi: 10.7551/mitpress/2526.
001.0001. URL https://doi.org/10.7551/mitpress/2526.001.0001.

[3] Mark Asch, Marc Bocquet, and Ma¨elle Nodet. Data Assimilation: Methods, Al-
gorithms, and Applications. Society for Industrial and Applied Mathematics, 2016.
doi: 10.1137/1.9781611974546. URL https://epubs.siam.org/doi/abs/10.1137/
1.9781611974546.fm.

[4] G. Bard Ermentrout and David H. Terman. Mathematical Foundations of Neuroscience,
volume 35 of Interdisciplinary Applied Mathematics. Springer, New York, 2010. doi:
10.1007/978-0-387-87708-2.

[5] Guy Revach, Nir Shlezinger, Xiaoyong Ni, Adria Lopez Escoriza, Ruud J. G. van Sloun,
and Yonina C. Eldar. Kalmannet: Neural network aided kalman filtering for partially known
dynamics. IEEE Transactions on Signal Processing, 70:1532–1547, 2022. ISSN 1941-
0476. doi: 10.1109/tsp.2022.3158588. URL http://dx.doi.org/10.1109/TSP.2022.
3158588.

[6] Gabriel W. Vattendahl Vidal, Mathew L. Rynes, Zachary Kelliher, and Shikha Jain Good-
win. Review of brain-machine interfaces used in neural prosthetics with new perspective on
somatosensory feedback through method of signal breakdown. Scientifica, 2016:8956432,
2016. doi: 10.1155/2016/8956432.

[7] Ethan Sorrell, Michael E. Rule, and Timothy O’Leary. Brain–machine interfaces: Closed-
loop control in an adaptive system. Annual Review of Control, Robotics, and Autonomous
Systems, 4:167–189, 2021. doi: 10.1146/annurev-control-061720-012348.

[8] Luis H. Cubillos, Guy Revach, Matthew J. Mender, Joseph T. Costello, Hisham Temmar, Aren
Hite, Diksha Zutshi, Dylan M. Wallace, Xiaoyong Ni, Madison M. Kelberman, Matthew S.
Willsey, Ruud J. G. van Sloun, Nir Shlezinger, Parag Patil, Anne Draelos, and Cynthia A.
Chestek. Exploring the trade-off between deep-learning and explainable models for brain-
machine interfaces. Advances in Neural Information Processing Systems, 37:133975–133998,
2024.

[9] C. Morris and H. Lecar. Voltage oscillations in the barnacle giant muscle fiber. Biophysical
Journal, 35(1):193–213, July 1981. doi: 10.1016/S0006-3495(81)84782-0.

[10] Gregory Plett. Applied kalman filtering specialization. Coursera, University of Colorado Sys-
tem. [Online]. Available: https://www.coursera.org/specializations/kalman-filtering-applied.
Accessed: Mar. 10, 2026.

[11] Dan Simon. Optimal State Estimation: Kalman, H-Infinity, and Nonlinear Approaches. Wiley,
2006. ISBN 9780471708582. doi: 10.1002/0470045345.

[12] Yehonatan Dahan, Guy Revach, Jindrich Dunik, and Nir Shlezinger. Bayesian kalmannet:
Quantifying uncertainty in deep learning augmented kalman filter, 2025. URL https:
//arxiv.org/abs/2309.03058.
