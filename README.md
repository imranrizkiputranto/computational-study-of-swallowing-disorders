# Computational Study of Swallowing Disorders
This repository contains the code and accompanying report third-year Individual Research Project for the University of Bristol About the Computational Study of Swallowing Disorders. The project investigates the application of Smoothed Particle Hydrodynamics (SPH) to understand and diagnose dysphagia, a swallowing disorder.

## Abstract
This study investigates the application of Smoothed Particle Hydrodynamics (SPH) to simulate the swallowing process, which is of significant interest in understanding and diagnosing dysphagia. This research focuses on the development and validation of a two-dimensional model for simulating bolus flow and its interaction with anatomical structures during swallowing. The model utilised the Weakly Compressible SPH (WCSPH) approach and incorporated boundary conditions derived from two-dimensional videofluoroscopic swallowing study (VFSS) data. The code was implemented in Python using the NumPy, Matplotlib and Scikit-Learn libraries, and GPU acceleration was achieved using the Numba library. The code was validated by comparing the results of one-dimensional function approximations and a two-dimensional dam break problem against numerical and experimental data available in literature. The results of the two-dimensional swallowing simulation revealed the ability of SPH to capture the overall bolus flow during the swallowing process. However, the simulation exposed some limitations, such as slight fluid compression and challenges in conserving the bolus surface area. Due to the swallowing process inherently being a three-dimensional process with complex interactions with surrounding tissue, several assumptions made led to relatively inaccurate results. Despite these limitations, this study demonstrates the potential of SPH in modelling the swallowing process and lays the groundwork for future research involving more advanced and accurate three-dimensional incompressible models and potential medical applications to aid clinicians with the diagnosis and treatment of dysphagia.

## Contents
- 1D Function Approximation: Code for approximating a one-dimensional function.
- 2D Function Approximation: Code for approximating a two-dimensional function.
- 2D Dam Break Problem: Code for simulating a two-dimensional dam break problem.
- Swallowing Simulation: Code for simulating the swallowing process using SPH.

## Report
The repository also includes the full 20-page report of the research project. The report provides an in-depth analysis of the methodology, implementation details, validation process, results, and discussions. It serves as a comprehensive document detailing the research findings and conclusions.

## Usage
Feel free to explore the code and report in this repository for a deeper understanding of the SPH-based swallowing simulations. Contributions from the community to enhance the accuracy and efficiency of the swallowing simulation are welcome. If you have any suggestions, please feel free to submit a pull request or reach out through imranrizkiputranto@gmail.com.

Please note that the code and report are intended for educational and research purposes only. Proper attribution should be given if you use or refer to any part of this work in your own projects or research.

## Acknowledgments
I would like to express my deepest gratitude to Dr. Alberto Gambaruto from the University of Bristol for all his support and supervision over the course of this project, as well as Simon Zeng for his time discussing the smoothed particle hydrodynamics methodology and providing the necessary medical data. This endeavour would not have been possible without their support and insight.
