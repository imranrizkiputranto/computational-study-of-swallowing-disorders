# Computational Study of Swallowing Disorders
Third Year Individual Research Project for the University of Bristol About the Computational Study of Swallowing Disorders:

This study investigates the application of Smoothed Particle Hydrodynamics (SPH) to simulate the swallowing process, which is of significant interest in understanding and diagnosing dysphagia. This research focuses on the development and validation of a two-dimensional model for simulating bolus flow and its interaction with anatomical structures during swallowing. The model utilised the Weakly Compressible SPH (WCSPH) approach and incorporated boundary conditions derived from two-dimensional videofluoroscopic swallowing study (VFSS) data. The code was implemented in Python using the NumPy, Matplotlib and Scikit-Learn libraries, and GPU acceleration was achieved using the Numba library. The code was validated by comparing the results of one-dimensional function approximations and a two-dimensional dam break problem against numerical and experimental data available in liter- ature. The results of the two-dimensional swallowing simulation revealed the ability of SPH to capture the overall bolus flow during the swallowing process. However, the simulation exposed some limitations, such as slight fluid compression and challenges in conserving the bolus surface area. Due to the swallowing process inherently being a three-dimensional process with complex interactions with surrounding tissue, several assumptions made led to relatively inaccurate results. Despite these limitations, this study demonstrates the potential of SPH in modelling the swallowing process and lays the groundwork for future research involving more advanced and accurate three-dimensional incompressible models and potential medical applications to aid clinicians with the diagnosis and treatment of dysphagia.

