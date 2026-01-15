
<p align="center">
  <img src="docs/assets/MitoMimics_Logo.png" alt="MitoMimics Logo">
</p>

<p align="center">
  <em>
    Synthetic Microscopy Timelapse Data Generation for Zero-Annotation AI Mitochondrial Segmentation and Tracking
  </em>
</p>

---


## Introduction

MitoMimics is designed to simulate temporal mitochondrial dynamics-based microscopy data. MitoMimics employs a Python-based rendering pipeline to generate synthetic widefield microscopy datasets of mitochondrial dynamics events. Our algorithmic toolset provides researchers a means for testing and validating AI-based models, providing both ease of use and flexibility.

## Features

- Simulates mitochondrial motility, fission, and fusion in 2D space.
- Generates synthetic time-lapse microscopy data.
- Generates synthetic instance masks along with a JSON file that registers instances between frames, which can be used to validate segmentation and/or tracking models.
- Allows customization of simulation parameters, such as:
  - Mitochondrial Density
  - Mitochondrial Speed
  - Fission and Fusion Rates
  - Simulation Time Scale
  - Dimensions of the Simulation Space
- Allows customization of microscopy parameters, such as:
  - Point Spread Function (2D Gaussian PSF)
  - Noise (Background Offset, Camera Readout Noise, Poisson Noise)

## Installation

To install MitoMimics, follow these steps:

1. Clone the repository: `git clone https://github.com/aidanpcquinn/mito_sim_pack.git`
2. Navigate to the project directory: `cd mito_sim_pack`
3. Install the required dependencies: `conda env create -f environment.yml`

## Using the Simulation

To use MitoMimics, follow these steps:

1. Open a terminal activate the conda environment with `conda activate mito_sim`
2. Run the simulation script: `python sim.py`
3. Specify simulation parameters by modifying the `sim.py` file directly.
   - JSON file configuration will be added soon.
5. Wait for the simulation to complete.
   - The user can vary the timescale indefinitely by changing the `save_data` or `sim_length_seconds` variables.
6. On simulation completion, generated data will automatically save in the output directory: `./out_stack`.

## Viewing Saved Data

1. See `view_output.ipynb`
2. Demo output at `./out_stack/test_state_output.pkl`

## Citation

## License

## Contact
For any questions, feedback, or potential collaborations, please contact the original authors of MitoMimics: 

- **Aidan Quinn** 

  📧 [aidanpcquinn@gmail.com](mailto:aidanpcquinn@gmail.com)

  🐙 Github: [@aidanpcquinn](https://github.com/aidanpcquinn)

- **Volkan Ozcoban** 

  📧 [volkanozcoban1@gmail.com](mailto:volkanozcoban1@gmail.com)

  🐙 Github: [@VolkanOzcoban](https://github.com/VolkanOzcoban)

Please include “MitoMimics” in the subject line when contacting us.