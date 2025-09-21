# Bayesian Analysis of Proton Structure

A Python code for performing Bayesian analysis of proton structure parameters using a dipole QCD model and high energy electron-proton collision data. While the code was developed for this specific physics problem, most functions are written in a general way, allowing adaptation to other datasets and models.

I somewhat followed jbernhard's example https://github.com/jbernhard/hic-param-est/tree/master. The detailed description of the theory behind the code is available in my master's thesis: https://jyx.jyu.fi/jyx/Record/jyx_123456789_99785#

**This project is no longer maintained.** 

## Description
The goal of this project was to extract posterior distributions for proton structure parameters by fitting a mathematical model of the proton to high-energy electron-proton collision data using machine learning techniques:

1. **Training data** is calculated from the mathematical model of the proton.  
2. A **machine learning emulator** (via scikit-learn) learns the model.  
3. Bayesian inference is performed to find the posterior distributions of the model parameters that best fit the experimental data.  

The approach was successful and demonstrates a general pipeline for Bayesian parameter inference with emulators.

## Requirements
Listed in `requirements.txt`

## Usage
The code does not provide precompiled executables. You need to modify and run the Python scripts yourself. Most modifications should only be required in the `main` files or as new `main` file,
but its better to look into general stucture too.

### Recommended Function
I recommend to try the `make_posterior` function found in `bayesian_analysis_functions.py`. I tried to make it a more general and easier way to utilize this code for 
the posterior creations, but I cannot promise full functionality and bug free implementation yet.

```python
from bayesian_analysis_functions import make_posterior

samples = make_posterior(
    training_parameters,
    training_data,
    testing_parameters,
    testing_data,
    experimental_data,
    experimental_cov,
    parameters_limits,
    parameters_labels,
    MCMC_walkers=20,
    MCMC_steps=100,
    ...
)
```
- This function automates posterior creation.
- Can handle multiple datasets.
- Allows tuning of MCMC and emulator parameters via `kwargs`.
- Supports saving and loading emulators and MCMC samples.
- See the function docstring for full parameter details.
 
#### General Usage
- Prepare your training data (from a model or simulated data).
- Provide testing data and experimental data to fit.
- Call `make_posterior` to obtain posterior distributions.
- Optionally, use plotting functions to visualize the results.

## Notes
- The code is general-purpose; you can adapt it to different models and datasets by providing new training/testing/experimental data and parameter settings.
- Some functions might require minor debugging or tuning.

## TODO
- Bug testing and improved `kwargs` functionality for `make_posterior`.
- Enhanced plotting and automatic figure generation.
- More generality development
