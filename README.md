# Statistics for Earth Scientists

A self-taught module on statistics and data analysis for the Part II Earth Sciences course. It consists of eight Jupyter notebooks that can be run in your browser, with nothing to install, by clicking the **Open in Colab** links below. Each notebook takes roughly 30 to 45 minutes to work through, and ends with a few exercises whose solutions are hidden behind a *click to expand* link.

The emphasis is on ideas and on knowing which tool to reach for. The programming is kept simple, and almost everything is done with the `scipy.stats` library, which you can then use in your own work. The notebooks use a few small real data sets (geyser eruptions, atmospheric CO₂, global temperature, an earthquake catalogue) purely as illustrations.

## The notebooks

| | Notebook | What it covers |
|---|---|---|
| 1 | [Probability and describing data](notebooks/01_probability_and_data.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/PartIIStatistics/blob/main/notebooks/01_probability_and_data.ipynb) | Random variables, PDFs and CDFs, the common distributions, histograms, summary statistics, fitting a distribution to data |
| 2 | [Averaging and the uncertainty of an estimate](notebooks/02_averaging_and_uncertainty.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/PartIIStatistics/blob/main/notebooks/02_averaging_and_uncertainty.ipynb) | Sampling distributions, the standard error, the central limit theorem, confidence intervals and the *t*-distribution, weighted means, the bootstrap |
| 3 | [Propagation of uncertainty](notebooks/03_propagation_of_uncertainty.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/PartIIStatistics/blob/main/notebooks/03_propagation_of_uncertainty.ipynb) | Monte Carlo and linearised error propagation, covariance and correlation, when linearisation fails |
| 4 | [Hypothesis testing](notebooks/04_hypothesis_testing.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/PartIIStatistics/blob/main/notebooks/04_hypothesis_testing.ipynb) | p-values, Type I and II errors, power, the chi-squared test of a model, the standard `scipy.stats` tests, misuses of p-values |
| 5 | [Fitting models to data](notebooks/05_fitting_models.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/PartIIStatistics/blob/main/notebooks/05_fitting_models.ipynb) | Maximum likelihood and least squares, `curve_fit`, parameter uncertainties and confidence regions, goodness of fit, the Gutenberg–Richter *b*-value |
| 6 | [Correlation and regression](notebooks/06_correlation_and_regression.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/PartIIStatistics/blob/main/notebooks/06_correlation_and_regression.ipynb) | Correlation coefficients, rank correlation, Anscombe's quartet, spurious correlation, regression to the mean, errors in *x* |
| 7 | [Bayesian inference](notebooks/07_bayesian_inference.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/PartIIStatistics/blob/main/notebooks/07_bayesian_inference.ipynb) | Bayes' theorem, priors and posteriors, credible intervals, posteriors on a grid, a first Markov chain Monte Carlo sampler |
| 8 | [Resampling and Monte Carlo methods](notebooks/08_resampling_and_monte_carlo.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/PartIIStatistics/blob/main/notebooks/08_resampling_and_monte_carlo.ipynb) | Simulation, the bootstrap and its limits, permutation tests, cross-validation |
| 9 | [Inverse problems](notebooks/09_inverse_problems.ipynb) *(optional)* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/PartIIStatistics/blob/main/notebooks/09_inverse_problems.ipynb) | Non-uniqueness and the null space, minimum-norm and damped solutions, resolution, the Bayesian view, the same tomography in function space with `pygeoinf` |

Work through them in order: later notebooks use ideas and data from earlier ones. Notebook 9 is an optional extra, a little more advanced than the rest, on the inverse problems that arise throughout geophysics.

## Running in Colab

Click a badge above. Colab opens the notebook on Google's servers, with all the required libraries already installed. Run a cell with Shift+Enter, and run the cells in order from the top. Nothing you do is saved unless you choose *File → Save a copy in Drive*, which you may want to do if you work on the exercises.

## Running on your own computer

If you prefer to run the notebooks locally, clone or download the repository and install the requirements:

```
git clone https://github.com/da380/PartIIStatistics.git
cd PartIIStatistics
pip install -r requirements.txt
jupyter lab
```

The notebooks read their data files from GitHub, so an internet connection is needed the first time each data set is loaded.

## Data

The small data sets used in the notebooks are in the [`data`](data) folder, with their sources listed in [`data/README.md`](data/README.md).

## For the curious

The notebooks are generated from source files in the [`build`](build) folder, which also contains the script that produced the data files. You do not need any of this to use the course.

## Q&A session

There will be a Q&A session on the material on **Friday 20th November, 12–1 pm, in Harker 1**, where you can ask about anything in the notebooks.

## Issues or feedback

If you have any problems running the notebooks, spot a mistake, or have questions about the material, please let me know.
