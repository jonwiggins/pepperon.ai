# pepperon.ai

pepperon.ai is an open-source Python toolset for machine learning and data science and is distributed under the MIT License.

This project was started in 2019 by @JonWiggins as a central repository for some of the fun algorithms from various courses at the University of Utah. 


It is intended to be a lot like scikit-learn, except more buggy, with less functionality, worse documentation, and not used by anyone. But that's okay, because making it will be fun.

## Version
pepperon.ai is currently on version 0.123.4.0

## Installation

### Requirements
pepperon.ai requires:
- Python (>= 3.5)
- NumPy (>= 1.11.0)
- A lot of Patience
- Scipy (>= 0.11)
- Pandas (>= 0.24)

### User Installation
There's no 'pip' or 'conda' install, just yoink the file you want from this repo and paste it into your project.


## Files

### Cluster
- kmeans++ ; for all you unlabeled data needs
- Fowlkes-Mallows Index ; for comparing clusterings;
- Purity Index ; for comparing clusterings

### Decision Trees
- ID3 ; for all your decison needs

### n-grams
- ngrams ; for all your simplistic corpus needs 
- kgrams ; for all your shingling needs

### Perceptron
- Simple Perceptron ; for all your needs that are both linearly seperable and basic
- Average Perceptron ; for all your needs that are both linearly seperable and noisy

### Data Set Utils
- *In progress, do not attempt to use*
- Will contain grid search, jackknifing, and model evaluation tools

## Usage Examples
Maybe one day I will make some files that show off how to go about using these systems

## To do
- Add examples usage files
- Add streaming algorithms Misa Greis and Count Min Sketch
- Add clustering algorithms gonzales, lloyds, mean link, single link, and complete link

