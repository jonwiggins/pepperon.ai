# [pepperon.ai](https://pepperon.ai)

pepperon.ai is an open-source Python toolset for machine learning and data science and is distributed under the MIT License (Please don't sue me).

This project was started in 2019 by @JonWiggins as a central repository for some of the fun algorithms from various courses at the University of Utah. 

It is intended to be a lot like scikit-learn, except more buggy, with less functionality, worse documentation, and not used by anyone. But that's okay, because making it will be fun.

## Motivation
pepperon.ai is built on a few core ideas:

1. Machine Learning should be accessable to the masses
2. Bugs should be more common in software packages
3. Cython is for posers, python3 is the future
4. Runtime, much like digging holes, builds character

## Version
pepperon.ai is currently on version 0.124

## Installation

### Requirements
pepperon.ai requires:
- Python (>= 3.5)
- NumPy (>= 1.11.0)
- A lot of Patience
- Scipy (>= 0.11)
- Pandas (>= 0.24)

### User Installation
There is no `pip` or `conda` install, just yoink the file you want from this repo and paste it into your project.

# Modules
## Models
### Decision Trees
- Decision Trees created with ID3 ; for all your decison needs
### Perceptron
- Simple Perceptron ; for all your needs that are both linearly seperable and basic
- Average Perceptron ; for all your needs that are both linearly seperable and noisy
### SVM
- SVM on SGD ; for when you are using Perceptron, and decide you want it to be better
### Niave Bayes
- Guassian and Bernoulli
### Random Forest
- Random Forest on ID3
## Language
### Ngrams
- ngrams ; for all your simplistic corpus needs 
### Kgrams
- kgrams ; for all your shingling needs
## Data
### Misa Gries
- For creating a heavy hitters lower bound
### Count Min Sketch
- For creating a heavy hitters upper bound
### Utils
- *In progress, do not attempt to use*
- Grid Search for finding the best hyperparameters for your model
- Jackknifing to create cross validation folds
- General testing and evaluating methods
## Cluster
### kmeans++
Clusters based on the kmeans++ algorithm, all based on probability
### Gonzales
Clusters based on the Greedy Golzales algorithm, iteratively picks the furthest point from the existing clusters to be a new center.
### Heirarchical Clustering
Rather than basing off of clusters, Heirarchial methods iteratively merge the two nearest clusters. What defines *near* is based on the linking method given, the built in linking functions provided are:
- Single link: finds the smallest distance between two points in clusters
- Complete link: find the largest distance between two points in clusters
- Mean link: finds the average distance between two points in clusters
### Utils
- Fowlkes-Mallows Index ; for comparing clusterings;
- Purity Index ; for comparing clusterings
- Various distance and similarity functions

## Usage Examples
Maybe one day I will make some files that show off how to go about using these systems

## To do
- Add examples usage files

# Contact
Feel free to send all comments, questions, or concerns to [contact@pepperon.ai](mailto:pepperon.ai)

