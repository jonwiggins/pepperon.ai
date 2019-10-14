# [pepperon.ai](pepperon.ai)

pepperon.ai is an open-source Python toolset for machine learning and data science and is distributed under the MIT License.

This project was started in 2019 by @JonWiggins as a central repository for some of the fun algorithms from various courses at the University of Utah. 


It is intended to be a lot like scikit-learn, except more buggy, with less functionality, worse documentation, and not used by anyone. But that's okay, because making it will be fun.

pepperon.ai is also developed entirely in python3, _which is the future_.

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
There's no 'pip' or 'conda' install, just yoink the file you want from this repo and paste it into your project.

# Modules
## Models
### Decision Trees
- Decision Trees created with ID3 ; for all your decison needs
### Perceptron
- Simple Perceptron ; for all your needs that are both linearly seperable and basic
- Average Perceptron ; for all your needs that are both linearly seperable and noisy
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
- Coming Soon: Unlabeled clustering method
### Gonzales
- Coming Soon: Unlabeled clustering method
### Utils
- Fowlkes-Mallows Index ; for comparing clusterings;
- Purity Index ; for comparing clusterings

## Usage Examples
Maybe one day I will make some files that show off how to go about using these systems

## To do
- Add examples usage files
- Add streaming algorithms Misa Greis and Count Min Sketch
- Add clustering algorithms gonzales, lloyds, mean link, single link, and complete link

# Contact
Feel free to send all comments, questions, or concerns to [contact@pepperon.ai](mailto:pepperon.ai)

