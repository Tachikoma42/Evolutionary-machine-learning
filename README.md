# Evolutionary-machine-learning

This repository contains various implementations of evolutionary algorithms for machine learning. The algorithms are implemented in Julia.

## Introduction

Evolutionary algorithms are a class of optimization algorithms that are inspired by the process of natural selection. They are often used to solve optimization problems, but can also be used to solve machine learning problems. The algorithms are often used in combination with other machine learning algorithms, such as neural networks.


## Dataset
The dataset used in this project is a basketball player stat dataset. The goal is to predict 5-Year Career Longevity for NBA Rookies. The y = 0 if career years played < 5 and y = 1 if career years played >= 5. 

The dataset can be downloaded from [here](https://data.world/exercises/logistic-regression-exercise-1).

## Models

The model used in this project is a simple neural network with 2 hidden layers. The model is implemented using Flux package in Julia.
Various evolutionary algorithms are used to train the model.

## Projects

The following algorithm are implemented in this repository:

- [x] [Genetic Algorithm]()
- [x] [Evolution Strategies]()
- [x] [Particle Swarm Optimization]()
- [x] [Learning classifier system]()
- [x] [Artificial bee colony algorithm]()


## Environment
- Windows 10
- Julia 1.7.1
- Flux 0.11.1


## Usefull links
- [Evolutionary algorithms](https://en.wikipedia.org/wiki/Evolutionary_algorithm)
- [Flux](https://fluxml.ai/Flux.jl/stable/)
- [Julia](https://julialang.org/)



## Books

- [Evolutionary Deep Learning](https://www.amazon.com/Evolutionary-Deep-Learning-algorithms-networks/dp/1617299529)
- [Evolutionary Computation in Julia](https://www.amazon.com/Evolutionary-Computation-Julia-Practical-Applications/dp/1484262025)