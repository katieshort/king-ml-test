# Game Design Recommender - King ML Tech Task

## Overview

This project analyses the impact of two game design versions (A and B) on player engagement and monetization in a mobile
game,
using a non-randomized/observational dataset of player behaviour. The primary goal is to determine which game design
maximizes player activity and in-game spending,
and to develop a personalized recommendation system for game design based on predictive modeling.

## Project Objectives

1. **Global Design Evaluation**: Identify which of the two designs (A or B) performs better overall in terms of player
   engagement and monetization.
2. **Personalized Design Recommendation**: Develop a predictive model to recommend a game design for each player that
   maximizes future monetization while maintaining or enhancing engagement.
3. **Performance Reporting**: Analyze and report the model performance to justify selection of approach and model.
4. **API Implementation**: Develop a locally runnable API that serves predictions from the trained model, containerized with Docker for cloud readiness.

## Methodology

The project is structured into distinct phases, each documented and aimed at addressing specific tasks

- **Exploratory Data Analysis (EDA)**: Conducts a thorough exploration of data quality and distribution, setting the
  foundation for the causal
  analysis and modelling approach. Detailed visualizations and insights are reported in
  the [Exploratory Data Analysis Notebook](/notebooks/EDA.ipynb).

- **Causal Inference**: Leverages insights from the EDA to apply causal inference methods aimed at identifying the more
  effective game design between A and B (Task 1).
  Complete analysis and findings can be found in
  the [Causal Inference Analysis Notebook](/notebooks/CausalAnalysis.ipynb).

- **Predictive Modeling**: The main codebase contains the implementation of a Doubly Robust Learning approach to predict
  individual treatment effects, enabling personalized game design recommendations. This contains the model training,
  evaluation and the API implementation (Tasks 2-4). The model approach is detailed in [/docs/ModelApproach.md](/docs/ModelApproach.md).

# Repository Structure and Usage

```plaintext
.
├── config                         # Configuration files
├── data/                          # Dataset directory
│   └── toydata_mltest.csv         # Main dataset
├── docs/                          # Documentation files
├── models/                        # Saved models
├── notebooks/                     # Jupyter notebooks for analysis
│   ├── EDA.ipynb                  # Exploratory Data Analysis notebook
│   └── Causal_Inference.ipynb     # Causal analysis notebook
├── reports                       
│   └── figures                    # Figures and graphics for reports
├── src/                           # Source code for the project
│   ├── data_processing/           # Data loading and preprocessing scripts
│   ├── models/                    # Scripts for model training and inference
│   ├── pipelines/                 # Pipeline utilities
│   └── utils/                     # General utilities
├── tests/                         # Test scripts
├── .dockerignore                  # Specifies what to ignore in Docker builds
├── api_demo.py                    # Demonstration script for API functionality
├── app.py                         # API implementation for serving predictions
├── Dockerfile                     # Docker configuration file
├── README.md                      # Overview and setup instructions (this file)
├── main.py                        # Main script for model training and 
├── poetry.lock                    # Poetry lock file for managing dependencies
├── poetry.toml                    # Poetry configuration file
└── pyproject.toml                 # Project configuration file
 ```

## Setup Instructions

### Prerequisites

- Python 3.12
- Docker
- Poetry (for Python dependency management)

### Installation

To install and run the project, execute the following steps:

```bash
# Clone the repository
git clone [Your-Repository-URL]
cd [Repository-Name]

# Install dependencies using poetry
poetry install
```

### Running the Model

To train the models and evaluate performance, execute

```
python main.py
```

### Using the API

Start the API server by running:

```
python app.py
```

Example API call:

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '[{"id": "33553", "c1": "VFc", "c2": "aW9z", "c3": "cHQ", "c4": "aHVhd", "c5": true, "c6": "LTAzOjAw", "n1": 303.488539, "n2": 1085.434206, "n3": 90.097357, "n4": 21.113713, "n5": 404.121039, "n6": 3859250336.874662, "n7": 10.312303, "n8": 1112.885066, "n10": 0.409991, "n11": 3.289646, "n12": 0.08629}]'
````

or run the API demo script:

```
python api_demo.py
```

### Setting Up with Docker

1. Build the Docker image:

```bash
docker build -t mymodelapi .
```

2. Run the Docker container:

```bash
docker run -p 5000:5000 mymodelapi
```

3. Test the API

```bash
docker exec -it <container_name> python api_demo.py
```
