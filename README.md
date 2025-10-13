# Using Behavioural Data to support the Hypothesis

### Prerequisites

To run the notebooks, please run in the terminal 
```
conda env create -f environment.yml
```

to install dependencies.

### Methodology

1. Find open-source datasets to take the data from
    - Check for completeness of the datasets
    - Carefully read the documentation
    - Pick 3-4 datasets that are fitting and do a comparative analysis
2. Preprocessing stage
    - Carefully look through the datasets and complete / cut them if needed (i.e. put N/A for empty columns)
3. Feature Engineering 
    - Extract features you find useful for a later-on Machine Learning Analysis
    - Build it as a separate module, avoid redundancy
4. Machine Learning
    - Pick the model you will use (depending on the size of the dataset, specific paradigm etc)
    - Complete a pipeline in Jupyter notebook
    - Please ensure everything is reproducible
5. Analysis
    - Do visualisations / plots for results
    - Discussion on empirical meaning behind the above-mentioned results