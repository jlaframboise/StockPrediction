# Climate Search Popularity for Stock Trend Prediction

## Goal
We aim to predict the stock prices of energy companies, and to see if using Google Search Trends for the search popularity of key climate related terms will help us do so. 

## Overview of Repository
### /data
The search popularity for climate keywords was generated using pytrends (generate_pytrends_data.ipynb), and the results were stored in the /data folder with one csv per term. 

### experiments notebooks
To answer the question of whether climate search trends can be used to improve a model, accross 3 sectors of interest, we have two notebooks per sector.  
notebook A in each sector runs through a 3 stage transfer learning training and evaluation of a model *with* climate trends, and notebook B is an identical process except the model does not use climate trends. 

For example, to see if climate search trends aid in trend classification for electric vehicle and battery stocks we selected, compare the results from sections 4.2 and 5.2 in experiment-EV-A.ipynb with the results in the same sections of experiment-EV-B.ipynb. 

### Other important files
- indicators.py: Contains functions implementing our technical indicators. 
- data_loading.py: Contains functions that abstract the loading of data from yahoo finance and climate files to clean up notebooks and improve reusability of code. 
- util_functions.py: A very important set of functions, which handles tasks like normalizing the data per stock, splitting the data for notebooks that use a model with multiple inputs, and a very important function called generate_dataset which abstracts out a lot of work preparing a dataset with given stocks and features and made it possible to have streamlined experiment notebooks with multiple stages of transfer learning.


## Reproducibility
### Dependancies
To run the experiments, you will need a python environment with the following dependancies:
- tensorflow >= 2
- Python3
- matplotlib
- seaborn
- numpy
- scikit-learn
- pydot
- graphvis
- jupyter notebook
- yfinance
- pandas

### Steps
1. Open a terminal (or cmd) window in the main top level folder of the repository
2. Make sure your environment is activated
3. run the command: "jupyter notebook"
4. Click to open an experiment notebook, like "experiment-traditional-oil-A.ipynb"
5. Click cell > run all

### Noise in results
It should be noted that stock trend classification is a very difficult task, and there was a lot of noise in the performance of our models. Since we are using a GPU to train, there is unavoidable variation in the output because order of operations is not guaranteed.  
To get our results, we made sure the notebooks were running well, then we decided when we would run it for the final time and took those results.   
Your results may vary. 


