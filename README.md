# Group 3 - Project 2 - Machine Learning
## Analyzing & Predicting Housing Data
This project aims to analyze and predict house prices in various cities using machine learning techniques. The dataset includes city housing prices data (Atlanta, Charlotte, Cincinnati, Denver, Indianapolis, Providence, and Tampa) from Zillow, Education data from National Center for Educations Statistics (NCES), and mortgage rates from Federal Reserve Econimic Data (FRED), with the goal to develop a predictive model that can accurately estimate future home prices.

## Presentation:
https://docs.google.com/presentation/d/173BV_MnqTNAOLiIGF60CaWKpuU6meASNKPr83_WyCG8/edit#slide=id.g28768449b51_0_6

## GitHub:
https://github.com/JimGile/group3-project2-ml.git

## Clone Project
git clone https://github.com/timwillard24/house-prices-prediction.git

## Requirements - following must be installed
- matplotlib
- numpy
- pandas
- statsmodels
- seaborn
- scikit-learn
- time

## Notebooks & Scripts
* all_city_compiler.ipynb
* bls_data_cleanup.ipynb
* city_house_prices_modeling.ipynb
* fred_data_additional_cleanup.ipynb
* nces_data_cleanup
* prophet_model_all_cities.ipynb
* sale_price_city_cleanup.ipynb
* supervised_model_pipeline.py
* zillow_metro_data_cleanup.ipynb
* zillow_metro_data_combiner.ipynb

## Data Files
* Atlanta_combined.csv
* Charlotte_combined.csv
* Cincinnati_combined.csv
* combined_cities.csv
* Denver_combined.csv
* Indianapolis_combined.csv
* Tampa_combined.csv

# Main
## city_house_prices_modeling.ipynb
This is the primary modeling notebook that utilizes data extraction, cleaning, transformation, and building a machine learning models to predict housing prices.  It can be used to analyze the indvidual and combined city CSV files and was used to predict 'MeanSalePrice' and compare against the actual 'MeanSalePrice' for individual cities and the combined file. 

Features:
 - Load and inspect data
 - Remove columns that were too predictive
 - Utilize the EdaToolbox (from supervised_model_pipeline.py) for data clean-up
 - Apply proper data transformations
 - Evaluate RSME for various models (Linear Regression, Ridge Regression, Lasso Regression, ElasticNet Regression, Lasso CV, ElasticNetCV, RidgeCVm and HistGradientBoost)
 - Set the target and feature column transformers and prepare for feature selection
 - Run and plot feature analysis 
 - Perform unsupervised feature selection
 - Final EDA with selected features
 - Check feature analysis
 - Model for multiple regression models
 - Utilize best model 
 - Plot Predictions vs actuals

## supervised_model_pipeline.py 
The script provides a framework for building a supervised learning model pipeline, primarily focusing on preprocessing, feature selection, model training, and evaluation. It aims to streamline the process of model creation while offering flexibility for different types of supervised learning tasks, such as regression or classification.

Key Components:
1. Libraries and Imports:
    - It uses pandas, numpy, and matplotlib for data manipulation and visualization.
    - sklearn is used for machine learning tasks like model selection, feature scaling, and pipeline creation.
    - statsmodels is used for statistical modeling and checking multicollinearity using the variance inflation factor (VIF).
2. Custom Transformer Class (SelectedFeaturesTransformer):
    - This class is designed to select specific features from a DataFrame based on a given list of features.
    - It implements methods like fit, transform, and fit_transform, making it compatible with scikit-learn's pipeline functionality.
    - This transformer allows users to control which features are passed into the model.
3. Feature Engineering:
    - There is a utility function (feature_name_combiner) to combine feature names and categories, which is useful for encoding categorical features and working with feature transformation pipelines.
4. Model Selection:
    - The script appears to support different model selection strategies, including cross-validation (cross_val_score, KFold) and recursive feature elimination (RFECV, SequentialFeatureSelector).
5. Evaluation Metrics:
    - mean_squared_error is used for regression tasks.
    - roc_auc_score may be employed for binary classification tasks.
6. Additional Preprocessing:
    - Preprocessing steps, such as imputation, scaling (using StandardScaler), and encoding (with OneHotEncoder, OrdinalEncoder), are included in the script.
    - PCA is included for dimensionality reduction.

Features:
- Users can import this script and use it as part of their machine learning projects by utilizing the custom transformer and pipeline structure.
- The script supports building scikit-learn pipelines using the Pipeline and ColumnTransformer structures, where users can chain feature transformations, model training, and evaluation steps.
- Users can specify a list of selected features when initializing the SelectedFeaturesTransformer to control which columns are used for training.
- The user can use different scikit-learn models like LinearRegression, RidgeCV, and LogisticRegression with built-in feature selection techniques.
- The script supports cross-validation and scoring techniques, making it easy to evaluate models using a variety of metrics.

Future Expansion:
This pipeline can be extended with additional models, feature selection techniques, or custom scoring functions to adapt to more complex machine learning tasks.

# Ancillary
1. all_city_compiler.ipynb: 
Compiles and merges city-level data from various sources into a unified dataset for further analysis.
2. bls_data_cleanup.ipynb: 
Cleans and prepares data from the Bureau of Labor Statistics (BLS) for integration with other datasets in the project.
3. fred_data_additional_cleanup.ipynb: 
Processes additional economic data from the Federal Reserve Economic Data (FRED), focusing on ensuring consistency and preparing it for modeling.
4. nces_data_cleanup.ipynb: 
Cleans and formats educational data from the National Center for Education Statistics (NCES) for inclusion in the overall analysis.
5. prophet_model_all_cities.ipynb:
Provides a prediction model for 12 months pricing into the future for each individual city
6. sale_price_city_cleanup.ipynb: 
Focuses on cleaning and organizing city-level housing price data for predictive modeling and analysis.
7. zillow_new_metro_data_cleanup.ipynb: 
Cleans and organizes new metro-level housing price data sourced from Zillow.
8. zillow_new_metro_data_combiner.ipynb: 
Combines various metro-level datasets from Zillow into a single dataset for streamlined analysis.

# Results
![Model and Transformer effectiveness ](image.png)
![Atlanta Predictions](image-1.png)

# Conclusion
Based on the modeling we did we came to the conclusion that Atlanta would be the best place to purchase a new home.  It had steady increase in home values over the years and should continue to see increasing home values over the years to come.  Both the HistGradientBoost and Prophet models were used to come to this conclusion.

# Contributors
Anna Fine, Carl Peterson, Jim Gile, Tim Willard
