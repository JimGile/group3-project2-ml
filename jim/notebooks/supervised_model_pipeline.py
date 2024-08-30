import pandas as pd
import numpy as np
import statsmodels.api as sm
import time
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, RegressorMixin
from sklearn.linear_model import RidgeCV, LinearRegression
# accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.outliers_influence import variance_inflation_factor


def feature_name_combiner(feature_name: str, category: str) -> str:
    return f"{feature_name}__{category}"


class SelectedFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, selected_feature_list: Optional[list[str]] | None):
        self.selected_feature_list: list[str] = selected_feature_list

    def fit(self, X: pd.DataFrame, y=None):
        if self.selected_feature_list is None:
            self.x_selected_features_df = X
        else:
            selected_cols = [col for col in X.columns if col in self.selected_feature_list]
            self.x_selected_features_df = X[selected_cols]
        return self

    def transform(self, X) -> pd.DataFrame:
        return self.x_selected_features_df

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.x_selected_features_df.columns


class SimplePruner(BaseEstimator, TransformerMixin):
    def __init__(self,
                 infrequent_threshold: float,
                 pct_missing_threshold: float):
        self.infrequent_threshold = infrequent_threshold
        self.pct_missing_threshold = pct_missing_threshold

    def fit(self, X, y=None):
        x_df = pd.DataFrame(X)
        self.x_pruned: pd.DataFrame = x_df.loc[:, x_df.apply(
            lambda col: col.value_counts(normalize=True).max() < self.infrequent_threshold
        )]
        self.x_pruned: pd.DataFrame = self.x_pruned.loc[:, self.x_pruned.apply(
            lambda col: col.isna().mean() < self.pct_missing_threshold
        )]
        return self

    def transform(self, X) -> pd.DataFrame:
        return self.x_pruned

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.x_pruned.columns


class GenericSupervisedModelExecutor:
    def __init__(self, df: pd.DataFrame, target_column: str, test_size=0.2, random_state=42):
        """
        Initialize a GenericMultiModelEvaluationPipeline object.

        Parameters:
            df (pd.DataFrame): The dataframe containing the data.
            target_column (str): The name of the target column.
            test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
            random_state (int, optional): The random seed for reproducibility. Defaults to 42.

        Attributes:
            test_size (float): The proportion of the dataset to include in the test split.
            random_state (int): The random seed for reproducibility.
            prune_threshold_numerical(float): The column pruning threshold for numerical columns.
                Defaults to 1.0, no pruning.
            prune_threshold_categorical(float): The column pruning threshold for categorical columns.
                Defaults to 1.0, no pruning.
            ordinal_encoding_cols(dict[str, list]): The dictionary of columns that need ordinal encoding.
        """
        self.df = df
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.prune_threshold_numerical = 1.0
        self.prune_threshold_categorical = 1.0
        self.prune_threshold_pct_missing = 1.0
        self.ordinal_encoding_cols: dict[str, list] = {}
        self.selected_feature_list: Optional[list[str]] = None
        self.basic_feature_transformer: Optional[ColumnTransformer] = None
        self.feature_transformer: Optional[ColumnTransformer] = None
        self.target_transformer: Optional[TransformerMixin] = None

    def get_categorical_columns(self, X: pd.DataFrame) -> list[str]:
        return X.select_dtypes(include=['object', 'category']).columns

    def get_numerical_columns(self, X: pd.DataFrame) -> list[str]:
        return X.select_dtypes(include=['int64', 'float64']).columns

    def check_col_features_for_ordinal_encoding(self, col_name: str) -> pd.DataFrame:
        col_features = self.df[col_name].value_counts().index
        feature_data = []
        for feature in col_features:
            feature_data.append(int(self.df.loc[self.df[col_name] == feature, self.target_column].mean()))
        col_feature_df = pd.DataFrame(
            data=feature_data,
            columns=['target_col_mean'],
            index=col_features)
        return col_feature_df.sort_values(by='target_col_mean')

    def get_ordinal_encoding_categories(self, col_name: str) -> list:
        col_feature_df = self.check_col_features_for_ordinal_encoding(col_name)
        return list(col_feature_df.index)

    def add_ordinal_encoding_column(self, col_name: str, categories: list | None):
        if categories is None:
            categories = self.get_ordinal_encoding_categories(col_name)
        self.ordinal_encoding_cols[col_name] = categories

    def scale_and_encode(
            self,
            feature_transformer: ColumnTransformer,
            target_transformer: Optional[TransformerMixin] = None) -> tuple[pd.DataFrame, pd.Series]:
        # Separate features and target variable
        X, y = self.split_x_y()
        X_scaled = feature_transformer.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_transformer.get_feature_names_out())
        if target_transformer is not None:
            y_scaled = target_transformer.fit_transform(self._get_target_df()).ravel()
            y = pd.Series(y_scaled, name=self.target_column, index=y.index)
        return X_scaled, y

    def split_x_y(self) -> tuple[pd.DataFrame, pd.Series]:
        # Separate features and target variable
        return self._split_x_y(self.df, self.target_column)

    def _get_x(self) -> pd.DataFrame:
        return self.df.drop(columns=[self.target_column])

    def _get_target_df(self) -> pd.DataFrame:
        return self.df[[self.target_column]]

    def _split_x_y(self, df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
        # Separate features and target variable
        X: pd.DataFrame = df.copy().drop(columns=[target_column])
        y = df[target_column]
        return X, y

    def train_test_split(self, X, y):
        """
        Split features and target into training and testing datasets.

        Parameters:
            X (numpy array): feature matrix
            y (pandas Series): target variable

        Returns:
            X_train (numpy array): training feature matrix
            X_test (numpy array): testing feature matrix
            y_train (pandas Series): training target variable
            y_test (pandas Series): testing target variable
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)
        return X_train, X_test, y_train, y_test

    def get_basic_feature_transformer(self) -> ColumnTransformer:
        """
        Returns a ColumnTransformer object that preprocesses the input DataFrame `X`.
        The preprocessing includes pruning unnecessary columns, handling missing values,
        scaling numerical features, and ordinal encoding categorical features.

        This will return the basic_feature_transformer if it has already been created or create a new one.

        Returns:
            ColumnTransformer: A ColumnTransformer object used to prune unnecessary columns, handle missing
            data, scale, and encode.
        """
        if self.basic_feature_transformer is None:
            X = self._get_x()
            for col_name in self.get_categorical_columns(X):
                self.add_ordinal_encoding_column(col_name, None)
            self.basic_feature_transformer = self.create_basic_eda_transformer(X)
        return self.basic_feature_transformer

    def create_basic_eda_transformer(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Creates a basic ColumnTransformer object to preprocess the input DataFrame `X`.
        The preprocessing includes pruning unnecessary columns, handling missing values,
        scaling numerical features, and ordinal encoding categorical features.

        Parameters:
            X (pandas DataFrame): containing the features to be transformed

        Returns:
            ColumnTransformer object used to prune unnecessary columns, handle missing data, scale, and encode
        """
        # Initialize column transformer list
        column_transformers = []

        # Create numerical preprocessing pipeline
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        numerical_steps = []
        numerical_steps.append(('pruner', SimplePruner(0.9, 0.9)))
        numerical_steps.append(('imputer', SimpleImputer(strategy='mean')))
        numerical_steps.append(('scaler', StandardScaler()))
        numerical_transformer = Pipeline(steps=numerical_steps, memory='named_steps')
        column_transformers.append(('numerical', numerical_transformer, numerical_cols))

        # Create Ordinal preprocessing pipeline for all categorical columns
        ordinal_cols = list(self.ordinal_encoding_cols.keys())
        for i, col in enumerate(ordinal_cols):
            ord_step_name = f'ordinal_{i}'
            ordinal_steps = []
            ordinal_steps.append((col,
                                  OrdinalEncoder(categories=[self.ordinal_encoding_cols[col]],
                                                 handle_unknown='use_encoded_value',
                                                 encoded_missing_value=-2,
                                                 unknown_value=-1)))
            ordinal_steps.append(('scaler', StandardScaler()))
            ordinal_transformer = Pipeline(steps=ordinal_steps, memory='named_steps')
            column_transformers.append((ord_step_name, ordinal_transformer, [col]))

        # Combine transformers using ColumnTransformer
        # apply different transformations to different columns of a dataset
        return ColumnTransformer(transformers=column_transformers, remainder='drop')

    def create_feature_transformer(
            self,
            num_strategy='mean',
            cat_strategy='constant',
            max_categories=5) -> ColumnTransformer:
        """
        Creates a ColumnTransformer object to prune unnecessary columns, handle missing values,
        scale numerical features, and encode categorical variables.

        In order to prune unnecessary columns, set prune_threshold_numerical and prune_threshold_categorical
        properties to some value less than 1.0. For example, if you set prune_threshold_numerical to 0.9,
        only columns with less than 90% missing values will be pruned.

        Use the set_selected_feature_list method to prune unnecessary columns prior to calling this.
        Use the add_ordinal_encoding_column method to add ordinal encoding to one or more columns prior to calling this.

        Parameters:
            X (pandas DataFrame): containing the features to be transformed
            num_strategy (str): strategy to use for numerical imputation (ie. 'mean')
            cat_strategy (str): strategy to use for categorical imputation (ie. 'constant')
            max_categories (int): maximum number of categories to consider for one-hot encoding

        Returns:
            ColumnTransformer object used to prune unnecessary columns, handle missing data, scale, and encode
        """
        X = self._get_x()

        # Initialize column transformer list
        column_transformers = []

        # Create numerical preprocessing pipeline
        numerical_cols = self.get_numerical_columns(X)
        numerical_steps = []
        numerical_steps.append(('pruner', SimplePruner(
            infrequent_threshold=self.prune_threshold_numerical,
            pct_missing_threshold=self.prune_threshold_pct_missing)
        ))
        numerical_steps.append(('selector', SelectedFeaturesTransformer(
            selected_feature_list=self.selected_feature_list)
        ))
        numerical_steps.append(('imputer', SimpleImputer(strategy=num_strategy)))
        numerical_steps.append(('scaler', StandardScaler()))
        numerical_transformer = Pipeline(steps=numerical_steps, memory='named_steps')
        column_transformers.append(('numerical', numerical_transformer, numerical_cols))

        # Create categorical preprocessing pipeline
        categorical_cols = self.get_categorical_columns(X)
        categorical_cols = self._filter_categorical_cols(categorical_cols)
        if len(categorical_cols) > 0:
            categorical_steps = []
            categorical_steps.append(('pruner', SimplePruner(
                infrequent_threshold=self.prune_threshold_categorical,
                pct_missing_threshold=self.prune_threshold_pct_missing)
            ))
            categorical_steps.append(('selector', SelectedFeaturesTransformer(
                selected_feature_list=self.selected_feature_list)
            ))
            categorical_steps.append(('imputer', SimpleImputer(strategy=cat_strategy)))
            categorical_steps.append(('onehot', OneHotEncoder(
                handle_unknown='infrequent_if_exist',
                max_categories=max_categories,
                sparse_output=False,
                feature_name_combiner=feature_name_combiner))
            )

            categorical_transformer = Pipeline(steps=categorical_steps, memory='named_steps')
            column_transformers.append(('categorical', categorical_transformer, categorical_cols))

        # Create Ordinal preprocessing pipeline
        ord_num = 0
        ordinal_cols = list(self.ordinal_encoding_cols.keys())
        for col in ordinal_cols:
            ord_num += 1
            ordinal_steps = []
            ordinal_steps.append((col,
                                  OrdinalEncoder(categories=[self.ordinal_encoding_cols[col]],
                                                 encoded_missing_value=-1,
                                                 handle_unknown='use_encoded_value',
                                                 unknown_value=-1)))
            ordinal_steps.append(('scaler', StandardScaler()))
            ordinal_transformer = Pipeline(steps=ordinal_steps, memory='named_steps')
            column_transformers.append(('ordinal_'+str(ord_num), ordinal_transformer, [col]))

        # Combine transformers using ColumnTransformer
        # apply different transformations to different columns of a dataset
        return ColumnTransformer(transformers=column_transformers, remainder='drop')

    def _filter_categorical_cols(self, column_list) -> list:
        if self.selected_feature_list is None:
            return [col for col in column_list if col not in self.ordinal_encoding_cols.keys()]
        return [col for col in column_list
                if col not in self.ordinal_encoding_cols.keys()
                and col in self.selected_feature_list]

    def _filter_out_ordinal_cols(self, column_list):
        return list(set(column_list) - set(self.ordinal_encoding_cols.keys()))


class EdaToolbox(GenericSupervisedModelExecutor):
    def __init__(self, df, target_column: str):
        super().__init__(df, target_column)

    def plot_regression_target_column_distribution(self):
        # Visualize the distribution of target_column
        sns.histplot(self.df[self.target_column], kde=True)
        plt.title(f'Distribution of {self.target_column}')
        plt.xlabel(self.target_column)
        plt.ylabel('Frequency')
        plt.show()

    def plot_regression_target_column_distribution_variations(
            self,
            target_transformation_dict: dict[str, TransformerMixin]) -> None:

        # Visualize the distribution of transformed SalePrice variables
        plt.figure(figsize=(15, 10))
        i = 0
        for name, transformer in target_transformation_dict.items():
            i += 1
            plt.subplot(2, 2, i)
            sns.histplot(transformer.fit_transform(self.df[[self.target_column]]), kde=True)
            plt.title(f'Distribution of {self.target_column} with {name}')
            plt.xlabel(name)
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

    def analyze_target_column_transformation_predictions(
            self,
            target_transformation_dict: dict[str, TransformerMixin],
            model_dict: dict[str, BaseEstimator]) -> pd.DataFrame:
        x_basic_transformer: ColumnTransformer = self.get_basic_feature_transformer()
        X = self._get_x()
        X_scaled = x_basic_transformer.fit_transform(X)
        y = self._get_target_df()

        # Evaluate each model
        model_results = {}
        for model_name, model in model_dict.items():
            # Evaluate each transformation
            trans_results = {}
            for name, y_transformer in target_transformation_dict.items():
                y_scaled = y_transformer.fit_transform(y)
                X_train, X_test, y_train, y_test = self.train_test_split(X_scaled, y_scaled)
                model.fit(X_train, y_train.ravel())
                y_pred = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                trans_results[name] = round(rmse, 6)
            model_results[model_name] = trans_results
        return pd.DataFrame(model_results)

    def gather_initial_column_info(self) -> pd.DataFrame:
        # Separate features and target variable
        X, _ = self.split_x_y()
        col_info_df = pd.DataFrame(index=X.columns)
        col_info_df['data_type'] = X.apply(lambda col: col.dtype)
        col_info_df['max_value_count_pct'] = X.apply(lambda col: col.value_counts(normalize=True).max())
        col_info_df['most_frequent_value'] = X.apply(
            lambda col: col.value_counts().sort_values(ascending=False).index[0]
        )
        col_info_df['unique_values'] = X.apply(lambda col: col.nunique())
        col_info_df['missing_values'] = X.isnull().sum()
        col_info_df['missing_values_pct'] = X.isnull().mean()
        col_info_df['top_10_values'] = X.apply(
            lambda col: col.value_counts().sort_values(ascending=False).index[:10]
        )
        return col_info_df.sort_values(by=['data_type', 'max_value_count_pct'], ascending=[False, False])

    def get_columns_to_prune(self, prune_threshold_categorical: float, prune_threshold_numerical: float) -> list:
        col_info_df = self.gather_initial_column_info()
        return list(col_info_df[
            (((col_info_df['data_type'] == 'object') | (col_info_df['data_type'] == 'category'))
             & (col_info_df['max_value_count_pct'] > prune_threshold_categorical))
            | (((col_info_df['data_type'] != 'object') & (col_info_df['data_type'] != 'category'))
                & (col_info_df['max_value_count_pct'] > prune_threshold_numerical))
        ].index)

    def get_regression_feature_analysis_df(
            self,
            feature_transformer: ColumnTransformer,
            target_transformer: TransformerMixin) -> pd.DataFrame:
        X, y = self.scale_and_encode(feature_transformer, target_transformer)
        corr_df = self._get_pd_correlation_abs_features_df(X, y)
        coef_df = self._get_ridge_coefficient_abs_features_df(X, y)
        pvals_df = self._get_lr_pval_features_df(X, y)
        vif_df = self._get_vif_features_df(X)
        pca_df = self._get_pca_features_df(X)
        fa_df = pd.concat([corr_df, coef_df, pvals_df, vif_df, pca_df], axis=1)
        fa_df['total'] = fa_df['correlation_abs'] + fa_df['coefficient_abs']
        return fa_df.sort_values(by='total', ascending=False)

    def plot_regression_feature_correlations(
            self,
            feature_transformer: ColumnTransformer,
            target_transformer: TransformerMixin,
            corr_min=0.5):
        X, y = self.scale_and_encode(feature_transformer, target_transformer)
        corr_df = self._get_pd_correlation_abs_features_df(X, y)
        temp_df = X[list(corr_df[corr_df['correlation_abs'] > corr_min].index)]
        numeric_cols = temp_df.select_dtypes(include=[np.number])

        # Visualize the distribution of transformed SalePrice variables
        plt.figure(figsize=(15, 10))
        i = 0
        for col in numeric_cols.columns:
            i += 1
            plt.subplot(4, 3, i)
            sns.regplot(x=numeric_cols[col], y=y)
            plt.title(f'Regression plot of {col}')
            plt.ylabel(self.target_column)
            plt.xlabel(col)
        plt.tight_layout()
        plt.show()

    def get_regression_important_features_df(self,
                                             fa_df: pd.DataFrame,
                                             total_min: float,
                                             pval_max: float,
                                             vif_max: float | None) -> pd.DataFrame:
        """
        Filter the given feature analysis DataFrame `fa_df` to retrieve the important features for regression analysis.

        Parameters:
            fa_df (pd.DataFrame): The DataFrame containing the feature analysis results.
            total_min (float): The minimum total value to filter by.
            pval_max (float): The maximum p-value to filter by (0.05 is commonly used to
                filter out statistically insignificant features).
            vif_max (float): The maximum VIF value to filter by (5 is commonly used to
                filter out features with high multicollinearity).

        Returns:
            pd.DataFrame: The filtered DataFrame containing the important features for regression analysis.
        """
        if vif_max is None:
            return fa_df[(fa_df['total'] > total_min)
                         & (fa_df['pval'] < pval_max)]

        return fa_df[(fa_df['total'] > total_min)
                     & (fa_df['pval'] < pval_max)
                     & (fa_df['vif'] < vif_max)]

    def get_important_features_list(self, important_features_df: pd.DataFrame) -> list:
        return list(important_features_df.index.str.split('__').str.get(1))

    def get_potential_ordinal_cols(self, important_features_df: pd.DataFrame) -> list:
        return list(set([
                value.split('__')[1] for value in important_features_df.index
                if str(value).startswith('categorical__')
        ]))

    def get_final_ordinal_col_names(self, important_features_list: list) -> list:
        return [value for value in self.ordinal_encoding_cols.keys() if value in important_features_list]

    def plot_important_features(self, important_features_df: pd.DataFrame) -> None:
        cols_to_drop = ['correlation', 'coefficient', 'pval', 'vif', 'PCA2']
        plot_df = important_features_df.drop(columns=cols_to_drop)
        plot_df.plot(
            kind='bar',
            title='Feature selection',
            legend=True,
            xlabel='Feature',
            ylabel='Values', figsize=(15, 8))

    def _get_pd_correlation_abs_features_df(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        corr = X.corrwith(other=y)
        corr_abs = corr.abs()
        return pd.DataFrame({'correlation': corr, 'correlation_abs': corr_abs}, index=X.columns)

    def _get_ridge_coefficient_abs_features_df(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        ridge = RidgeCV(alphas=np.logspace(-6, 6, num=7)).fit(X, y)
        print(f"Best RidgeCV alpha: {ridge.alpha_} (R^2 score: {ridge.score(X, y): .2f})")
        coef = ridge.coef_
        coef_abs: np.ndarray = np.abs(coef)
        coef_max = coef_abs.max()
        if coef_max > 0:
            coef_abs /= coef_max
        return pd.DataFrame({'coefficient': coef, 'coefficient_abs': coef_abs}, index=X.columns)

    def _get_lr_pval_features_df(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        # Looking for p-values < 0.05 which are statistically signicant
        lr: RegressionResultsWrapper = sm.OLS(y, X).fit()
        pvals_df = pd.DataFrame(
            lr.pvalues,
            columns=['pval'],
            index=lr.pvalues.index,
        )
        return pvals_df

    def _get_vif_features_df(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the variance inflation factor (VIF) for each feature in the input DataFrame.
        A VIF value greater than 5 indicates high colinearity.

        Parameters:
            X (pd.DataFrame): The input DataFrame containing the features.

        Returns:
            pd.DataFrame: A DataFrame with two columns: "variables" and "vif". The "variables"
            column contains the feature names, and the "vif" column contains the corresponding
            VIF values. The DataFrame is indexed by the "variables" column.
        """
        vif_df = pd.DataFrame()
        vif_df["variables"] = X.columns
        vif_df["vif"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return vif_df.set_index("variables")

    def _get_pca_features_df(self, X: pd.DataFrame) -> pd.DataFrame:
        # Apply PCA
        n_components = 2
        pca_cols = []
        for n in range(1, n_components + 1):
            pca_cols.append(f"PCA{n}")

        pca = PCA(n_components=n_components, random_state=self.random_state)
        pca.fit_transform(X)
        return pd.DataFrame(pca.components_.T, columns=pca_cols, index=X.columns)


class ClassifierMultiModelEvaluator(GenericSupervisedModelExecutor):

    def __init__(self, df: pd.DataFrame, target_column: str) -> None:
        super().__init__(df, target_column)
        self.evaluations = {'model': [], 'train': [], 'test': [], 'time': []}
        self.best_test_accuracy = 0
        self.best_model = None
        self.best_model_name = None

    def evaluate_models(self, transformer: ColumnTransformer, models: dict[str, ClassifierMixin]) -> pd.DataFrame:
        # Separate features and target variable
        X, y = self.split_x_y()
        print(f'Number original X cols: {X.shape[1]}')

        # Preprocess the data
        X = transformer.fit_transform(X)
        print(f'Number preprocessed X cols: {X.shape[1]}')

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = self.train_test_split(X, y)

        # Train, evaluate, and select the best model
        for model_name, model in models.items():
            start_time = time.time()
            self.evaluations['model'].append(model_name)
            self._train_model(model, X_train, y_train)
            self._evaluate(model_name, model, X_test, y_test)
            self.evaluations['time'].append(time.time() - start_time)

        # Return a dataframe from the evaluations dictionary with model as the index
        return pd.DataFrame(self.evaluations).set_index('model')

    def set_best_model(self, model_name: str) -> None:
        """
        Sets the best model for the classifier pipeline to be used in the predict() method.
        Use this method to override the best_model set in the evaluate_models() method as needed.

        Parameters:
            model_name (str): The name of the model to be set as the best model.

        Returns:
            None
        """
        self.best_model_name = model_name
        self.best_model: ClassifierMixin = self.models[model_name]

    def predict(self, new_data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions for new data using the best trained model.

        Parameters:
            new_data (pd.DataFrame): new data for prediction

        Returns:
            predictions (np.ndarray): predicted values
        """
        # Preprocess the new data
        X_new_data = self.transformer_pipeline.fit_transform(new_data)

        # Make predictions
        predictions = self.best_model.predict(X_new_data)
        return predictions

    def _split_x_y(self, df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
        X, y = super()._split_x_y(df, target_column)
        # If target column is multiclass, encode it to binary
        if y.nunique() > 2:
            y: np.ndarray = LabelEncoder().fit_transform(y)
        return X, y

    def _train_model(self, model: ClassifierMixin, X_train: np.ndarray, y_train: pd.Series) -> None:
        """
        Train a classification model using the training data.

        Parameters:
            model (ClassifierMixin): a scikit-learn classification model instance
            X_train (numpy.ndarray): training feature matrix
            y_train (pandas.Series) training target variable
        """
        # Train the model
        model.fit(X_train, y_train)
        self.evaluations['train'].append(model.score(X_train, y_train))

    def _evaluate(self, model_name: str, model: ClassifierMixin, X_test: np.ndarray, y_test: pd.Series) -> None:
        """
        Evaluate the performance of the model using the training and testing data.
        It appends the training accuracy and testing accuracy to the  respective
        lists in the object.

        Parameters:
            model_name (str): The name of the model.
            model (ClassifierMixin): The trained model object.
            X_train (numpy.ndarray): The training feature matrix.
            y_train (pandas.Series): The training target variable.
            X_test (numpy.ndarray): The testing feature matrix.
            y_test (pandas.Series): The testing target variable.

        Returns:
            None
        """
        test_score = model.score(X_test, y_test)
        self.evaluations['test'].append(test_score)
        if test_score > self.best_test_accuracy:
            self.best_model_name = model_name
            self.best_model = model
            self.best_test_accuracy = test_score


class RegressorMultiModelEvaluator(GenericSupervisedModelExecutor):

    def __init__(self, df: pd.DataFrame, target_column: str) -> None:
        super().__init__(df, target_column)
        self.evaluations = {
            'model_name': [],
            'train_r2': [],
            'mse': [],
            'r2': [],
            'r2_adj': [],
            'lr_cv_mean': [],
            'lr_cv_std': [],
            'time': []
        }
        self.best_test_accuracy = 0
        self.best_model: Optional[RegressorMixin] = None
        self.best_model_name: Optional[str] = None
        self.y_test_pred_list: list[np.ndarray] = []

    def set_column_transformer_properties(self,
                                          selected_feature_list: list[str] | None,
                                          ordinal_feature_list: list[str] | None,
                                          num_strategy='mean',
                                          cat_strategy='constant',
                                          max_categories=5) -> None:
        self.selected_feature_list = selected_feature_list
        self.num_strategy = num_strategy
        self.cat_strategy = cat_strategy
        self.max_categories = max_categories
        if ordinal_feature_list is not None:
            for col_name in ordinal_feature_list:
                self.add_ordinal_encoding_column(col_name, None)

    def set_target_column_transformer(self, transformer: TransformerMixin) -> None:
        self.target_transformer = transformer

    def evaluate_models(self, models: dict[str, RegressorMixin]) -> pd.DataFrame:

        # Create the feature transformer
        self.feature_transformer = self.create_feature_transformer(
            num_strategy=self.num_strategy,
            cat_strategy=self.cat_strategy,
            max_categories=self.max_categories
        )

        # Preprocess the data
        X, _ = self.split_x_y()
        print(f'Number original X cols: {X.shape[1]}')
        X_scaled = self.feature_transformer.fit_transform(X)
        y_scaled = self.target_transformer.fit_transform(self.df[[self.target_column]])
        print(f'Number scaled X cols: {X_scaled.shape[1]}')

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = self.train_test_split(X_scaled, y_scaled)

        self.X_test = X_test
        self.y_test = y_test

        # Train, evaluate, and select the best model
        for model_name, model in models.items():
            start_time = time.time()
            self.evaluations['model_name'].append(model_name)
            self._train_model(model, X_train, y_train)
            self._evaluate(model_name, model, X_test, y_test)
            self.evaluations['time'].append(time.time() - start_time)

        # Return a dataframe from the evaluations dictionary with model as the index
        return pd.DataFrame(self.evaluations).set_index('model_name').sort_values(by='r2_adj', ascending=False)

    def set_best_model(self, model_name: str) -> None:
        """
        Sets the best model for the classifier pipeline to be used in the predict() method.
        Use this method to override the best_model set in the evaluate_models() method as needed.

        Parameters:
            model_name (str): The name of the model to be set as the best model.

        Returns:
            None
        """
        self.best_model_name = model_name
        self.best_model: RegressorMixin = self.models[model_name]

    def predict(self, new_data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions for new data using the best trained model.

        Parameters:
            new_data (pd.DataFrame): new data for prediction

        Returns:
            predictions (np.ndarray): predicted values
        """
        # Preprocess the new data
        X_scaled = self.feature_transformer.fit_transform(new_data)

        # Make predictions
        self.predictions = self.best_model.predict(X_scaled)
        pred_df = pd.DataFrame(self.predictions, columns=[self.target_column])
        return self._target_inverse_transform(pred_df)

    def _train_model(self, model: RegressorMixin, X_train: np.ndarray, y_train: pd.Series) -> None:
        """
        Train a classification model using the training data.

        Parameters:
            model (RegressorMixin): a scikit-learn classification model instance
            X_train (numpy.ndarray): training feature matrix
            y_train (pandas.Series) training target variable
        """
        # Train the model
        model.fit(X_train, y_train.ravel())
        self.evaluations['train_r2'].append(model.score(X_train, y_train))

    def _evaluate(self, model_name: str, model: RegressorMixin, X_test: np.ndarray, y_test: pd.Series) -> None:
        """
        Evaluate the performance of the model using the training and testing data.
        It appends the training accuracy and testing accuracy to the  respective
        lists in the object.

        Parameters:
            model_name (str): The name of the model.
            model (RegressorMixin): The trained model object.
            X_train (numpy.ndarray): The training feature matrix.
            y_train (pandas.Series): The training target variable.
            X_test (numpy.ndarray): The testing feature matrix.
            y_test (pandas.Series): The testing target variable.

        Returns:
            None
        """
        # Score the predictions with mse, r2, and r2_adj
        y_pred = model.predict(X_test)
        self.y_test_pred_list.append(y_pred)
        r2 = model.score(X_test, y_test)
        r2_adj = self._r2_adj(X_test, y_test, r2)
        cv_scores = cross_val_score(LinearRegression(), X_test, y_test, cv=5, scoring="r2")
        self.evaluations['mse'].append(mean_squared_error(y_test, y_pred))
        self.evaluations['r2'].append(r2)
        self.evaluations['r2_adj'].append(r2_adj)
        self.evaluations['lr_cv_mean'].append(cv_scores.mean())
        self.evaluations['lr_cv_std'].append(cv_scores.std())
        if r2_adj > self.best_test_accuracy:
            self.best_model_name = model_name
            self.best_model = model
            self.best_test_accuracy = r2_adj
            self.best_y_pred_raw = y_pred

    def _r2_adj(self, x, y, r2) -> float:
        n_cols = x.shape[1]
        return 1 - (1 - r2) * (len(y) - 1) / (len(y) - n_cols - 1)

    def _target_inverse_transform(self, y: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform the target variable.

        Parameters:
            y (numpy.ndarray): target variable

        Returns:
            y (numpy.ndarray): inverse transformed target variable
        """
        if self.target_transformer is not None:
            y = self.target_transformer.inverse_transform(y).flatten()
        return y
