import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class HomePriceModel:
    def __init__(self, city_file, metro_file, state_file):
        # Load datasets
        self.city_df = pd.read_csv(city_file)
        self.metro_df = pd.read_csv(metro_file)
        self.state_df = pd.read_csv(state_file)

    def clean_data(self):
        # Implement data cleaning for all datasets
        self.city_df = self._clean_individual_dataset(self.city_df, 'City')
        self.metro_df = self._clean_individual_dataset(self.metro_df, 'Metro')
        self.state_df = self._clean_individual_dataset(self.state_df, 'State')

    def _clean_individual_dataset(self, df, level_name):
        # General cleaning operations (e.g., filling missing values, dropping unnecessary columns)
        df.fillna(method='ffill', inplace=True)
        df.dropna(inplace=True)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Removing unnamed columns if present
        return df

    def merge_data(self):
        # Merge data from the three datasets if necessary
        merged_df = pd.merge(self.city_df, self.metro_df, on=['common_column'], how='inner')
        merged_df = pd.merge(merged_df, self.state_df, on=['common_column'], how='inner')
        self.final_df = merged_df

    def prepare_features(self):
        # Select relevant features and target variable
        self.X = self.final_df.drop(columns=['Price'])
        self.y = self.final_df['Price']

    def train_model(self):
        # Train a Linear Regression model
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Evaluation
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Mean Squared Error: {mse}")
        print(f"R-Squared Score: {r2}")

    def export_cleaned_data(self):
        # Export cleaned data to CSV for further use
        self.final_df.to_csv('cleaned_home_prices.csv', index=False)


