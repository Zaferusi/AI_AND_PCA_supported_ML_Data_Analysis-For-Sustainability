import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('arranged-last.csv')


# Drop the 'id' column and select only numerical features, excluding the target variable
numerical_features = df.select_dtypes(include=[np.number]).drop(columns=['CO2_EMISSIONS_CURRENT','ENVIRONMENT_IMPACT_CURRENT','ENERGY_CONSUMPTION_CURRENT','CURRENT_ENERGY_EFFICIENCY','GLAZED_TYPE','SOLAR_WATER_HEATING_FLAG','ENERGY_CONSUMPTION_CURRENT','CO2_EMISS_CURR_PER_FLOOR_AREA','UPRN','Unnamed: 0'])
#CURRENT ENERGY EFFICIENCY EKLENDİ DEĞİŞİKLİK OLARAK.

# Separate the target variable
y = df['CO2_EMISS_CURR_PER_FLOOR_AREA']

# Standardize the numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numerical_features)

# Apply PCA
pca = PCA(n_components=0.95)  # Adjust the number of components or explained variance ratio
X_pca = pca.fit_transform(X_scaled)

# Extract the PCA coefficients (loadings)
loadings = pca.components_.T  # Transpose to align with the original features

# Create a DataFrame for the PCA loadings
loadings_df = pd.DataFrame(loadings, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])], index=numerical_features.columns)

# Create a DataFrame for the PCA features
pca_df = pd.DataFrame(data=X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])

# Add the target variable and id back to the DataFrame
pca_df['CO2_EMISS_CURR_PER_FLOOR_AREA'] = y


# Save the new DataFrame and loadings to CSV files
pca_df.to_csv('PCA_transformed_features.csv', index=False)
loadings_df.to_csv('PCA_loadings.csv', index=True)

# Save scaler mean and variance for later use
scaler_params = pd.DataFrame({'mean': scaler.mean_, 'variance': scaler.var_}, index=numerical_features.columns)
scaler_params.to_csv('scaler_parameters.csv', index=True)


print("PCA transformation and loadings are computed and saved.")
