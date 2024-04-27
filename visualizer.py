from ucimlrepo import fetch_ucirepo
import pandas as pd

# Fetch del dataset
real_estate_valuation = fetch_ucirepo(id=477)

df_features = pd.DataFrame(real_estate_valuation.data.features, columns=real_estate_valuation.feature_names)
df_targets = pd.DataFrame(real_estate_valuation.data.targets, columns=real_estate_valuation.target_names)

import pandas as pd

# Calcolare la correlazione tra features e target
correlation_with_target = df_features.corrwith(df_targets.squeeze())

# Stampare la correlazione tra features e target
print(correlation_with_target)


units_features = {
    'X1 transaction date': 'anno.mese (es. 2013.250 = 2013 marzo)',
    'X2 house age': 'anni',
    'X3 distance to the nearest MRT station': 'metri',
    'X4 number of convenience stores': 'unità',
    'X5 latitude': 'gradi',
    'X6 longitude': 'gradi'
}

units_target = {
    'house price of unit area': '10,000 New Taiwan Dollar/Ping'
}

# Features
print("DataFrame delle Features:")
print(df_features.head())
print("\nUnità di misura delle Features:")
for feature, unit in units_features.items():
    print(f"{feature}: {unit}")

# Targets
print("\nDataFrame delle Targets:")
print(df_targets.head())
print("\nUnità di misura della Target:")
for target, unit in units_target.items():
    print(f"{target}: {unit}")
