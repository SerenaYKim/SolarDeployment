# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
import shap

# Load data
solar_bg_master = pd.read_csv("/data/solardep-bg-co.csv")

# Subset data with selected features
solar_bg_count_withpolicy = solar_bg_master[['pvcountperhouse2020', "prop_treearea", "dniavg",
                        "mh_income_2019", "medianage", "prop_age65over", "prop_belowpov",
                        'prop_bachelorhigher', 'prop_renteroccupied', 'YearStructureBuilt', "avg_bdrooms",
                        'MedianHomeValue', 'ruca_primary','dem_votes', "black_prop", 'his_prop', 'asian_prop',
                        'others_prop', 'ln_TransLineVolt', 'ln_TransLineLength',
                        'utiltypemuni', 'utiltypecoop', 'res_rate', 'comm_rate',
                        "solarmandate", "nem",
                        'solsmart_awardee', 'online_permitting',
                        'sameday_inperson_permitting', "permitandpreinstalldays",
                        "DRGT_EALS", 'WFIR_EALS', 'HAIL_EALS', "WNTW_EALS", "SWND_EALS", "TRND_EALS",
                        'EP_DISABL', 'EP_SNGPNT', 'EP_LIMENG', 'EP_MUNIT', 'EP_MOBILE', 'EP_CROWD','EP_NOVEH', 'EP_UNEMP'
                        ]]

# Rename columns for better readability
solar_bg_count_withpolicy = solar_bg_count_withpolicy.rename(columns={"pvareaperroof2020": "PV-to-Roof Ratio", "pvcountperhouse2020": "PV Count Per HH", "prop_treearea": "% Tree-to-Land Area", "dniavg": "Solar Radiation",
                                                                      "mh_income_2019": "Median HH Income", "medianage": "Median Age", "prop_age65over": "% 65+", "prop_belowpov": "% Below Poverty",
                                                                      'prop_bachelorhigher': "% Bachelors+", 'prop_renteroccupied': "% Renters", 'YearStructureBuilt': "Year Structure Built", "avg_bdrooms": "Avg. No. Bedrooms",
                                                                      'MedianHomeValue': "Median Home Value",'ruca_primary': "Rurality", 'dem_votes': "% Dem. Votes", "black_prop": "% African American",
                                                                      'his_prop': "% Hispanic", 'asian_prop': "% Asian", 'others_prop': "% Other Race", 'ln_TransLineVolt': "Transmission Volt.", 'ln_TransLineLength': "Transmission Length",
                                                                      'utiltypemuni': "Muni. Utilities", 'utiltypecoop': "Rural Co-Ops", 'res_rate': "Resident. Elec. Rate", 'comm_rate': "Commercial Elec. Rate", 'ind_rate': "Industial Elec. Rate",
                                                                      "solarmandate": "Solar Mandate", "nem": "Net Metering", 'solsmart_awardee': "SolSmart Awardee", 'online_permitting': "Online Permitting",
                                                                      'sameday_inperson_permitting': "Sameday Inperson Permit",
                                                                      "permitandpreinstalldays": "Permit & Pre-Install Days",
                                                                      "DRGT_EALS": "Drought Risk", 'WFIR_EALS': "Wildfire Risk", 'HAIL_EALS': "Hail Risk", "WNTW_EALS": "Winter Weather Risk", "SWND_EALS": "Strong Wind Risk", "TRND_EALS": "Tornado Risk",
                                                                      'EP_DISABL': "% Disability", 'EP_SNGPNT': "% Single Parent", 'EP_LIMENG': "% Limited English", 'EP_MUNIT': "% 10+ Unit Housing", 'EP_MOBILE': "% Mobile Home",
                                                                      'EP_CROWD': "% Ppl. > Rooms",'EP_NOVEH': "% No Vehicle", 'EP_UNEMP': "% Unemployed"
                                                                      }
                                                            )

# Drop rows with infinite values and NaNs
solar_bg_count_withpolicy = solar_bg_count_withpolicy.replace(np.inf, np.nan)
solar_bg_count_withpolicy = solar_bg_count_withpolicy.dropna()

# Define the input and predicted values
y_df = solar_bg_count_withpolicy["PV Count Per HH"]
X_df = solar_bg_count_withpolicy.drop(["PV Count Per HH"], axis=1)
X = X_df.values
y = y_df.values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, test_size=0.1, random_state=1982)

# Create the Random Forest Regressor model
regr = RandomForestRegressor(n_estimators=700,
                             max_depth=150,
                             min_samples_split=2,
                             max_features="sqrt",
                             min_samples_leaf=2,
                             random_state=372)

# Fit the model
regr.fit(X_train, y_train)

# Make predictions on the test data
y_pred = regr.predict(X_test)

# Feature Importance for Random Forest
fis_regr = pd.DataFrame({'Feature': X_df.columns, 'Random Forest FIS': regr.feature_importances_})
fis_regr = fis_regr.sort_values('Random Forest FIS', ascending=False).reset_index()

# Standardization
fis_regr["Random Forest FIS STD"] = (fis_regr["Random Forest FIS"] - fis_regr["Random Forest FIS"].min()) / (
        fis_regr["Random Forest FIS"].max() - fis_regr["Random Forest FIS"].min())

fis_regr_tomerge = fis_regr[["Feature", "Random Forest FIS STD"]]

# Calculate metrics for Random Forest Regressor
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
ev = explained_variance_score(y_test, y_pred)

# Print metrics
print('Random Forest Decision Tree Mean Absolute Error:', round(mae, 4))
print('Random Forest Mean Squared Error:', round(mse, 4))
print('Random Forest R-squared Scores:', round(r2, 4))
print('Random Forest Explained Variance Scores:', round(ev, 4))

# SHAP for Random Forest Regressor
explainer = shap.TreeExplainer(regr)
shap_values = explainer.shap_values(X_df)
shap.summary_plot(shap_values, X_df)