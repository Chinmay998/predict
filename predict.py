# import streamlit as st
# from snowflake.snowpark.session import Session
# from snowflake.snowpark.functions import month, year, col, sum, max, dateadd, current_date
# from snowflake.snowpark.version import VERSION
# import json
# import logging
# from sklearn.linear_model import LinearRegression
# from sklearn.impute import SimpleImputer
# import pandas as pd
# import calendar
# from snowflake.snowpark.types import FloatType

# logger = logging.getLogger("snowflake.snowpark.session")
# logger.setLevel(logging.ERROR)
# connection_parameters = json.load(open('connect.json'))
# session = Session.builder.configs(connection_parameters).create()
# session.sql_simplifier_enabled = True

# # Snowflake environment details
# snowflake_environment = session.sql('select current_user(), current_version()').collect()
# snowpark_version = VERSION

# # Fetch credit consumption data from Snowflake
# snow_df_spend = session.table('METERING_HISTORY')
# snow_df_spend_per_month = snow_df_spend.group_by(year('END_TIME'), month('END_TIME')).agg(sum('CREDITS_USED').as_('CREDITS')). \
#     with_column_renamed('"YEAR(END_TIME)"', "YEAR").with_column_renamed('"MONTH(END_TIME)"', "MONTH").sort('YEAR', 'MONTH')

# # Convert Snowflake DataFrame to Pandas DataFrame
# df = snow_df_spend_per_month.to_pandas()
# df1 = snow_df_spend.to_pandas()

# # Preprocess the data
# imputer = SimpleImputer(strategy='mean')
# input_features_processed = imputer.fit_transform(df)

# # Separate the preprocessed features and target variable
# processed_features = input_features_processed[:, :-1]
# target_variable = input_features_processed[:, -1]

# # Function to predict next month's credit consumption
# def predict_next_month_consumption(num_of_months):
#     # Adjust the selected number of months based on the available data
#     available_months = processed_features.shape[0]
#     selected_num_of_months = min(num_of_months, available_months-1)

#     # Split the data into training and testing sets based on the selected number of months
#     training_features = processed_features[:-selected_num_of_months, :]
#     training_target = target_variable[:-selected_num_of_months]
#     testing_features = processed_features[-selected_num_of_months-1:-1, :]

#     # Train the model
#     model = LinearRegression()
#     model.fit(training_features, training_target)

#     # Make predictions for the next 6 months
#     next_months_features = []
#     for i in range(1, num_of_months + 1):
#         next_month_features = testing_features[-1, :].reshape(1, -1)
#         next_month_prediction = model.predict(next_month_features)
#         next_months_features.append(next_month_prediction[0])
#         testing_features = dateadd(testing_features, '1 MONTH')

#     return next_months_features

# # Streamlit application
# st.title("Credit Consumption Prediction")

# # Adjust the number of months
# num_of_months = st.slider("Select the number of months", min_value=1, max_value=12, value=6, step=1)

# # Predict next 6 months' credit consumptions
# next_months_predictions = predict_next_month_consumption(num_of_months)

# # Display the predictions
# st.write("Predictions for the next 6 months' credit consumption:")
# for i, prediction in enumerate(next_months_predictions, start=1):


import streamlit as st
import pandas as pd
from snowflake.snowpark.session import Session
import json
import logging
# from snowflake.connector import connect, snowflake
from snowflake.connector.pandas_tools import pd_read_sql

logger = logging.getLogger("snowflake.snowpark.session")
logger.setLevel(logging.ERROR)
connection_parameters = json.load(open('connect.json'))
session = Session.builder.configs(connection_parameters).create()
session.sql_simplifier_enabled = Tru
#Create a Streamlit app
st.title('Credit Consumption Prediction')

# Function to fetch credit consumption data
def fetch_credit_consumption():
    snow_df_spend = session.table('METERING_HISTORY')
    snow_df_spend_per_month = snow_df_spend.group_by(year('END_TIME'), month('END_TIME')).agg(sum('CREDITS_USED').as_('CREDITS')). \
        with_column_renamed('"YEAR(END_TIME)"', "YEAR").with_column_renamed('"MONTH(END_TIME)"', "MONTH").sort('YEAR', 'MONTH')
    return snow_df_spend_per_month.to_pandas()

# Fetch credit consumption data from Snowflake
credit_data = fetch_credit_consumption()

# Display the last 6 months' credit consumption
st.subheader('Last 6 Months Credit Consumption')
st.dataframe(credit_data.tail(6))

# Predict the next 6 months' credit consumption
st.subheader('Predicted Next 6 Months Credit Consumption')
# You can add your prediction logic here using machine learning or statistical methods.

# For demonstration, let's assume a simple prediction (average growth rate)
average_growth_rate = (credit_data['CREDITS'].iloc[-1] - credit_data['CREDITS'].iloc[-7]) / 6
next_6_months = [credit_data['CREDITS'].iloc[-1] + i * average_growth_rate for i in range(1, 7)]
next_6_months_df = pd.DataFrame({
    'YEAR': [credit_data['YEAR'].iloc[-1] + (i // 12) for i in range(1, 7)],
    'MONTH': [credit_data['MONTH'].iloc[-1] + (i % 12) for i in range(1, 7)],
    'CREDITS': next_6_months
})
st.dataframe(next_6_months_df)

# Close the Snowflake connection
conn.close()

#     st.write(f"Month {i}: {prediction}")



