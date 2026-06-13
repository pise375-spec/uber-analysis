
import streamlit as str
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import os

# Set page configurations
st.set_page_config(page_title="Uber Data Dashboard", layout="wide")

st.title("🚗 Uber Rides Analytics Dashboard")
st.markdown("Convert your raw dataset into actionable trip insights.")

# ---------------------- DATA LOADING ----------------------
st.sidebar.header("Data Settings")
uploaded_file = st.sidebar.file_uploader("Upload your UberDataset.csv file", type=["csv"])

# Local default path fallback if no file is uploaded via the interface
default_path = r"C:\Users\LENOVO\Downloads\main\UberDataset.csv"

@st.cache_data
def load_and_clean_data(file_source):
    # Read data
    dataset = pd.read_csv(file_source)
    
    # Data Cleaning
    dataset['PURPOSE'].fillna("NOT", inplace=True)
    dataset['START_DATE'] = pd.to_datetime(dataset['START_DATE'], errors='coerce')
    dataset['END_DATE'] = pd.to_datetime(dataset['END_DATE'], errors='coerce')
    
    # Feature Engineering
    dataset['date'] = dataset['START_DATE'].dt.date
    dataset['time'] = dataset['START_DATE'].dt.hour
    
    dataset['day-night'] = pd.cut(
        dataset['time'], 
        bins=[0, 10, 15, 19, 24], 
        labels=['Morning', 'Afternoon', 'Evening', 'Night'],
        include_lowest=True
    )
    
    dataset.dropna(inplace=True)
    dataset.drop_duplicates(inplace=True)
    return dataset

# Establish data source logic
if uploaded_file is not None:
    dataset = load_and_clean_data(uploaded_file)
    st.sidebar.success("Loaded uploaded file!")
elif os.path.exists(default_path):
    dataset = load_and_clean_data(default_path)
    st.sidebar.info("Displaying data from local fallback path.")
else:
    st.warning("Please upload an Uber dataset CSV file in the sidebar to view metrics.")
    st.stop()

# ---------------------- OVERVIEW METRICS ----------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Trips Evaluated", f"{dataset.shape[0]:,}")
with col2:
    st.metric("Total Mileage Tracked", f"{int(dataset['MILES'].sum()):,} mi")
with col3:
    st.metric("Max Single Trip", f"{dataset['MILES'].max()} mi")
with col4:
    st.metric("Avg Trip Length", f"{dataset['MILES'].mean():.2f} mi")

# Show raw dataframe sample if requested
if st.checkbox("Show dataset sample overview"):
    st.dataframe(dataset.head(50))

# ---------------------- DATA ANALYSIS & VISUALIZATION ----------------------
st.header("📊 Categorical & Distribution Analysis")

tab1, tab2, tab3 = st.tabs(["Category & Purpose Distributions", "Time & Mileage Trends", "Advanced Correlation Matrix"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=dataset, x='CATEGORY', ax=ax)
        plt.xticks(rotation=45)
        plt.title("Ride Category Distribution")
        st.pyplot(fig)
        plt.close()
        
    with c2:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=dataset, x='PURPOSE', ax=ax)
        plt.xticks(rotation=45)
        plt.title("Ride Purpose Distribution")
        st.pyplot(fig)
        plt.close()

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.countplot(data=dataset, x='PURPOSE', hue='CATEGORY', ax=ax)
    plt.xticks(rotation=45)
    plt.title("Purpose vs Ride Category")
    st.pyplot(fig)
    plt.close()

with tab2:
    c3, c4 = st.columns(2)
    with c3:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=dataset, x='day-night', ax=ax)
        plt.title("Rides by Time of Day")
        st.pyplot(fig)
        plt.close()
        
    with c4:
        fig, ax = plt.subplots(figsize=(8, 4))
        # Filtering trips under 40 miles
        sns.histplot(dataset[dataset['MILES'] < 40]['MILES'], kde=True, bins=30, ax=ax)
        plt.title("Distribution of Miles (Under 40)")
        plt.xlabel("Miles")
        st.pyplot(fig)
        plt.close()

with tab3:
    st.subheader("One-Hot Encoded Structural Correlation")
    
    # Process categories for core matrix visualization safely on runtime cached data
    categorical_cols = ['CATEGORY', 'PURPOSE']
    OH_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    OH_cols = pd.DataFrame(OH_encoder.fit_transform(dataset[categorical_cols]))
    OH_cols.index = dataset.index
    OH_cols.columns = OH_encoder.get_feature_names_out()

    # Create local temporary merged frame for numeric heat mapping matrix
    heatmap_df = pd.concat([dataset.drop(categorical_cols, axis=1), OH_cols], axis=1)
    numeric_dataset = heatmap_df.select_dtypes(include=['number'])
    
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(numeric_dataset.corr(), cmap='BrBG', annot=True, linewidths=1.5, fmt='.2f', ax=ax)
    plt.title("Correlation Heatmap Matrix")
    st.pyplot(fig)
    plt.close()

# ---------------------- TIME SERIES & TEMPORAL TRENDS ----------------------
st.header("📅 Temporal Breakdown Trends")

# 1. Monthly Performance Charting
dataset['MONTH'] = dataset['START_DATE'].dt.month
month_label = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
dataset['MONTH_NAME'] = dataset['MONTH'].map(month_label)
... 
... monthly_counts = dataset['MONTH_NAME'].value_counts(sort=False)
... monthly_max_miles = dataset.groupby('MONTH_NAME', sort=False)['MILES'].max()
... 
... # Reorder indices sequentially to correct any display jumps
... ordered_months = [m for m in month_label.values() if m in monthly_counts.index]
... df_month = pd.DataFrame({
...     "TOTAL RIDES": [monthly_counts[m] for m in ordered_months],
...     "MAX MILES": [monthly_max_miles[m] for m in ordered_months]
... }, index=ordered_months).reset_index().rename(columns={'index': 'MONTHS'})
... 
... # 2. Weekday Performance Charting
... dataset['DAY'] = dataset['START_DATE'].dt.weekday
... day_label = {0: 'Mon', 1: 'Tues', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
... dataset['DAY_NAME'] = dataset['DAY'].map(day_label)
... ordered_days = ['Mon', 'Tues', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
... day_counts = dataset['DAY_NAME'].value_counts().reindex(ordered_days)
... 
... c5, c6 = st.columns(2)
... with c5:
...     fig, ax = plt.subplots(figsize=(8, 4.2))
...     sns.lineplot(data=df_month, x='MONTHS', y='TOTAL RIDES', label='Total Rides', marker='o', ax=ax)
...     sns.lineplot(data=df_month, x='MONTHS', y='MAX MILES', label='Max Miles', marker='o', ax=ax)
...     plt.title("Monthly Ride and Distance Trends")
...     plt.xlabel("Month")
...     plt.ylabel("Count / Distance")
...     st.pyplot(fig)
...     plt.close()
... 
... with c6:
...     fig, ax = plt.subplots(figsize=(8, 4.2))
...     sns.barplot(x=day_counts.index, y=day_counts.values, ax=ax)
...     plt.title("Rides by Day of Week")
...     plt.xlabel("Day")
...     plt.ylabel("Count")
...     st.pyplot(fig)
