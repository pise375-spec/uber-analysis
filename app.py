import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set page configurations
st.set_page_config(page_title="Uber Fares Analytics Dashboard", layout="wide")

st.title("🚗 Uber Rides & Fares Analytics Dashboard")
st.markdown("Convert your raw geospatial and fare dataset into actionable trip insights.")

# ---------------------- DATA LOADING ----------------------
st.sidebar.header("Data Settings")
uploaded_file = st.sidebar.file_uploader("Upload your uber.csv file", type=["csv"])

# Local default path fallback if no file is uploaded via the interface
default_path = "uber.csv"

# Vectorized Haversine formula to compute distance in miles from GPS coordinates
def calculate_distance_miles(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    miles = 6367 * c * 0.621371
    return miles

@st.cache_data
def load_and_clean_data(file_source):
    # Read data
    dataset = pd.read_csv(file_source)
    
    # Strip spaces from column headers
    dataset.columns = dataset.columns.str.strip()
    
    # Verify critical columns exist
    required_cols = ['fare_amount', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']
    for col in required_cols:
        if col not in dataset.columns:
            st.error(f"The dataset is missing the required '{col}' column.")
            st.stop()
            
    # Parse dates
    dataset['pickup_datetime'] = pd.to_datetime(dataset['pickup_datetime'], errors='coerce')
    dataset.dropna(subset=['pickup_datetime'], inplace=True)
    
    # Feature Engineering
    dataset['date'] = dataset['pickup_datetime'].dt.date
    dataset['time'] = dataset['pickup_datetime'].dt.hour
    dataset['MONTH'] = dataset['pickup_datetime'].dt.month
    dataset['DAY'] = dataset['pickup_datetime'].dt.weekday
    
    # Calculate Trip Distance
    dataset['distance_miles'] = calculate_distance_miles(
        dataset['pickup_longitude'], dataset['pickup_latitude'],
        dataset['dropoff_longitude'], dataset['dropoff_latitude']
    )
    
    # Classify periods of the day
    dataset['day-night'] = pd.cut(
        dataset['time'], 
        bins=[0, 6, 12, 17, 21, 24], 
        labels=['Late Night', 'Morning', 'Afternoon', 'Evening', 'Night'],
        include_lowest=True
    )
    
    # Filter out obvious coordinate noise/errors (0 values) to keep visualizations clean
    dataset = dataset[(dataset['distance_miles'] >= 0) & (dataset['distance_miles'] < 100)]
    dataset = dataset[(dataset['fare_amount'] > 0) & (dataset['fare_amount'] < 200)]
    
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
    st.warning("Please upload your Uber dataset CSV file in the sidebar to view metrics.")
    st.stop()

# ---------------------- OVERVIEW METRICS ----------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Trips Evaluated", f"{dataset.shape[0]:,}")
with col2:
    st.metric("Total Revenue Generated", f"${dataset['fare_amount'].sum():,.2f}")
with col3:
    st.metric("Avg Fare Amount", f"${dataset['fare_amount'].mean():.2f}")
with col4:
    st.metric("Avg Calculated Distance", f"{dataset['distance_miles'].mean():.2f} mi")

# Show raw dataframe sample if requested
if st.checkbox("Show dataset sample overview"):
    st.dataframe(dataset.head(50))

# ---------------------- DATA ANALYSIS & VISUALIZATION ----------------------
st.header("📊 Fare & Distribution Analysis")

tab1, tab2, tab3 = st.tabs(["Passenger & Fare Distributions", "Time & Mileage Trends", "Advanced Correlation Matrix"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=dataset, x='passenger_count', ax=ax, palette="Blues_d")
        plt.title("Distribution of Passenger Counts per Trip")
        plt.xlabel("Passenger Count")
        plt.ylabel("Trip Count")
        st.pyplot(fig)
        plt.close()
        
    with c2:
        fig, ax = plt.subplots(figsize=(8, 4))
        # Filtering normal fare range for clearer distribution display
        sns.histplot(dataset[dataset['fare_amount'] < 50]['fare_amount'], kde=True, bins=30, ax=ax, color="green")
        plt.title("Distribution of Fare Amounts (Under $50)")
        plt.xlabel("Fare ($)")
        st.pyplot(fig)
        plt.close()

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=dataset[dataset['fare_amount'] < 60], x='passenger_count', y='fare_amount', ax=ax)
    plt.title("Fare Amount vs Passenger Count")
    plt.xlabel("Passenger Count")
    plt.ylabel("Fare Amount ($)")
    st.pyplot(fig)
    plt.close()

with tab2:
    c3, c4 = st.columns(2)
    with c3:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=dataset, x='day-night', ax=ax, order=['Morning', 'Afternoon', 'Evening', 'Night', 'Late Night'])
        plt.title("Rides by Time of Day")
        plt.xlabel("Time Category")
        st.pyplot(fig)
        plt.close()
        
    with c4:
        fig, ax = plt.subplots(figsize=(8, 4))
        # Filtering trips under 20 miles for layout focus
        sns.histplot(dataset[dataset['distance_miles'] < 20]['distance_miles'], kde=True, bins=30, ax=ax, color="orange")
        plt.title("Distribution of Calculated Miles (Under 20 mi)")
        plt.xlabel("Distance (Miles)")
        st.pyplot(fig)
        plt.close()

with tab3:
    st.subheader("Geospatial & Metric Structural Correlation")
    
    # Select numerical attributes for correlation metrics
    numeric_dataset = dataset[['fare_amount', 'passenger_count', 'distance_miles', 'time', 'MONTH', 'DAY']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_dataset.corr(), cmap='BrBG', annot=True, linewidths=1.5, fmt='.2f', ax=ax)
    plt.title("Metrics Correlation Heatmap Matrix")
    st.pyplot(fig)
    plt.close()

# ---------------------- TIME SERIES & TEMPORAL TRENDS ----------------------
st.header("📅 Temporal Breakdown Trends")

month_label = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
dataset['MONTH_NAME'] = dataset['MONTH'].map(month_label)

monthly_counts = dataset['MONTH_NAME'].value_counts(sort=False)
monthly_avg_fare = dataset.groupby('MONTH_NAME', sort=False)['fare_amount'].mean()

ordered_months = [m for m in month_label.values() if m in monthly_counts.index]
df_month = pd.DataFrame({
    "TOTAL RIDES": [monthly_counts[m] for m in ordered_months],
    "AVG FARE": [monthly_avg_fare[m] for m in ordered_months]
}, index=ordered_months).reset_index().rename(columns={'index': 'MONTHS'})

day_label = {0: 'Mon', 1: 'Tues', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
dataset['DAY_NAME'] = dataset['DAY'].map(day_label)
ordered_days = ['Mon', 'Tues', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
day_counts = dataset['DAY_NAME'].value_counts().reindex(ordered_days)

c5, c6 = st.columns(2)
with c5:
    fig, ax1 = plt.subplots(figsize=(8, 4.2))
    
    color = 'tab:blue'
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Total Rides', color=color)
    sns.lineplot(data=df_month, x='MONTHS', y='TOTAL RIDES', marker='o', color=color, ax=ax1, label='Total Rides')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Avg Fare ($)', color=color)
    sns.lineplot(data=df_month, x='MONTHS', y='AVG FARE', marker='s', color=color, ax=ax2, label='Avg Fare')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("Monthly Performance & Fare Trends")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

with c6:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    sns.barplot(x=day_counts.index, y=day_counts.values, ax=ax, palette="plasma")
    plt.title("Rides volume by Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Trip Count")
    st.pyplot(fig)
    plt.close()
