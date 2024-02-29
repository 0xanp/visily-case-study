import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, pearsonr


st.set_page_config(
    page_title="Visily Case Study",
    page_icon="🧊",
    layout="wide",
    )

# Load and preprocess the dataset
@st.cache_data
def load_data(filepath):
    df = pd.read_csv(filepath)
    df['is_current_paying_user'] = df['is_current_paying_user'].map({'YES': 1, 'NO': 0})
    size_mapping = {'1-10': 1, '11-50': 2, '51-200': 3, '201-500': 4, '501-1000': 5, '1001-5000': 6, '5001+': 7}
    df['company_size_mapped'] = df['company_employee_count'].map(size_mapping)
    return df

# Perform a T-test for two groups
def perform_ttest(group1, group2):
    stat, p_value = ttest_ind(group1, group2, nan_policy='omit')
    return stat, p_value

# Calculate Pearson correlation for all numerical columns and format as a DataFrame
def calculate_correlations(df):
    corr_df = df.select_dtypes(include=['int64', 'float64']).corr(method='pearson')
    return corr_df

data = load_data('pda_sample_data_set.csv')

# Streamlit App Interface
st.title('Visily Case Study')

# EDA and Visualization Section
st.header('Exploratory Data Analysis and Visualizations')

with st.expander('Data Explorer'):
    st.header('Dataset')
    st.write(data)
    st.write("Descriptive Stats")
    st.write(data.describe(include='all'))
    # Select attributes for visualization
    attributes = ['user_days_in_visily_count', 'total_team_members_invited_to_workspace_count', 'company_size_mapped']
    selected_attributes = st.multiselect('Select attributes to visualize:', options=attributes, default=attributes[:2])

    for attribute in selected_attributes:
        st.subheader(f'Distribution of {attribute}:')
        fig, ax = plt.subplots()
        sns.histplot(data=data, x=attribute, hue='is_current_paying_user', kde=True, ax=ax, palette='viridis', multiple="stack")
        st.pyplot(fig)


# Correlation Analysis
st.header('Pearson Correlation Analysis')
corr_df = calculate_correlations(data)
st.write("Correlation Table:")
st.dataframe(corr_df.style.background_gradient(cmap='coolwarm').format("{:.2f}"))


# Hypothesis Testing Section
st.header('Hypothesis Testing')

# Hypothesis 1
st.subheader('Hypothesis 1: Paying users are active for more days than non-paying users.')
days_active_paying = data[data['is_current_paying_user'] == 1]['user_days_in_visily_count']
days_active_nonpaying = data[data['is_current_paying_user'] == 0]['user_days_in_visily_count']
stat, p = perform_ttest(days_active_paying, days_active_nonpaying)
st.success(f"Test Statistic: {stat:.3f}, P-value: {p:.3g}", icon="✅")
st.success("Conclusion: " + ("There is a significant difference in activity days between paying and non-paying users." if p < 0.05 else "There is no significant difference in activity days between paying and non-paying users."), icon="✅")

# Hypothesis 2
st.subheader('Hypothesis 2: Users from larger companies are more likely to be paying users.')
company_size_paying = data[data['is_current_paying_user'] == 1]['company_size_mapped']
company_size_nonpaying = data[data['is_current_paying_user'] == 0]['company_size_mapped']
stat, p = perform_ttest(company_size_paying, company_size_nonpaying)
st.success(f"Test Statistic: {stat:.3f}, P-value: {p:.3g}", icon="✅")
st.success("Conclusion: " + ("Larger companies are more likely to have paying users." if p < 0.05 else "Company size does not significantly affect the likelihood of being a paying user."), icon="✅")

# Hypothesis 3
st.subheader('Hypothesis 3: Users who invite more team members are more likely to be paying users.')
team_invites_paying = data[data['is_current_paying_user'] == 1]['total_team_members_invited_to_workspace_count']
team_invites_nonpaying = data[data['is_current_paying_user'] == 0]['total_team_members_invited_to_workspace_count']
stat, p = perform_ttest(team_invites_paying, team_invites_nonpaying)
st.success(f"Test Statistic: {stat:.3f}, P-value: {p:.3g}", icon="✅")
st.success("Conclusion: " + ("Users who invite more team members are more likely to be paying users." if p < 0.05 else "The number of team members invited does not significantly affect the likelihood of being a paying user."), icon="✅")