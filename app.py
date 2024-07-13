import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def run_two_way_anova(df, dv, factor1, factor2):
    formula = f"{dv} ~ C({factor1}) + C({factor2}) + C({factor1}):C({factor2})"
    model = stats.formula.ols(formula, data=df).fit()
    anova_table = stats.anova_lm(model, typ=2)
    return anova_table

def main():
    st.title("CSV Statistics App")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.write(df.head())

        st.subheader("Two-way ANOVA")
        dependent_var = st.selectbox("Select Dependent Variable", df.columns)
        factor1 = st.selectbox("Select First Factor", df.columns)
        factor2 = st.selectbox("Select Second Factor", df.columns)

        if st.button("Run Two-way ANOVA"):
            anova_results = run_two_way_anova(df, dependent_var, factor1, factor2)
            st.write(anova_results)

            # Visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x=factor1, y=dependent_var, hue=factor2, data=df, ax=ax)
            plt.title(f"Box Plot of {dependent_var} by {factor1} and {factor2}")
            st.pyplot(fig)

if __name__ == "__main__":
    main()

