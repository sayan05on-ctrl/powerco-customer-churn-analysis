PowerCo Customer Churn Prediction Report

Project Overview

This project was completed as part of the BCG Data Science Virtual Experience Program. 
The objective was to investigate customer churn at PowerCo and determine whether customer price sensitivity is a significant factor influencing customer attrition. 
The project followed an end-to-end data science workflow, including exploratory data analysis, feature engineering, predictive modeling, and business recommendation generation.

Business Problem

PowerCo observed an increasing number of customers switching to competing energy providers.
The company hypothesized that customers had become more sensitive to price changes and that pricing was a key driver of churn. 
The objective of this project was to validate this hypothesis and develop a predictive model capable of identifying customers at risk of leaving.

Exploratory Data Analysis

The initial phase focused on understanding customer characteristics, pricing behavior, and consumption patterns. 
Data quality checks, descriptive statistics, and visualizations were used to examine customer distributions and identify potential relationships with churn. 
The analysis revealed substantial variation in energy consumption, customer tenure, profitability, and pricing variables. 
Correlation analysis suggested that several price-related features may influence customer behavior and warranted further investigation.

Feature Engineering

Additional features were created to improve predictive performance and capture customer behavior more effectively.
These included average yearly price changes, average six-month price changes, price sensitivity measures, customer tenure, profitability ratios, and consumption-related metrics. 
These engineered variables provided additional insight into how customers respond to changes in pricing and energy usage.

Predictive Modeling

A Random Forest Classifier was developed to predict customer churn. 
The dataset was prepared through categorical variable encoding, train-test splitting, and feature alignment. 
Model performance was evaluated using Accuracy, Precision, Recall, F1-Score, ROC-AUC, and a Confusion Matrix.
Feature importance analysis was also conducted to identify the variables that contributed most significantly to churn prediction.

Key Findings

The model successfully identified important patterns associated with customer churn. 
Feature importance analysis indicated that pricing-related variables contribute meaningfully to churn prediction, supporting the hypothesis that some customers are highly sensitive to price changes. 
Customer tenure, energy consumption, and profitability metrics were also found to influence churn behavior. These findings suggest that customer retention strategies should consider both pricing and behavioral characteristics.

Business Impact

The developed model enables PowerCo to proactively identify customers at risk of churn before contract renewal periods. 
This capability allows the company to implement targeted retention campaigns, personalized pricing strategies, and customer engagement initiatives. By focusing efforts on high-risk customers, PowerCo can improve customer lifetime value, reduce revenue loss, and allocate retention resources more efficiently.

Recommendations

1. Deploy the churn prediction model within customer retention workflows.
2. Prioritize high-risk and price-sensitive customers for proactive engagement.
3. Offer targeted discounts or incentives before contract renewal dates.
4. Continuously monitor customer behavior and retrain the model as new data becomes available.
5. Expand future analyses by incorporating customer service interactions and external market factors.

Conclusion

This project demonstrates how machine learning and data-driven decision-making can be used to address customer churn in the energy sector. 
Through exploratory analysis, feature engineering, and predictive modeling, the study provided evidence that price sensitivity plays a meaningful role in customer attrition. 
The resulting model offers actionable insights that can support customer retention strategies and improve long-term business performance.
