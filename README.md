# online-news-popularity

**README: Predicting Online News Popularity**
---------------------------------------------

### **Project Overview**

The goal of this project is to develop a machine learning model that predicts the popularity of online news articles based on their features. Using the Online News Popularity dataset from the UCI Machine Learning Repository, this project explores which factors most significantly influence an article's social media engagement, as measured by the number of shares.

### **Problem Statement**

In today's digital landscape, understanding what makes an online news article popular is crucial for content creators and publishers. This project seeks to answer the question: **Which features of online news articles contribute most to their popularity, and can we accurately predict the number of shares an article will receive?**

By building a predictive model, we aim to provide insights into the attributes that make a news article more likely to be shared, thus helping publishers optimize their content for better reach and engagement.

### Data Description

The dataset used in this project is located in the `data` directory:
- `data/OnlineNewsPopularity.csv`: The Online News Popularity dataset, which contains 39,644 observations of news articles from various publishers. Each article is characterized by 61 features, including:

-   **Content Attributes**: Word count, title length, presence of specific keywords, etc.
-   **Metadata**: Day of the week published, category of news (e.g., technology, lifestyle), etc.
-   **Engagement Metrics**: Number of shares on various social media platforms.

The dataset provides a rich source of information to understand and predict article popularity.

### **Project Approach**

1.  **Data Exploration and Cleaning**: We began by thoroughly exploring the dataset to understand the distribution of features, identify missing values, and detect any anomalies. This step also included visualizing the data to uncover patterns and relationships that could inform our modeling process.

2.  **Feature Engineering**: Several new features were created to enhance the model's predictive power:

    -   Interaction terms between content features (e.g., multiplying content and title word counts).
    -   Text-based features like sentiment polarity extracted using TextBlob.
    -   Temporal features such as the day of the week.
    -   Log transformation of highly skewed features (like the number of shares) to reduce skewness.

3.  **Model Building and Hyperparameter Tuning**: We built multiple machine learning models, including Random Forest, Gradient Boosting, Support Vector Regression (SVR), Neural Networks (MLP Regressor), XGBoost, LightGBM, CatBoost, Elastic Net, K-Nearest Neighbors, Decision Tree Regressor, Lasso, and Ridge Regression. We applied hyperparameter tuning to optimize the performance of the LightGBM model using GridSearchCV and RandomizedSearchCV. Various parameters, such as the number of estimators, learning rate, maximum depth, number of leaves, and regularization terms, were tuned to find the best combination for minimizing the model's error.:

    These models were trained, tested, and validated using cross-validation techniques to ensure robust and generalizable performance.

5.  **Model Evaluation**: The tuned LightGBM model was evaluated using multiple metrics:

    -   Mean Absolute Error (MAE)
    -   Mean Squared Error (MSE)
    -   Root Mean Squared Error (RMSE)
    -   R² Score

### **Lessons Learned**

    -   **Hyperparameter Tuning Matters**: Tuning hyperparameters using GridSearchCV and RandomizedSearchCV significantly improved the model’s performance.
        The best combination of hyperparameters for the LightGBM model led to a reduced RMSE and higher R² score, indicating better prediction accuracy and generalization ability.

    -   Feature Importance Analysis Provides Insights: By analyzing feature importance using LightGBM’s "gain" metric, we identified key features that contribute the most to the 
        model's predictions. This analysis showed that certain content attributes, such as the number of tokens in the title and the sentiment of the title, are critical predictors 
        of article popularity. Understanding feature importance helped refine the feature set and guided further feature engineering efforts.

    -   Cross-Validation is Essential for Robust Performance: Cross-validation using k-fold (e.g., 5-fold) ensured that the model's performance was stable and generalizable across 
        different subsets of the data. It provided a more accurate measure of the model’s true error rate and helped prevent overfitting.
        
    -   Regularization Helps Prevent Overfitting: Adding regularization (e.g., reg_alpha and reg_lambda for L1 and L2 regularization) helped control overfitting in the LightGBM model, 
        especially when dealing with a large number of features. This made the model more robust and reliable.
        
    -   Monitoring and Iteration Improve Model Performance: After deploying the model, continuous monitoring and periodic retraining with new data will be necessary to maintain and improve
        the model’s performance. Model evaluation metrics will be tracked to ensure the model remains accurate and relevant over time.

### **Key Findings**

    -   **Top Predictors of Popularity**: Certain attributes, such as the length of the title, presence of keywords, and publication day, were found to be strong predictors of an article's popularity.
    -   **Model Performance**: The optimized LightGBM model provided the best overall performance, achieving a lower RMSE and higher R² score compared to other models.
    -   **Insights for Content Creators**: The analysis suggests that content creators should focus on optimizing specific attributes, such as using engaging titles and timing the release 
    of articles, to maximize social media shares.

### **Recommendations**

-   **Content Optimization**: Publishers can use the findings to strategically optimize content features like title length and keyword usage to enhance the likelihood of an article being shared.
-   **Future Research**: Further studies could incorporate additional data sources, such as user demographics or sentiment analysis, to improve the model's predictive power.

### **Next Steps**

-   **Model Refinement**: Experiment with additional machine learning algorithms and hyperparameter tuning to further improve model accuracy.
-   **Deployment**: Develop a web-based application or dashboard to allow content creators to input article features and receive real-time predictions on expected social media shares.

### **Repository Contents**

-   **Jupyter Notebooks**:

    -   `1_EDA_and_Data_Cleaning.ipynb`: Notebook containing the exploratory data analysis and data cleaning process.
    -   `2_Feature_Engineering_and_Modeling.ipynb`: Notebook detailing feature engineering, model building, and evaluation.
    -   `3_Final_Analysis_and_Results.ipynb`: Final analysis notebook summarizing the model findings and insights.

-   **README.md**: This document, providing a non-technical overview of the project.

-   **data/**: Directory containing the dataset used in the project.

-   **results/**: Directory containing the model outputs, such as prediction results and evaluation metrics.

### **Contributors**

-   **[Matt Moline]**: Data Scientist, AI/ML Enthusiast

### **Acknowledgments**

-   UCI Machine Learning Repository for providing the Online News Popularity dataset.
-   [UC Berkeley Haas Scvhool of Business AI/ML Certificate] for the guidance and resources in completing this capstone project.

### **License**

This project is licensed under the MIT License - see the LICENSE file for details.
