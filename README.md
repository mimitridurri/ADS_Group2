# ADS_Group2 
## NBA MVP prediction based on basic stats scraped from Basketball-Reference
Preview seems not to be working in Github, use https://nbviewer.jupyter.org/ instead.


## Group Members
- Dimitri Murri (murridim@students.zhaw.ch)
- Lukas Barth (barthluk@students.zhaw.ch)
- Oliver Müller (muellol7@students.zhaw.ch)

## Objective
This project aims to predict the Most Valuable Player (MVP) of the NBA for the 2023-2024 season using machine learning techniques and data analysis.

## Project Overview

1. **Data Collection:**
   - **NBA Player Statistics:**
     - Utilized the `nba_api` to gather player statistics such as points, rebounds, assists, steals, blocks, and other relevant metrics.
     - Extracted data for the current season to ensure the model uses the most recent information.
   - **Web Scraping:**
     - Employed Selenium to automate the web browser and scrape additional player data from NBA's official website and other reliable sources.
     - Used BeautifulSoup to parse HTML content and extract the necessary information.
     - Data collected includes advanced metrics, player profiles, and historical performance data.

2. **Data Preprocessing:**
   - **Cleaning Data:**
     - Removed duplicates and irrelevant entries.
     - Handled missing values by either filling them with appropriate statistics or removing the affected rows/columns.
   - **Transforming Data:**
     - Converted categorical variables into numerical formats using one-hot encoding.
     - Normalized numerical features to ensure all features contribute equally to the model's performance.
   - **Feature Engineering:**
     - Created new features based on domain knowledge, such as per-minute statistics, player efficiency ratings, and impact metrics.

3. **Exploratory Data Analysis (EDA):**
   - **Visualizations:**
     - Used Seaborn and Matplotlib to create various plots (e.g., histograms, scatter plots, box plots) to visualize the distributions and relationships of the data.
   - **Statistical Analysis:**
     - Conducted correlation analysis to identify relationships between different features and the target variable (MVP status).
     - Used pair plots and heatmaps to visually inspect correlations and potential collinearity.

4. **Feature Selection:**
   - **Mutual Information Regression:**
     - Applied mutual information regression to measure the dependency between each feature and the target variable.
     - Selected the most influential features for the prediction model to enhance accuracy and reduce overfitting.
   - **Recursive Feature Elimination:**
     - Implemented recursive feature elimination to iteratively select features by training the model and removing the least important ones.

5. **Modeling:**
   - **Model Selection:**
     - Considered various models such as RandomForestRegressor and LinearRegression.
   - **Training Models:**
     - Split the data into training and testing sets to evaluate model performance.
     - Trained the models on the training set and optimized hyperparameters using grid search and cross-validation.
   - **Evaluating Models:**
     - Assessed model performance using metrics like Mean Absolute Error (MAE) and R-squared (R²) to ensure robustness and accuracy.
     - Compared the performance of different models to select the best-performing one.

6. **Model Interpretation:**
   - **SHAP (SHapley Additive exPlanations):**
     - Used SHAP values to interpret the contribution of each feature to the model’s predictions.
     - Created SHAP summary plots to visualize the importance and impact of features.

7. **Storing Results:**
   - **Database Integration:**
     - Saved data into a MySQL on AWS database using `pymysql`.
     - Ran queries on the MySQL database to validate and manipulate the stored data.

## Tools and Libraries
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn (RandomForestRegressor, LinearRegression)
- **Statistical Analysis:** Statsmodels
- **Web Scraping and APIs:** Selenium, BeautifulSoup, nba_api
- **Database:** pymysql
- **Model Interpretability:** SHAP
- **Progress Tracking:** tqdm

## Repository Structure
- `data/`: Contains raw and processed data files.
- `notebooks/`: Jupyter notebooks with code and analysis.
- `scripts/`: Python scripts for data collection, preprocessing, and modeling.
- `results/`: Output files and model results.
- `README.md`: Project summary and instructions.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/nba-mvp-prediction.git
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook:
   ```bash
   jupyter notebook notebooks/Group2.ipynb
   ```

## Contact
For any questions or collaboration, feel free to reach out to the group members via their provided emails.
