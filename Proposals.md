#### Analysis 1
#### Research Question: 
Has the level of emotional loneliness experienced by Canadians changed from 2021 to 2022?

#### Variables:
LONELY_dejong_emotional_loneliness_sub_scale_score from both the 2021 cross-sectional survey and the 2022 cohort survey, reported by the same individuals (same ID across two surveys). This score assesses the level of emotional loneliness experienced by individuals.

Relevance: Emotional loneliness is a critical indicator of social connections. 

#### Visualizations:
Boxplots for both the loneliness scores of 2021 and 2022.
Histogram for the paired difference in scores (2022 score minus 2021 score for the same individual).Boxplots effectively display medians, quartiles, and potential outliers, making them good for comparing the central tendencies of scores between two years. A histogram visualizes the distribution, sample size, modality, and potential skew of the differences, providing an intuitive assessment.
#### Analysis:
Using data from participants who completed both surveys, we will calculate the difference in emotional loneliness scores for each individual. A 95% confidence interval for the mean difference (μ) will be constructed via bootstrapped resampling.

#### Assumptions:
The differences are independent and identically distributed.
The sample of paired differences is representative of the population.
The differences in emotional loneliness scores are independent from one individual to another.

#### Hypotheses:
Null Hypothesis (H₀): μ=0 (No significant difference in mean emotional loneliness scores between 2021 and 2022).

Alternative Hypothesis (H₁): μ!=0 (significant difference). 

#### Possible Results:
Significant Decrease: The 95% confidence interval does not include zero and is negative.

Significant Increase: The 95% confidence interval does not include zero and is positive.

No Significant Change: The 95% confidence interval includes zero. We fail to reject the null hypothesis.

Relevance to Question: Results help us infer emotional loneliness among Canadians has significantly increased, decreased, or seen no significant change from 2021 to 2022.

#### Analysis 2
#### Research Question: 
Is there an association between the number of hours Canadians work per week and their burnout levels in 2022?
#### Variables:
Outcome: WELLNESS_malach_pines_burnout_measure_score. This score assesses the level of burnout experienced by individuals, which can affect mental health, job performance, and social relationships.

Predictor: WORK_hours_per_week. This Represents the total number of hours an individual works in a week. Longer work hours may contribute to increased stress and burnout.

Relevance: Burnout negatively affects well-being and social connections, aligning with the Canadian Social Connection Survey (CSCS) project's goals.

#### Visualizations:
Kernel Density Estimation (KDE) plots will be used for both variables. They are useful because they visualize the distribution, sample size, and modality while providing smooth and visually appealing representation, approportae for numeric variables that have many different values.

#### Analysis:
Using data from the 2022 cross-sectional survey, a least squares linear regression model will be fitted with WORK_hours_per_week predicting WELLNESS_malach_pines_burnout_measure_score. A 95% confidence interval for the slope coefficient (beta1) will be constructed via bootstrapped resampling.

#### Assumptions:
Linearity, independence of observations, normality of residuals, homoscedasticity, and minimal measurement error in WORK_hours_per_week must be assessed before conclusions are drawn.

#### Hypotheses:
Null Hypothesis (H₀): beta1=0 (There is no linear association between work hours and burnout levels).As work hours increase, burnout levels also tend to increase.

Alternative Hypothesis (H₁): beta1!=0 (Significant linear association between work hours and burnout levels).

#### Possible Results:
Statistically significant negative association: The 95% CI for beta1 does not include zero and is negative. As work hours increase, burnout levels tend to decrease.

Statistically significant positive association: The 95% CI does not include zero and is positive. We fail to reject the null hypothesis.

No statistically significant association: The 95% CI includes zero. We fail to reject the null hypothesis that there is no linear relationship between work hours and burnout levels in the studied sample of Canadians. 

Relevance to Question: These results will help us infer if work hours are positively, negatively, or not linearly associated with burnout levels in Canadians. 

#### Analysis 3
#### Research Question:
Is there an association between the time Canadians spend on social media and their self-reported negative body image in 2022?
#### Variables:
PSYCH_body_self_image_questionnaire_think_unattractive. **ordinal variable** from the 2022 cross-sectional survey, assessing agreement with the statement *"I think my body is unattractive."* through 5 categorical responses.

CONNECTION_social_media_time_per_day. Also ordinal and from the 2022 cross survey, assessing time spent on social media with 6 categorical responses.
#### Visualizations:
Bar plots for both ordinal variables. Bar plots are useful for visualizing the distribution of ordinal categorical variables, because they represent the count of values of each category as the height of a bar.
#### Analysis:
Using data from the 2022 cross-sectional survey, we will perform **Fisher's Exact Test** on the 5×6 contingency table constructed from the two variables. Due to the computational intensity of calculating each possible combination, we will utilize a **Monte Carlo simulation** to estimate the p-value accurately. This method involves generating a large number of simulated contingency tables under the null hypothesis and determining the proportion of tables that are as or more extreme than the observed table. 
#### Assumptions:
Independence of observations, Mutually exclusive categories, Random sampling, Independent observations and Fixed row & column totals.
#### Hypotheses:
Null Hypothesis (H₀):There is no association between time spent on social media and negative body image.

Alternative Hypothesis (H₁):There is an association between time spent on social media and negative body image.
#### Possible Results
Statistically significant association: p-value <= 0.05. 
No statistically significant association: p-value > 0.05. 
#### Relevance to Question:
These results will help us determine if time spent on social media is associated with negative body image among Canadians.



```python
#Analysis 1
import pandas as pd
import numpy as np

# Read in the variable names and data
cols = pd.read_csv("https://raw.githubusercontent.com/pointOfive/stat130chat130/main/CP/var_names.csv")
data = pd.read_csv(
    "https://raw.githubusercontent.com/pointOfive/stat130chat130/main/CP/CSCS_data_anon.csv",
    na_values=["9999", "", " ", "Presented but no response", "NA"]
)

# Remove empty columns
empty = (data.isna().sum() == data.shape[0])
data = data[empty.index[~empty]]  # Keep non-empty columns only

# Remove cases recommended for removal
dataV2 = data[data['REMOVE_case'] == 'No'].copy()

# Select participants who are part of the cohort data
dataV2_cohort = dataV2[dataV2['SURVEY_cohort_participant']].copy()

# Remove data from the year 2023
dataV2_cohortV2 = dataV2_cohort[dataV2_cohort['SURVEY_collection_year'] != 2023].copy()

# Remove columns with too many missing values
missingness_limit = 100  # Retain columns with fewer missing values
columns2keep = dataV2_cohortV2.isna().sum() < missingness_limit
columns2keep = columns2keep.index[columns2keep]
dataV2_cohortV3 = dataV2_cohortV2[columns2keep].copy()

# Exclude participants who took more than 30 seconds per question
dataV2_cohortV4 = dataV2_cohortV3[dataV2_cohortV3['Secs_per_q'] < 30].copy()

# Create a survey year identifier
dataV2_cohortV4['SURVEY_YEAR'] = (
    dataV2_cohortV4['SURVEY_collection_type'] + ' ' +
    dataV2_cohortV4['SURVEY_collection_year'].astype(str)
)

# Focus on the variable of interest
variables_of_interest = [
    'LONELY_dejong_emotional_loneliness_sub_scale_score'
]

# Reshape the data to long format
dataV2_cohortV4_wide = dataV2_cohortV4.melt(
    id_vars=['UNIQUE_id', 'SURVEY_YEAR'],
    value_vars=variables_of_interest
)

# Append the survey year to variable names
dataV2_cohortV4_wide['variable'] = (
    dataV2_cohortV4_wide['variable'] + ' (' + dataV2_cohortV4_wide['SURVEY_YEAR'] + ')'
)

# Pivot the data to wide format to align variables side by side for each participant
dataV2_cohortV4_wide = dataV2_cohortV4_wide.pivot(
    index='UNIQUE_id',
    columns='variable',
    values='value'
)

# Consider fully observed data only
dataV2_cohortV4_wideV2 = dataV2_cohortV4_wide.dropna()

# Extract the emotional loneliness scores for 2021 and 2022
loneliness_2021 = dataV2_cohortV4_wideV2[
    'LONELY_dejong_emotional_loneliness_sub_scale_score (cross 2021)'
]
loneliness_2022 = dataV2_cohortV4_wideV2[
    'LONELY_dejong_emotional_loneliness_sub_scale_score (cohort 2022)'
]

# Calculate the difference in scores for each participant
score_difference = loneliness_2022 - loneliness_2021

# Bootstrap resampling to compute the 95% confidence interval
bootstrap_means = []
for _ in range(10000):
    resampled_difference = score_difference.sample(frac=1, replace=True)
    bootstrap_means.append(resampled_difference.mean())

# Calculate the confidence interval bounds
lower_bound = np.percentile(bootstrap_means, 2.5)
upper_bound = np.percentile(bootstrap_means, 97.5)

print(f'95% Confidence Interval for the mean difference: ({lower_bound}, {upper_bound})')
print("Based on preliminary analysis, the 95% confidence interval for the mean difference is entirely negative, suggesting a significant decrease in emotional loneliness from 2021 to 2022.")
```

    /tmp/ipykernel_277/1917690716.py:6: DtypeWarning: Columns (129,408,630,671,689,978,1001,1002,1006,1007,1008,1080,1113,1115,1116,1117,1118,1119,1120,1121,1124,1125,1126,1127,1128,1213,1214,1215,1216,1217,1218,1263,1266,1342,1343,1344,1345,1346,1347,1348,1349,1390,1391,1393,1439,1442,1463,1546,1549,1552,1555,1558,1561) have mixed types. Specify dtype option on import or set low_memory=False.
      data = pd.read_csv(


    95% Confidence Interval for the mean difference: (-0.3682539682539683, -0.1873015873015873)
    Based on preliminary analysis, the 95% confidence interval for the mean difference is entirely negative, suggesting a significant decrease in emotional loneliness from 2021 to 2022.



```python
#Analysis 3
from IPython.display import Image

Image(url='https://raw.githubusercontent.com/zeptabot/STA130/refs/heads/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-31%20220256.jpg')


```




<img src="https://raw.githubusercontent.com/zeptabot/STA130/refs/heads/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-31%20220256.jpg"/>


