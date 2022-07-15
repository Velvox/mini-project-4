# Mini-project IV

### [Assignment](assignment.md)

## Project/Goals
1. Build a pipeline that accurately predicts loan approval status
2. Ship an API that accurately predicts loan approval status

## Hypothesis
1. Gender has no bearing on approval rate.
2. Marital status will have an impact on approval status
3. If loan amount exceeds income over the loan term, should be low approval

## EDA 
1. Substantial outliers for income and loanamount, apply log transformation
2. Genders are skewed heavily male
3. Most applicants are approved.
4. Strong correlation between education and income
5. Higher the income, the better the approval rate.


## Process
1. EDA with histograms, checking distributions, unique value counts, and more.
2. Merge incomes into total income, and apply log transformation to income and LoanAmount.
3. Onehot encode categorical variables and scale numerical variables.
4. Apply PCA and Select K best to find the best features.
5. Use a pipeline and grid search to find the best features, hyperparameters and classifier for the problem.
6. Export the model with pickle, use Flask to make an app and deploy on AWS.

## Results/Demo
Best test set accuracy: 0.8617886178861789
Achieved with hyperparameters: {'classifier': SVC(C=0.5), 'classifier__C': 0.5, 'preprocessing__cat_transform__pca__n_components': 2, 'preprocessing__num_transform__scaling': StandardScaler(), 'preprocessing__num_transform__select_best__k': 1}

## Challanges 
-Using a full pipeline was challenging but helped me learn best practices for the future.
-Learning to properly apply and sequence transformers for the use-case took quite a bit of time.
-Troubleshooting the API is very time-consuming. Difficult to troubleshoot each step.
-Using the incorrect environment caused errors that took hours to solve

## Future Goals
-Clean formatting for the API