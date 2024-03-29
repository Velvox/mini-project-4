# Mini-project IV

### [Assignment](assignment.md)

## Project/Goals
1. Build a pipeline that accurately predicts loan approval status
2. Ship an API that accurately predicts loan approval status

## Hypothesis
1. Gender has no bearing on approval rate.
2. Marital status and education will be an indicator of income
3. If loan amount exceeds income over the loan term, should be low approval
5. Credit history will be correlated with approval

## EDA 
1. Genders are skewed heavily male
2. Substantial outliers for income and loanamount, apply log transformation

Applicant Income           |  Loan Amount
:-------------------------:|:-------------------------:
![Outliers](images/Applicantbox.png)  |  ![LoanAmount](images/LoanAmountHist.png)

3. Education and marital status is correlated with income

Education                  |  Marital Status
:-------------------------:|:-------------------------:
![Education](images/IncomeEducation.png)  |  ![Married](images/IncomeMaritalStatus.png)

4. Credit History is strongly correlated with approval

![Credit](images/CreditHistoryPlot.png)


## Process
![Process](images/process.png)
1. EDA with plots, checking distributions, unique value counts, and more.
2. Merge incomes into total income, and apply log transformation to income and LoanAmount.
3. Onehot encode categorical variables and scale numerical variables.
4. Apply PCA and Select K best to find the best features.
5. Build entirely within a pipeline
6. Grid search to find the best features, hyperparameters and classifier for the problem.
7. Export the model with pickle, use Flask to make an app and deploy on AWS.

## Results/Demo

![api_demo](images/Screen_Recording_2022-07-15_at_11_04_57_AM_AdobeExpress.gif)

Best test set accuracy: 0.86
Best test set balanced accuracy (for recall): 0.75
Achieved with hyperparameters: 
* 'classifier': SVC(C=0.5, kernel=linear)
* PCA Components: 2
* SelectKBest: 1
* Scaling: Standard Scaler


## Challanges 
* Troubleshooting in a pipeline and app proved difficult, but it has helped me learn best practices for the future.
* Learning to properly apply and sequence transformers in a pipeline for the use-case took quite a bit of time.
* I spent a few hours troubleshooting the api, that was bugged due to a versioning issue from using the wrong environment. :(
* Adding files to the AWS EC2 was surprisingly difficult, I ended up using filezilla for ease of use.

## Future Goals
* Clean formatting for the API, maybe add more functionality / an HTML gui
* Spend more time on EDA and understanding the features selected from GridSearch. Explore more chart types
