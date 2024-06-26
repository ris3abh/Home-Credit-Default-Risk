# Home Credit Default Risk

## Description
Many people struggle to get loans due to insufficient or non-existent credit histories. Unfortunately, this population is often taken advantage of by untrustworthy lenders. 

**Home Credit Group** strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. To ensure this underserved population has a positive loan experience, Home Credit utilizes a variety of alternative data—including telco and transactional information—to predict their clients' repayment abilities.

While Home Credit is currently using various statistical and machine learning methods to make these predictions, they are challenging Kagglers to help them unlock the full potential of their data. Doing so will ensure that clients capable of repayment are not rejected and that loans are given with a principal, maturity, and repayment calendar that will empower their clients to be successful.

## Dataset Description
The datasets used in this project include:

- **application_{train|test}.csv**: Static data for all applications. One row represents one loan in our data sample.
- **bureau.csv**: All client's previous credits provided by other financial institutions that were reported to Credit Bureau.
- **bureau_balance.csv**: Monthly balances of previous credits in Credit Bureau.
- **POS_CASH_balance.csv**: Monthly balance snapshots of previous POS (point of sales) and cash loans that the applicant had with Home Credit.
- **credit_card_balance.csv**: Monthly balance snapshots of previous credit cards that the applicant has with Home Credit.
- **previous_application.csv**: All previous applications for Home Credit loans of clients who have loans in our sample.
- **installments_payments.csv**: Repayment history for the previously disbursed credits in Home Credit related to the loans in our sample.
- **HomeCredit_columns_description.csv**: Descriptions for the columns in the various data files.

## Evaluation
Submissions are evaluated on the area under the ROC curve between the predicted probability and the observed target.

### Submission File
For each `SK_ID_CURR` in the test set, you must predict a probability for the `TARGET` variable. The file should contain a header and have the following format:


1. python -m venv henv
2. source henv/bin/activate
3. pip install -r requirements.txt