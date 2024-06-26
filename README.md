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
4. Refer to the code in EDA and XGB modeling notebook for initial EDA and model building


<h2 style="text-align: center;">Model Performance</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Train</th>
            <th>Test</th>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td>0.890630</td>
            <td>0.889550</td>
        </tr>
        <tr>
            <td>Precision</td>
            <td>0.221430</td>
            <td>0.214374</td>
        </tr>
        <tr>
            <td>Recall</td>
            <td>0.140572</td>
            <td>0.139826</td>
        </tr>
        <tr>
            <td>ROC AUC</td>
            <td>0.548564</td>
            <td>0.547492</td>
        </tr>
    </table>

Overall, the model shows reasonably consistent performance across the training and test sets, as indicated by similar values for accuracy, precision, recall, and ROC AUC. This consistency suggests that the model is not overfitting excessively to the training data and is generalizing reasonably well to unseen data. However, the low values of precision and recall indicate that the model may need further tuning or feature engineering to improve its ability to correctly classify positive cases (defaults) while minimizing false positives and false negatives.

The fact that, even the SMOTING algorithm was not helpful to improve the model performance, it is possible that the model is not able to capture the underlying patterns in the data effectively. This could be due to the presence of noise or irrelevant features in the dataset, or it could be a limitation of the chosen algorithm. Further exploration of the data and experimentation with different algorithms and hyperparameters may be necessary to improve the model's performance.