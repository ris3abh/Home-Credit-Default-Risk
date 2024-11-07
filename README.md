# Home Credit Default Risk

## Description
Many people struggle to get loans due to insufficient or non-existent credit histories. Unfortunately, this population is often taken advantage of by untrustworthy lenders. 

**Home Credit Group** strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. To ensure this underserved population has a positive loan experience, Home Credit utilizes a variety of alternative data—including telco and transactional information—to predict their clients' repayment abilities.

While Home Credit is currently using various statistical and machine learning methods to make these predictions, they are challenging Kagglers to help them unlock the full potential of their data. Doing so will ensure that clients capable of repayment are not rejected and that loans are given with a principal, maturity, and repayment calendar that will empower their clients to be successful.

## Dataset Description
The datasets used in this project include:

![Data Representation](src/home_credit.png)


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


# Home Credit Default Risk Prediction

This project predicts the probability of default for Home Credit loan applications. The model generates predictions for each `SK_ID_CURR` in the test set, estimating the probability for the `TARGET` variable.

## Setup and Installation

There are two ways to run this project:

### Method 1: Traditional Setup (Local Environment)
1. Create a virtual environment:
```bash
python -m venv henv
```

2. Activate the virtual environment:
```bash
# On Unix/macOS:
source henv/bin/activate

# On Windows:
.\henv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Refer to the code in EDA and XGB modeling notebook for initial EDA and model building.

### Method 2: Docker Setup (Recommended)

1. Make sure you have Docker installed on your system. If not, [download and install Docker](https://docs.docker.com/get-docker/).

2. Build the Docker image:
```bash
docker build -t home-credit-model .
```

3. Run the container to train the model:
```bash
docker run -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data home-credit-model
```

Alternatively, using docker-compose:
```bash
docker-compose up
```

## Project Structure
```
home_credit_default_risk/
├── code/
│   ├── Loan defaulters prediction.ipynb
│   ├── model_testing.py
│   ├── model_training.py
│   ├── new EDA.ipynb
│   └── test_results/
├── data/
│   ├── HomeCredit_columns_description.csv
│   ├── application_test.csv
│   ├── application_train.csv
│   └── ... (other data files)
├── models/
│   └── model.pkl
├── src/
│   └── home_credit.png
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Model Training

The model can be trained using either of these methods:

1. Using Python directly:
```bash
python model_training.py --train-path ../data/application_train.csv --model-output ../models/model.pkl
```

2. Using Docker:
```bash
docker run -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data home-credit-model
```

## Docker Commands Reference

- Build the image:
```bash
docker build -t home-credit-model .
```

- Run the container:
```bash
docker run -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data home-credit-model
```

- Check container status:
```bash
docker ps
```

- View container logs:
```bash
docker logs <container_id>
```

- Stop the container:
```bash
docker stop <container_id>
```

- Access container shell (for debugging):
```bash
docker exec -it <container_id> /bin/bash
```

## Notes
- The model outputs are saved to the `models` directory
- Docker volumes are used to persist the trained model and access data
- Make sure all required data files are present in the `data` directory before running the container


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