# Real Estate Financial Assistant

A full stack financial assistant for a real estate company (Prologis demo). It combines three data sources, two machine learning models hosted on AWS SageMaker, and a chatbot built on Google Agent Platform (the renamed Vertex AI), behind a Streamlit web interface.

**Live app:** https://financial-assistant-aryan.streamlit.app
**Repository:** https://github.com/aryan211103/financial-assistant

## What it does

- A conversational chatbot that interprets natural language questions and routes them to the right data source.
- Company financials pulled from SEC EDGAR (annual and latest quarter).
- A property portfolio backed by Postgres.
- Company press releases with text derived insight extraction and AWS Comprehend key phrase and sentiment extraction.
- Two ML predictions served from hosted SageMaker endpoints: house value (regression) and customer subscription (classification).
- A sidebar system status check that pings the database and both endpoints live, error handling on every external call, and latency plus raw response display on each prediction.

## Architecture and data flow

```
                         Streamlit web app (app/app.py)
                                     |
        ---------------------------------------------------------------
        |                |                   |                        |
   Chatbot tab     Properties tab      Press Releases tab        ML Predictions tab
        |                |                   |                        |
   Agent Platform   Postgres (Neon)    press_releases.json       SageMaker runtime
   ADK agent        + EDGAR cache      + AWS Comprehend          (2 endpoints)
   (Gemini)
        |
   3 function tools
   (Postgres, EDGAR, press releases)
```

The chatbot is the integrating layer. It uses Gemini on Agent Platform to read a question, pick the correct tool, fetch from Postgres, the EDGAR cache, or the press releases, and answer in plain language. The other tabs expose the same data sources and the ML endpoints directly.

## Project structure

```
financial-assistant/
  .streamlit/
    config.toml             Theme (slate palette, fonts)
  app/
    app.py                  Streamlit front end (all four tabs)
  chatbot/
    agent.py                Agent Platform ADK agent with three tools
    test_vertex.py          Standalone connectivity check
  data/
    db_setup.py             Creates Postgres tables and seeds 20 properties
    schema.sql              Table definitions on their own
    edgar_fetch.py          Pulls Prologis financials from SEC EDGAR (annual + quarterly)
    company_financials.json Cached EDGAR output
    press_releases.json     Mock press releases (6 items)
  ml/
    regression/
      train_regression.py   Trains the Random Forest on California Housing
      inference.py          SageMaker inference handler
      deploy_regression.py  Deploys the serverless endpoint
      eda_regression.py     Exploratory data analysis
    classification/
      train_classification.py
      inference_classification.py
      deploy_classification.py
      eda_classification.py
    delete_endpoint.py      Tears down endpoints after the demo
  requirements.txt          Dependencies for the hosted app
  requirements_local.txt    Dependencies for local training and deploy
```

## Data sources

### A. SEC EDGAR
`data/edgar_fetch.py` calls the SEC XBRL company facts API for Prologis (ticker PLD, CIK 0001045609). It extracts revenue, net income, and operating expenses from the latest 10-K (annual) and the most recent 10-Q (single quarter), then caches them to `company_financials.json` so the app never depends on EDGAR being reachable at run time. The SEC requires a descriptive User-Agent header, which the script sets.

### B. Postgres (Neon)
`data/db_setup.py` creates two tables in one Postgres database and seeds 20 property records.

- `properties`: property_id (primary key), address, metro_area, sq_footage, property_type
- `financials`: property_id (foreign key), revenue, net_income, expenses

The schema is also provided on its own in `data/schema.sql`. The database is hosted on Neon so the deployed app can reach it over a public address.

### C. Press releases
`data/press_releases.json` holds six mock press releases, each with a category and a summary. The app derives an insight type from the text itself (acquisition, expansion, quarterly update, leasing, sustainability) and shows a count summary. AWS Comprehend extracts key phrases and sentiment from the same text, and a summarize button routes the releases through the chatbot for a short generated summary.

## Machine learning models

### Regression: house value
- Dataset: California Housing from scikit-learn, 20,640 rows, 8 numeric features.
- Model: RandomForestRegressor, 100 trees.
- No feature scaling, because tree models split on thresholds and do not need it.
- Test metrics: RMSE 0.5057, MAE 0.3276, R2 0.8049.
- Endpoint: `housing-regression` (serverless).

### Classification: customer subscription
- Dataset: UCI Bank Marketing (id 222), 45,211 rows, 16 features, 11.7 percent positive rate.
- Model: LogisticRegression inside a scikit-learn Pipeline (OneHotEncoder plus StandardScaler).
- Numeric features are standardized because logistic regression is scale sensitive.
- Test metrics: Accuracy 0.9015, Precision 0.6462, Recall 0.3488, F1 0.453. Accuracy is high mainly because the data is imbalanced, so precision and recall are the meaningful numbers.
- Endpoint: `bank-subscription` (serverless).

## How the chatbot routes queries

The agent in `chatbot/agent.py` is built with the Google Agent Development Kit and runs Gemini (`gemini-2.5-flash`) on Agent Platform. It is given three tools, each a plain Python function with a descriptive docstring:

- `query_properties` for specific buildings and their financials (queries Postgres).
- `get_company_financials` for company wide revenue, net income, and operating expenses, annual or latest quarter (reads the EDGAR cache).
- `search_press_releases` for announcements such as acquisitions and expansions (reads the press releases).

ADK turns each function's type hints and docstring into a schema the model can call. When a question arrives, Gemini reads the descriptions, decides which tool fits, the Runner executes that function, feeds the result back to the model, and Gemini writes the final natural language answer. Routing is done by the model, not by hardcoded rules, and the agent can call more than one tool in a single turn to combine sources. The synchronous entry point retries on transient Agent Platform 429 RESOURCE_EXHAUSTED errors with exponential backoff so brief rate limits do not surface as failures.

## Local setup

Prerequisites: Python 3.12, a Neon Postgres database, an AWS account with the two SageMaker endpoints deployed, and gcloud authenticated for Agent Platform.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set the database connection string
export DATABASE_URL="postgresql://user:password@host.neon.tech/dbname?sslmode=require"

# 3. Seed the database and refresh the EDGAR cache
python data/db_setup.py
python data/edgar_fetch.py

# 4. Authenticate for Agent Platform (local uses your gcloud login)
gcloud auth application-default login
gcloud config set project financial-assistant-498905

# 5. Run the app
streamlit run app/app.py
```

AWS credentials are read from your standard `~/.aws` configuration, so SageMaker and Comprehend calls work once `aws configure` is set with a region of us-east-1.

## Cloud setup

### AWS SageMaker
Train, then deploy each model as a serverless endpoint. Done from a dedicated environment with the SageMaker v2 SDK and scikit-learn pinned to 1.2.2 to match the serving container.

```bash
python -m venv ~/sm-deploy && source ~/sm-deploy/bin/activate
pip install "sagemaker<3" scikit-learn==1.2.2 numpy==1.26.4 pandas==2.2.2 boto3 ucimlrepo

# regression
cd ml/regression
python train_regression.py
python deploy_regression.py

# classification
cd ../classification
python train_classification.py
python deploy_classification.py
```

Each deploy uploads the model tarball to S3, wraps it in the prebuilt scikit-learn 1.2-1 container with the matching inference handler, and creates a serverless endpoint. A SageMaker execution role is required and its ARN is set in each deploy script.

Note on the classification handler: the first deployment returned a 500 because the OneHotEncoder called numpy isnan on string categories when a categorical column arrived as a numeric dtype. The fix is in `inference_classification.py`, where `predict_fn` reindexes the input to the training columns and forces every categorical column to a string and every numeric column to a number before the model runs. The training script saves the categorical and numeric column lists with the model so the handler knows which is which.

### GCP Agent Platform (Vertex AI)
The chatbot needs the Agent Platform API enabled on the project (service id `aiplatform.googleapis.com`) and either gcloud Application Default Credentials (local) or a service account key (hosted). The agent sets `GOOGLE_GENAI_USE_VERTEXAI=TRUE`, the project, and the region so the google-genai library routes to Agent Platform. The hosted app uses the `vertex-streamlit` service account, which has the Agent Platform User role.

### Hosting on Streamlit Community Cloud
The app is deployed from the GitHub repo with `app/app.py` as the entry point and Python 3.12. The following secrets are set in the Streamlit dashboard:

- `DATABASE_URL`: the Neon connection string.
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`: for SageMaker and Comprehend.
- `GCP_SA_KEY_JSON`: the full service account key JSON, which the agent writes to a temp file and points the credentials at.

## Multi cloud

The system spans two clouds: GCP Agent Platform runs the agent, and AWS runs both the SageMaker endpoints and Comprehend. AWS Comprehend performs live key phrase and sentiment extraction on the press releases. An AWS Bedrock summarization integration was also attempted, but this new AWS account ships with a near zero daily Bedrock token quota that requires a support increase, so it is not invoked live and summarization is routed through the chatbot instead.

## Notes and honest caveats

- Regression: the random train test split on data with latitude and longitude features makes the test score mildly optimistic due to spatial autocorrelation. A region based split would be a more honest estimate. There is no strict data leakage.
- Classification: call duration is the strongest predictor but is arguably leaky, since it is only known after the call. The UCI documentation recommends dropping it for a realistic model. Recall is low at the default threshold, which is expected for imbalanced data.
- EDGAR quarterly figures come from 10-Q filings, which report discrete quarters. Q4 usually appears only in the 10-K, so the latest quarter shown is the most recent one actually filed on a 10-Q.

## Cost note

Both SageMaker endpoints are serverless and scale to zero, so idle cost is negligible. Run `python ml/delete_endpoint.py` to tear them down after grading.
