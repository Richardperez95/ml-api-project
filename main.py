{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4b0a5737-4092-4be9-83ef-037791495861",
   "metadata": {},
   "outputs": [],
   "source": [
    "from fastapi import FastAPI\n",
    "import joblib\n",
    "import numpy as np\n",
    "\n",
    "# --- Create the API application ---\n",
    "app = FastAPI(title=\"Churn Prediction API\")\n",
    "\n",
    "# --- Load the saved model (from the .pkl file) ---\n",
    "model = joblib.load(\"churn_model.pkl\")\n",
    "\n",
    "# --- Endpoint 1: Home page (just confirms the API is alive) ---\n",
    "@app.get(\"/\")\n",
    "def home():\n",
    "    return {\"message\": \"Churn Prediction API is running\"}\n",
    "\n",
    "# --- Endpoint 2: Prediction (this is the one that does the work) ---\n",
    "@app.post(\"/predict\")\n",
    "def predict(data: dict):\n",
    "    # Pull the customer's features out of the incoming request\n",
    "    features = np.array([[\n",
    "        data[\"monthly_spend\"],\n",
    "        data[\"tenure_months\"],\n",
    "        data[\"support_calls\"]\n",
    "    ]])\n",
    "\n",
    "    # Use the model to get a churn probability\n",
    "    churn_prob = model.predict_proba(features)[0][1]\n",
    "\n",
    "    return {\n",
    "        \"churn_probability\": round(float(churn_prob), 4)\n",
    "    }\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:anaconda3] *",
   "language": "python",
   "name": "conda-env-anaconda3-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
