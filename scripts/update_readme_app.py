with open("README.md", "r", encoding="utf-8") as f:
    text = f.read()

app_section = """---

## 🌐 Interactive Streamlit Web App (`app.py`)

A full interactive web application is included to allow retention and customer success teams to simulate customer churn risk in real-time.

### Key Features:
* **Interactive Sliders & Dropdowns**: Configure any customer's tenure, billing method, contract type, and subscription services using the exact categories from the dataset.
* **Direct Pipeline Ingestion**: The input profile is fed directly into the serialized production pipeline (`models/churn_model.joblib`), executing the fitted scaling and encoding automatically.
* **Risk Categorization**:
  * **Low Risk (< 35%)**: Green badge; regular relationship maintenance.
  * **Medium / Borderline (35% – 50%)**: Amber badge; proactive monitoring and low-cost retention touches.
  * **High Risk (≥ 50%)**: Red badge; immediate intervention required.
* **Dynamic Action Plan**: Generates plain-language, personalized business recommendations tailored specifically to the customer's risk factors (e.g., month-to-month upgrade incentive, electronic check autopay credit, complimentary TechSupport bundle).

### How to Run the App Locally:
```bash
# Ensure dependencies are installed
pip install -r requirements.txt

# Launch the Streamlit application
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## ☁️ Deploying to Streamlit Community Cloud (Step-by-Step)

You can deploy this application for free to share a live URL on your portfolio and LinkedIn:

1. **Push to GitHub**:
   * Commit and push your code to a public GitHub repository:
   ```bash
   git add .
   git commit -m "feat: complete customer churn prediction system and streamlit app"
   git push origin main
   ```
2. **Sign in to Streamlit Cloud**:
   * Visit [share.streamlit.io](https://share.streamlit.io/) and log in using your GitHub account.
3. **Deploy New App**:
   * Click **"Create app"** (or **"New app"**).
   * Select **"I already have an app"**.
   * Choose your repository: `your-username/customer-churn-prediction`.
   * Set **Branch**: `main`.
   * Set **Main file path**: `app.py`.
   * Set **App URL**: choose an optional custom subdomain (e.g., `telco-churn-predictor.streamlit.app`).
4. **Deploy**:
   * Click **"Deploy!"**.
   * Streamlit Community Cloud will automatically detect `requirements.txt`, install all pinned dependencies (`scikit-learn==1.9.1`, `pandas==3.0.6`, `numpy==2.5.3`, `streamlit>=1.30.0`), load `models/churn_model.joblib`, and launch your live application.

"""

if "## 🌐 Interactive Streamlit Web App" not in text:
    text = text.replace("## 📁 Repository Structure", app_section + "## 📁 Repository Structure")
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(text)
    print("README.md successfully updated with Streamlit and Cloud Deployment guide!")
else:
    print("Section already present.")
