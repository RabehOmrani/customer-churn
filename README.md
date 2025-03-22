### 📌 **Customer Churn Prediction**  
**Telco Customer Churn Prediction**  
A machine learning project using LightGBM to predict customer churn based on Telco Customer data.  

### 📂 **Project Structure**  
- `assets/` → Stores the trained model (`lightgbm_model.pkl`).  
- `train.py` → Script for training the model with preprocessing and saving it.  
- `main.py` → Streamlit app for testing the model with user input.  
- `requirements.txt` → List of dependencies to install.  
- `README.md` → Documentation for the project.  

### 📊 **Dataset**  
[Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  

### 🚀 **Installation & Usage**  
```bash
git clone <repo-url>
cd <repo-name>
pip install -r requirements.txt
python train.py data.csv  # Train the model
streamlit run main.py  # Launch the app
```

### ⚙ **Technologies Used**  
- Python  
- LightGBM  
- Streamlit  
- Pandas, NumPy  
- Rich (for logging)  
