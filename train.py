import argparse
import os
import pandas as pd
import lightgbm as lgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from rich.console import Console

console = Console()

class ChurnModelTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = lgb.LGBMClassifier(
            colsample_bytree=1.0, learning_rate=0.05, max_depth=-1,
            n_estimators=100, num_leaves=31, subsample=0.8, objective="binary"
        )

    def load_and_preprocess_data(self):
        console.print("[bold green]Loading dataset...[/bold green]")
        df = pd.read_csv(self.data_path)
        
        console.print("[bold green]Splitting data...[/bold green]")
        X = df.drop(columns=["Churn"])
        y = df["Churn"]
        
        console.print("[bold green]Processing categorical variables...[/bold green]")
        cat_columns = X.select_dtypes(include=["object"]).columns
        for col in cat_columns:
            X[col] = X[col].astype("category")
        
        return train_test_split(X, y, test_size=0.2, random_state=42), cat_columns

    def train(self):
        (X_train, X_test, y_train, y_test), cat_columns = self.load_and_preprocess_data()
        console.print("[bold cyan]Training model...[/bold cyan]")
        self.model.fit(X_train, y_train, categorical_feature=list(cat_columns))
        
        console.print("[bold cyan]Evaluating model...[/bold cyan]")
        y_pred = self.model.predict(X_test)
        console.print(f"[bold yellow]Accuracy:[/bold yellow] {accuracy_score(y_test, y_pred):.4f}")
        
        self.save_model()

    def save_model(self):
        os.makedirs("assets", exist_ok=True)
        model_path = "assets/lightgbm_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        console.print(f"[bold magenta]Model saved to:[/bold magenta] {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="Path to the dataset CSV file")
    args = parser.parse_args()

    trainer = ChurnModelTrainer(args.data_path)
    trainer.train()
