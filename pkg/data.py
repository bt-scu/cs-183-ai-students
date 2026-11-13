import pandas as pd
import os

project_root = os.path.abspath(
  os.path.join(os.path.dirname(__file__), "..")
)

def load_raw_csv():
  return pd.read_csv(os.path.join(project_root, "data", "raw", "ai_student_impact_dataset.csv"))