import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import os

project_root = os.path.abspath(
  os.path.join(os.path.dirname(__file__), "..")
)

def ordinal_encode(df):
  df = df.copy()
  ordinal_cols = ["Year_of_Study", "Prompt_Engineering_Skill"]
  year_order  = ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"]
  skill_order = ["Beginner", "Intermediate", "Advanced"]
  
  encoder = OrdinalEncoder(categories=[year_order, skill_order])
  df[ordinal_cols] = encoder.fit_transform(df[ordinal_cols])
  return df

def one_hot_encode(df):
  categorical_features = ["Major_Category", "Primary_Use_Case", "Institutional_Policy", "Paid_Subscription"]
  df_encoded = pd.get_dummies(df,columns=categorical_features, dtype=float)
  return df_encoded
  

def numerical_scaling(X_train, X_test):
  numeric_cols = [
    "Pre_Semester_GPA", "Weekly_GenAI_Hours", "Traditional_Study_Hours",
    "Tool_Diversity", "Year_of_Study", "Prompt_Engineering_Skill",
  ]
  scaler = StandardScaler()
  
  X_train = X_train.copy()
  X_test = X_test.copy()
  
  X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
  X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
  
  return X_train, X_test

def preprocess_data():
  features = [
    "Pre_Semester_GPA", "Weekly_GenAI_Hours", "Primary_Use_Case",
    "Prompt_Engineering_Skill", "Tool_Diversity", "Paid_Subscription",
    "Traditional_Study_Hours", "Major_Category", "Year_of_Study",
    "Institutional_Policy",
  ]
  target="Post_Semester_GPA"
  
  
  #EXTRACT  
  df = pd.read_csv(os.path.join(project_root, "data", "raw", "ai_student_impact_dataset.csv"), usecols=features+[target])
  
  
  #NO NEED TO CLEAN DATASET
  
  #ENCODE
  df = ordinal_encode(df)
  df = one_hot_encode(df)
  
  #SPLIT
  X = df.drop(columns=target)
  y =df[target]
  X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=42)
  
  #SCALING
  X_train, X_test = numerical_scaling(X_train, X_test)
  
  return X_train, X_test, y_train, y_test 
  

  
