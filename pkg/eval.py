from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

#Use for XGboost and Linear regression
def regression_metrics(y, y_pred):
  mse=mean_squared_error(y, y_pred)
  return {
    "MSE": mse,
    "RMSE": math.sqrt(mse),
    "MAE": mean_absolute_error(y, y_pred),
    "R2": r2_score(y, y_pred)
  }
