# Lessons

Machine learning on the lakehouse data, one step at a time. Every lesson has the idea, a task you write yourself in a
notebook in the Databricks Git folder, and questions to check you can explain it in an interview. Commit your
notebook when the "done when" line is true, then ask for the next lesson. The later ones are written after the
earlier ones run, so they can use your real numbers.

| # | Lesson | You learn | Status |
|---|---|---|---|
| 1 | [The problem and the training table](01-the-problem.md) | target, unit of prediction, leakage, features in SQL | ready |
| 2 | Splitting by time and beating baselines | why random splits lie on time series, MAE, baselines in pandas | next |
| 3 | First model: LightGBM, tracked in MLflow | gradient boosting, categorical features, experiment tracking | |
| 4 | Where the model is wrong | error by mode/line/hour, feature importance, SHAP | |
| 5 | Correcting STIB's countdown | a second target from the raw snapshots, joining predictions to outcomes | |
| 6 | From notebook to product | register the model in Unity Catalog, a scoring job, predictions read by Power BI | |

The number to beat (lesson 2 makes it yours): **0.94 min** mean absolute error, from a plain average per
stop, line and hour of day, trained Tue to Thu and tested on Friday 16 Oct 2026.
