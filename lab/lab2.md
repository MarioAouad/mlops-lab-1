Question 1: Look at pyproject.toml and uv.lock. What changed?

In pyproject.toml we can see in the dependency list (dependencies = ["mlflow>=3.16.0", "pillow>=11.3.0",
"scikit-learn>=1.9.1", "torch>=2.14.0", "torchvision>=0.29.0",]) that we have the following liraries installed after running this command "uv add mlflow torch torchvision scikit-learn" : mlflow, torch and torchvision.

And in uv.lock we can see that we have a lot of new packages that have been added.

Question 2: What is --backend-store-uri used for? What is --default-artifact-root used for? What is the difference between the metadata mlflow stores and the artifacts it stores?

The --backend-store-uri sets the database or path where MLflow saves experiment and run metadata, while --default-artifact-root defines the default storage location for large output files like trained models and images.

MLflow metadata consists of lightweight, structured text and numbers like hyperparameters and accuracy metrics that are stored in a relational database for fast searching and comparison. In contrast, artifacts are the heavy, unstructured files generated during a run such as trained model weights, plots, and datasets which are kept in object storage like Amazon S3. Essentially, metadata tracks how a model performed, while artifacts contain the actual assets needed to redeploy or reproduce it.

P.S: i had a problem running it on the port 5000 so i ran it on 5001. i used this command:
uv run python -m mlflow server `
  --host 127.0.0.1 `
  --port 5001 `
  --workers 1 `
  --backend-store-uri sqlite:///mlflow.db `
  --default-artifact-root ./mlruns

Question 3: Why shouldn't mlflow.db and mlruns/ be tracked by git, and why shouldn't they be tracked by dvc either?

Git shouldn't track mlflow.db or mlruns/ because they contain binary database files and high-volume, auto-generated logs that cause repository bloat and severe merge conflicts. Similarly, DVC shouldn't track them because MLflow already acts as its own independent tracking system, forcing DVC to monitor these folders creates redundant tool overlap. Furthermore, because MLflow constantly updates runtime metrics and statuses, tracking them with DVC would result in continuous merge issues and the risk of accidentally overwriting  experiment history when switching code branches.

Question 4: What happens the first time you call set_experiment with a name that doesn't exist yet? Check the mlflow UI.

When we call mlflow.set_experiment() with an experiment name that does not exist yet, MLflow automatically creates a new experiment with that name and sets it as the active experiment for subsequent runs.
The first time set_experiment("food11") is called, MLflow creates a new experiment named food11 in the tracking server’s backend database and selects it as the active experiment. Calling it again reuses the existing experiment instead of creating another one.

Question 5: What is the difference between mlflow.log_param and mlflow.log_metric? Why does log_metric take a step argument and log_param doesn't?

The primary difference is that mlflow.log_param is used for static, immutable configuration settings like learning rate or batch size, whereas mlflow.log_metric tracks dynamic, numerical performance values like training loss or validation accuracy that change over time. 
Because metrics represent time-series data, log_metric includes a step argument to act as the X-axis coordinate, allowing the MLflow UI to plot line graphs that visualize how your model's performance evolves across training iterations or epochs. In contrast, parameters represent foundational conditions that remain constant throughout a single run, making an evolutionary timeline or step argument logically unnecessary.

Question 6: Open the run in the mlflow UI. Find the params, the metric charts, and the logged model artifact. Where does the model artifact actually live on disk?

The MLflow UI displays the run parameters, metric history charts, and the logged model under the Artifacts section. The model is stored locally inside the configured artifact root, mlruns. In this run, the model file is model.pth.

Question 7: In the mlflow UI, open the food11 experiment. Select these runs and click "Compare". Which learning rate gave the best val_accuracy? Is higher always better?

The learning rate 0.0001 gave the best validation accuracy, with a val_accuracy of approximately 0.7755 or 77.55%. 
Higher learning rates are not always better. In this experiment, 0.01 performed worst, while the smaller learning rate 0.0001 achieved the best result. A learning rate that is too high can cause the model to overshoot good solutions during optimization.

Question 8: Use the parallel coordinates plot on the compare page to look at lr, batch_size and val_accuracy together. What pattern do you see?

The plot shows that the smaller learning rate, 0.0001, achieved the highest validation accuracy. The learning rate 0.01 performed poorly. Batch size also affected the result: with lr=0.001, batch size 64 performed better than batch size 32.

Question 9: Sort the runs table by val_accuracy descending. Which run is the best one? Note its run ID, you'll need it in the next lab.

After sorting the runs by val_accuracy in descending order, the best run was the run with lr=0.0001 and batch_size=32. Its run ID is 8eadc6a2e9eb4237b7a81cfafccecdaa, and its validation accuracy was approximately 77.55%.