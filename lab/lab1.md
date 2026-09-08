Question 1: Observe the files created, what do you think they contain.
Running uv init generates the foundational configuration files for the Python project.
pyproject.toml: Stores project metadata, configuration, and Python dependencies.
.python-version: Specifies the exact Python version required for the environment.
README.md: Provides the template for documenting the project.
Source code folder (src/mlops_lab_1): Creates a boilerplate Python script to verify the environment works.

Question 2: What are the created files. What do you think they are used for? And which ones should be pushed to git?
Running dvc init creates the .dvc directory and the .dvcignore file:
.dvc/: Contains DVC configuration files (like config) and houses the local data cache where the actual file contents are stored.
.dvcignore: Instructs DVC on which files or directories to ignore, functioning exactly like a .gitignore file.
Git Tracking: we should push the .dvc tracking configuration files and .dvcignore to Git. we dont push the DVC cache or the large data files to Git.

Question 3: Where are the credentials stored? and what are the options other than --global? Should the credentials be pushed to github?
Using the --global flag stores the Dagshub credentials in the global DVC configuration on your host machine.
Alternative Options: --project, --local, and --system.
Git Tracking: Credentials must never be pushed to GitHub because they are private secrets. Only the non-secret remote configuration stored in .dvc/config should be committed and pushed to Git.

Question 4: Take a look at the .gitignore file. Explain what happened.
When We run dvc add data, DVC automatically make a new .gitignore file in the root folder by appending /data to it. This explicitly tells Git to ignore the actual large image files, ensuring they are not accidentally committed to the GitHub repository while DVC takes over versioning the dataset.

Question 5: Do you see a .dvc file? What does it contain?
Yes, running dvc add data generates a pointer file named data.dvc. This is a YAML file containing tracking metadata for the dataset, specifically:
The hash algorithm used (typically md5).
The unique hash value of the tracked directory.
The total size of the tracked data.
The number of files (nfiles) contained within the directory.
The relative path to the tracked data folder.

Question 6: The code is available on the GitHub main branch. The actual image data is not stored directly in GitHub. GitHub contains the data.dvc pointer file, which identifies the version and location of the DVC-tracked data. The image is available in the DagsHub web UI because dvc push uploaded the data to the configured DagsHub remote.

Question 7: After cloning the GitHub repository into a new temporary folder, the data folder is not present because Git only downloads the code and DVC pointer files. The command needed to download and restore the data folder is:

```bash
dvc pull
```