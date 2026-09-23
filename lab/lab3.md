Question 1: Open the "Models" tab in the mlflow UI. What version number was your model given? What's the difference between a run's logged model artifact and a registered model?

The model version number is 1. A run’s logged model artifact is the model saved from one specific training run, so it belongs to that run and is usually referenced as something like runs:/<run-id>/model. A registered model in MLflow is a named, versioned model in the Model Registry, such as food11 version 1, which is managed separately from the run and can be tracked, aliased, and promoted over time.

Question 2: What aliases replaced the old built-in stages in mlflow? Why version a model separately from the run that produced it, and why is an alias more flexible than a fixed stage name?

The aliases that replaced the old built-in MLflow stages are custom names such as champion, challenger, or any other user-defined alias. The old Staging/Production stages were fixed labels tied to a model version, but aliases are more flexible because they are just mutable pointers that can be reassigned to a different version at any time without changing the version number itself. We version a model separately from the run that produced it because a training run is only one experiment execution, while a registered model is the official, versioned artifact that we want to track, compare, and deploy over time. This makes model management much easier because we can keep the same model name, assign aliases for promotion, and move the alias to a better-performing version whenever needed.

Question 3: Why load the model through an mlflow model URI (models:/food11@champion) instead of pointing directly at the .pth file on disk? What would you have to change to serve a newer model version?

Loading the model through models:/food11@champion allows MLflow to manage the model’s versions, metadata, artifacts, dependencies, and deployment alias centrally instead of relying on a specific .pth file on disk. To serve a newer version, register the new model version and move the champion alias to it, the serving code does not need to change, but the service must be restarted or the model reloaded. Alternatively, the URI could be changed explicitly to models:/food11/2 for version 2. The successful prediction request confirms that the API loaded and served the registered model.

Question 4: Why copy pyproject.toml/uv.lock and run uv sync before copying the rest of the source code, instead of copying everything at once? What happens to the build cache when you only change a line in serve.py?

Copying pyproject.toml and uv.lock first allows Docker to cache the dependency-installation layer. If only one line in serve.py changes, Docker reuses the cached uv sync layer and rebuilds only the layers after COPY src ./src, making the rebuild much faster. If all files were copied before uv sync, every source-code change could invalidate the dependency layer and reinstall all packages.

Question 5: What's the size difference between a naive single-stage image and your multi-stage one? Use docker history <image> to see which layers are the biggest.

The multi-stage image was 9.58 GB, while the naive single-stage image was 9.71 GB, making the multi-stage image approximately 136 MB smaller. Docker history showed that the largest layer in the multi-stage image was the copied virtual environment at 6.19 GB. In the single-stage image, the largest layer was the dependency installation layer at 6.24 GB. The single-stage image also retained the separate 61 MB layer used to install uv, whereas this builder-only layer was excluded from the final multi-stage image. The largest layers mainly contain PyTorch and NVIDIA CUDA dependencies, which dominate the size of both images.

Question 6: What happens to build speed and image size if you forget the .dockerignore? Which of the excluded folders would actually break the build if they were sent to the Docker daemon?

Without `.dockerignore`, Docker sends unnecessary files such as datasets, MLflow runs, virtual environments, Git history, and caches to the Docker daemon. This increases build-context transfer time and can make the image larger if those files are copied into an image layer. With this Dockerfile, none of the excluded folders directly breaks the build, because it copies only `pyproject.toml`, `uv.lock`, and `src`. A host `.venv` would become a problem only if the Dockerfile copied it: it can contain Windows-specific binaries and paths.

Question 7: Why can't the container simply use 127.0.0.1:5000 to reach the mlflow server on your host? What does host.docker.internal resolve to?

A container has its own isolated network namespace, so `127.0.0.1:5000` or `127.0.0.1:5001` refers to the container itself, not the host machine. On Windows and macOS, `host.docker.internal` resolves to the host machine, allowing the container to reach the MLflow server running on the host at port 5001.

Question 8: Stop the container and start a new one from the same image. Does the model still load correctly without you rebuilding? What does that tell you about what's baked into the image versus fetched at runtime?

After stopping and starting a new container from the same image, the model loaded correctly without rebuilding. This shows that the application code and Python dependencies are baked into the image, while the registered model is fetched from the MLflow tracking server at runtime.

Question 9: The Dockerfile and image are versioned differently — one lives in git, the other doesn't (yet). What's still missing before another machine (like a CI runner or a Kubernetes cluster) could reliably pull and run the exact image you just built?

Before another machine, CI runner, or Kubernetes cluster can reliably run the exact image, the locally built Docker image must be tagged and pushed to a container registry such as Docker Hub, GitHub Container Registry, or Azure Container Registry. The other machine must have permission to pull it and should use an immutable image digest rather than the changeable latest tag. It also needs the required runtime configuration, including the MLFLOW_TRACKING_URI, network access to the MLflow server, and access to the registered food11 model using the champion alias.