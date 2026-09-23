# MLOps Lab Commands

Run these commands from the repository root:

```powershell
cd "C:\Users\MA21\Desktop\USJ\Final_Year\1th_semeter\ML_Ops\Lab\mlops-lab-1"
```

## 1. Start MLflow (every session)

Start MLflow first. The `--host 0.0.0.0` option allows Docker containers to reach the server. MLflow 3.16 reads the host allowlist from `MLFLOW_SERVER_ALLOWED_HOSTS`.

```powershell
$env:MLFLOW_SERVER_ALLOWED_HOSTS="*"
uv run python -m mlflow server `
  --host 0.0.0.0 `
  --port 5001 `
  --workers 1 `
  --backend-store-uri sqlite:///mlflow.db `
  --serve-artifacts `
  --artifacts-destination ./mlruns
```

Keep this terminal open. The `--serve-artifacts` option is required so a Docker
container can download the registered model through MLflow instead of trying to
read a Windows-only local file path. Do not register or train the model again
when starting another session.

MLflow UI:

```text
http://127.0.0.1:5001
```

Stop MLflow when you are finished by pressing `Ctrl+C` in this terminal. Stop the API container before stopping MLflow.

## 2. Start the API locally (optional)

Use a second terminal:

```powershell
uv run python -m uvicorn src.food11.serve:app --host 0.0.0.0 --port 8000
```

Local API URL:

```text
http://127.0.0.1:8000
```

## 3. Test the local API

Health check:

```powershell
curl.exe http://127.0.0.1:8000/health
```

Prediction check:

```powershell
curl.exe -X POST `
  -F "file=@BigData/food11_processed_mini/validation/Bread/0_0.jpg" `
  http://127.0.0.1:8000/predict
```

## 4. Build the multi-stage Docker image (only after source/dependency changes)

```powershell
docker build --progress=plain -t food11-api:multi .
docker tag food11-api:multi food11-api:latest
```

Check the image:

```powershell
docker images food11-api:multi food11-api:latest
```

## 5. Run the Docker container (every session when using Docker)

Stop and remove an older container with the same name if necessary:

```powershell
docker rm -f food11-api-container 2>$null
```

Start the container. MLflow remains on the host, so the container uses `host.docker.internal:5001`:

```powershell
docker run --rm --name food11-api-container `
  -p 8000:8000 `
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5001 `
  food11-api:latest
```

For a background container:

```powershell
docker run -d --name food11-api-container `
  -p 8000:8000 `
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5001 `
  food11-api:latest
```

## 6. Test the Docker API

```powershell
curl.exe http://127.0.0.1:8000/health
```

```powershell
curl.exe -X POST `
  -F "file=@BigData/food11_processed_mini/validation/Bread/0_0.jpg" `
  http://127.0.0.1:8000/predict
```

View container logs:

```powershell
docker logs food11-api-container
```

Check running containers:

```powershell
docker ps
```

Stop the background container:

```powershell
docker stop food11-api-container
```

## 7. Question 8: restart without rebuilding

```powershell
docker rm -f food11-api-container 2>$null
docker run -d --name food11-api-container `
  -p 8000:8000 `
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5001 `
  food11-api:latest
```

The model is downloaded from MLflow when the container starts, so rebuilding the image is not required for a container restart.

## 8. Quick daily Docker workflow

From the repository root, use two terminals. In Terminal 1, start MLflow with
the command in Section 1. In Terminal 2, start the existing image:

```powershell
docker rm -f food11-api-container 2>$null
docker run -d --name food11-api-container `
  -p 8000:8000 `
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5001 `
  food11-api:latest
```

Then verify the service:

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe -X POST `
  -F "file=@BigData/food11_processed_mini/validation/Bread/0_0.jpg" `
  http://127.0.0.1:8000/predict
```

The model registration and alias assignment are one-time setup tasks. Repeat
them only when you intentionally register a new model version.

## 9. Inspect image size and layers

```powershell
docker images food11-api:multi food11-api:single
```

```powershell
docker history --format "table {{.Size}}\t{{.CreatedBy}}" food11-api:multi
```

```powershell
docker history --format "table {{.Size}}\t{{.CreatedBy}}" food11-api:single
```

## 10. Build the single-stage comparison image

This is only needed for Question 5:

```powershell
docker build -f Dockerfile.single-stage -t food11-api:single .
```

## 11. Stop the lab services and clean up

Remove the container:

```powershell
docker rm -f food11-api-container
```

If MLflow is running in its own terminal, press `Ctrl+C` there. If the
terminal is no longer available, stop only the MLflow processes listening on
port `5001`:

```powershell
Get-NetTCPConnection -State Listen -LocalPort 5001 -ErrorAction SilentlyContinue |
  Select-Object -ExpandProperty OwningProcess -Unique |
  ForEach-Object { Stop-Process -Id $_ -Force }
```

Verify that the lab services are stopped:

```powershell
docker ps -a --filter name=food11-api-container
Get-NetTCPConnection -State Listen -LocalPort 5001,8000 -ErrorAction SilentlyContinue
```

This stops the lab services on ports `5001` and `8000` without affecting
unrelated applications. Do not stop Docker Desktop itself.

Remove only the single-stage comparison image after recording the answer to Question 5 (it uses roughly 9.5 GB of unique disk space):

```powershell
docker rmi food11-api:single
```

`food11-api:latest` and `food11-api:multi` are two tags for the same image, so keeping both does not consume extra image space. Remove both tags only when you no longer need the image for Question 8:

```powershell
docker rmi food11-api:multi food11-api:latest
```

## Important notes

- Use `uv run python -m uvicorn`, not `uv run uvicorn`, because the direct Uvicorn wrapper can fail on Windows with a trampoline error.
- PowerShell uses `curl` as an alias for `Invoke-WebRequest`; use `curl.exe` for upload commands.
- The local dataset path is `BigData/food11_processed_mini`, not `data/food11_processed_mini`.
- `127.0.0.1` inside a container refers to the container itself. Use `host.docker.internal` to reach MLflow on the Windows host.
- MLflow must be running before the API or Docker container starts because the model is loaded from `models:/food11@champion` at application startup.
- If port `8000` is occupied, find the process with `netstat -ano | findstr :8000` or use another host port such as `8001:8000`.
