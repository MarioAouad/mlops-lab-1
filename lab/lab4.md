Question 1: What happens to everything written to /mlflow-data if you never mount a volume there and just docker run this image standalone? Try it: run the container, register nothing, stop it, remove it, start a new one from the same image — what do you see in the UI?

Without a volume, /mlflow-data belongs to the container’s writable layer.
Stopping the container preserves it, but removing the container deletes it.
A new container starts with a fresh database and artifact directory.
Since nothing was registered, the UI looks empty both times.

Question 2: Why a named volume here instead of a bind mount to a folder in your repo (the way you might for local dev)? Would a bind mount work just as well?

A named volume keeps MLflow data independently of the containers.
Docker manages its location, avoiding dependence on a specific host folder.
A bind mount also provides persistence and would work with suitable permissions.
Named volumes are convenient here; bind mounts make files easier to inspect locally.

Question 3: In Lab 3 you had to use host.docker.internal or --network host to reach mlflow from inside the container. In this lab, MLFLOW_TRACKING_URI will simply be http://mlflow:5000. Why does that hostname resolve now when it didn't before?

Docker Compose creates a shared network where its internal DNS resolves service names.
Therefore, mlflow resolves to the MLflow container’s IP address, accessible on port 5000.
In Lab 3, MLflow ran on the host and had no mlflow service name on the container’s network.
We needed host.docker.internal or host networking to reach it instead.

Question 4: Why does the frontend read INFERENCE_URL from an environment variable instead of hardcoding http://inference:8000? Think about what happens if you ever docker run this frontend image on its own, outside Compose.

An environment variable lets the same frontend image connect to different inference addresses.
Inside Compose, http://inference:8000 works because Docker resolves the service name.
Outside that network, the hostname inference may not resolve.
You can supply another address, such as http://host.docker.internal:8000 for an API running on your host, without rebuilding the image.

Question 5: Only mlflow and frontend publish a port to the host. inference doesn't. Why not, and how does the frontend still reach it?

Only MLflow and the frontend need direct access from your computer.
Inference handles internal requests, so it does not need a published port.
The frontend reaches it at http://inference:8000 through Compose’s shared network.
Docker’s internal DNS resolves inference to the correct container.

Question 6: depends_on here only waits for the mlflow container process to start, not for the tracking server inside it to be ready to accept connections. If your serve.py tries to load the Staging model at startup and mlflow isn't ready yet, what happens to the inference container? Look at docker compose logs inference if it fails.

With basic depends_on, inference can start before MLflow accepts connections.
If model loading fails without retry handling, API startup fails and the container exits.
docker compose logs inference would show the connection or model-loading error.
Our configuration waits for MLflow’s health check and retries temporary loading failures; it uses champion instead of Staging.

Question 7: Run docker compose ps. Which services have a published port listed, and which don't? Does that match what you'd expect from the docker-compose.yml?

MLflow publishes host port 5001 to container port 5000.
The frontend publishes host port 8501 to container port 8501.
Inference shows 8000/tcp, but no host mapping, so its port is not published.
This matches our docker-compose.yml.

Question 8: Refresh the frontend and upload an image again. Does the prediction come from the new model version, or the old one? Your serve.py loads the model once, at startup — what single command lets you pick up the new Staging version without rebuilding any image?

Refreshing the frontend still uses the old model loaded in inference’s memory.
Changing the registry alias does not automatically reload that running model.
Run docker compose restart inference to load the version now assigned to champion.
No image rebuild is required.

Question 9: Why does restart alone work here — no rebuild needed? What does that tell you about what's baked into the inference image versus fetched at container startup?

The inference image contains the application code and dependencies.
The trained model is fetched from MLflow when the container starts.
Restarting runs startup again and resolves the alias to its current model version.
A rebuild is only needed when the image’s code or dependencies change.

Question 10: Is your registered model and its Staging assignment still there after this down/up cycle? Now try docker compose down -v followed by docker compose up — what's different this time, and why?

Yes, the registered versions and champion alias survive a normal down/up cycle.
Docker preserves the named volume containing MLflow’s database and artifacts.
Using docker compose down -v deletes that volume, so the next startup creates an empty registry.
Inference then cannot load champion until a model is registered and assigned that alias again.

Question 11: This compose file is still meant to run on one machine. What would have to change for the inference service to run as three replicas behind a load balancer, or for the mlflow service to survive a machine failure? (You don't need to implement this — just name what Docker Compose can't give you here.)

Three inference replicas need a load balancer to distribute requests between them.
Compose can run replicas on one machine, but cannot move them to another if that machine fails.
Multi-machine scheduling and automatic recovery require an orchestrator such as Kubernetes or Docker Swarm.
MLflow also needs a resilient shared database and artifact storage, rather than SQLite and a volume tied to one machine.