Question 1: Why is it a bad idea for a unit test to load the real model from your MLflow server? List at least two ways the test could fail even though your code is correct.

Loading the real model makes the test depend on services and artifacts outside the code being tested.
The test could fail because MLflow is stopped or unreachable, or because the champion alias or model artifacts are missing.
It could also fail because downloading the model times out or the machine does not have compatible model dependencies.
Using a fake model makes the unit test fast and repeatable without a GPU, DVC data, or a running MLflow server.

Question 2: The fake model always answers "Bread" with 0.90 confidence. What exactly does test_predict_returns_valid_json prove, and what does it not prove?

The test proves that the API accepts a multipart image upload, decodes and preprocesses it, calls the supplied model, and returns a successful JSON response with the expected category and a confidence between 0 and 1.
Our API expects logits, so the fake model returns the logarithms of the fixed probabilities. The API's softmax then produces 0.90 confidence for Bread.
The test does not prove that the real model classifies food accurately, that MLflow is reachable, or that the champion alias and its artifacts can be loaded.

Question 3: Which check caught it? Why is a train/test duplicate more dangerous than a corrupted image, even though the corrupted image makes the training script crash and the duplicate does not?

The cross-split content-hash check caught the copied image and returned exit code 1.
The existing mini dataset already contains 43 duplicate pairs across splits. Adding training/Bread/0_0.jpg as evaluation/Bread/lab5_duplicate_demo.jpg increased the count to 44. The deliberate copy was then removed; the original dataset was preserved.
The class folders, RGB/128x128 image format, minimum 50 images per class, and class balance checks passed.
A corrupted image causes a visible failure. A train/test duplicate can silently inflate test accuracy because the model has already seen the image during training, making the reported performance misleading.

Question 4: Your CI runner cannot see your data (it is in DVC, and maybe in a local remote). Where in your workflow should validate_data.py run so that it still protects you? Give two places.

Run it locally after preparing or pulling the processed data and before training.
Run it again on the training machine or a self-hosted runner with access to the real dataset, before training or accepting a new dataset version.
Cloud CI can test the validator using generated temporary datasets, but those tests do not validate the real DVC dataset.
On this project the real-data command is:

```powershell
uv run python src/food11/validate_data.py BigData/food11_processed_mini
```

Question 5: The smoke job has needs: test. What happens to it if a unit test fails? What would happen without needs?

If a unit test fails, the test job fails and GitHub skips smoke because its dependency did not succeed.
Without needs, GitHub can run smoke independently of test, so the image might build and start even while unit tests fail.

The first GitHub run demonstrated this: test failed during collection because the pytest command did not include the repository root on its import path, and smoke was skipped. Adding pythonpath = ["."] to the project's pytest settings fixes the src imports for the handout's uv run pytest -q command.

Question 7: Why tag with the commit SHA rather than latest? Describe a concrete situation where latest leads two machines to run different code.

The commit SHA identifies which code revision an image was built from and allows us to select that release explicitly. An image digest identifies the exact image bytes.
The latest tag can be reassigned. Machine A can pull latest on Monday and keep that image cached, while machine B pulls latest after Tuesday's update. Both say latest but run different code.
Our registry image name is ghcr.io/marioaouad/mlops-lab-1, using lowercase as required by the handout.

Question 8: Why does the push job declare permissions explicitly? What could a compromised step do with a token that had write access to everything in the repo?

The push job needs contents: read to check out code and packages: write to publish the image. Declaring these permissions gives the job only the rights it needs.
A compromised step with broad write access could alter repository code, publish malicious releases or packages, or change other resources permitted by its token.
