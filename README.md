# SIGNOVA — Personal Accuracy Trainer

This package includes the original three-class LSTM checkpoint plus an **Accuracy Studio** page that lets you collect labelled examples through the browser and fine-tune the model without hand-editing Python.

## What it recognizes

The current checkpoint recognizes only these three isolated classes: **Hello**, **How are you**, and **thank you**. It is not a general sign-language translator. The supplied checkpoint misclassified at least one test clip, so this package adds personalization training rather than pretending the original model is already accurate.

## Run on Windows

1. Extract this ZIP into a normal folder.
2. Double-click `RUN_SIGNOVA.bat`.
3. The first run creates a Python 3.10 environment and installs dependencies; allow several minutes and keep the terminal open.
4. Open the recognition website at `http://127.0.0.1:8001`.
5. Open **Improve Accuracy** in the top navigation, or visit `http://127.0.0.1:8001/trainer`.

## Improve the model

1. In Accuracy Studio, click **Start camera** and allow camera access.
2. Choose **Hello**, perform that exact sign once during the 3-second recording, and click **Record 3-second sample**. Repeat for the same sign in different takes.
3. Repeat for **How are you** and **thank you**. The page shows counts for each label. You need at least **15 samples per sign** before training; **30–50 per sign** is recommended. You can also upload existing short clips using the upload button.
4. Click **Train SIGNOVA on my samples** and wait for completion. The app fine-tunes the last LSTM/Dense layers from the original checkpoint and saves `backend/models/signova_personalized.h5`.
5. Return to the main page and test with **new clips that were not used for training**.

For useful samples, keep both hands, your upper body, and the signing area in frame. Avoid blurry/very dark clips. Vary camera distance, background, and lighting moderately. Do not mix different sign variants under the same label. Training with a few clips can overfit; more varied examples generally help.

The validation percentage shown by the trainer is only an estimate from the collected clips; it does **not** guarantee the same accuracy for new people or conditions. If results are still poor, record more clean examples. To undo personalization, click **Restore original model** on Accuracy Studio.

Collected landmark samples are saved locally under `backend/training_data`. Clear them from Accuracy Studio if needed. Browser recordings and labels are handled by your local backend, not uploaded to a cloud service by this code.

## Technical details

- Input: 45 time steps × 258 MediaPipe pose/hand landmark values.
- Original checkpoint: `backend/models/170-0.83.hdf5` (weights-only file; code rebuilds the architecture and loads the weights).
- Personalized checkpoint: `backend/models/signova_personalized.h5`.
- Training data: `backend/training_data/<class_index>/*.npy`.
- API docs: `http://127.0.0.1:8001/docs`.
- Health check: `http://127.0.0.1:8001/api/health`.

## If startup fails

The launcher prints any package installation error. Python 3.10 is recommended. The requirements use MediaPipe 0.10.10 because MediaPipe 0.10.2 is unavailable from pip for this setup. If pip fails, copy the last 10 lines from the backend terminal and troubleshoot that error before retrying.
