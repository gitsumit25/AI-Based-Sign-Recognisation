"""SIGNOVA FastAPI backend using the original 3-class LSTM checkpoint."""
from __future__ import annotations

import base64
import binascii
import os
import tempfile
import threading
import logging
import uuid
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential

PROJECT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_DIR / "frontend"
MODEL_PATH = Path(os.environ.get("SIGNOVA_MODEL_PATH", str(Path(__file__).resolve().parent / "models" / "170-0.83.hdf5")))
PERSONALIZED_MODEL_PATH = MODEL_PATH.parent / "signova_personalized.h5"
TRAINING_DATA_DIR = Path(os.environ.get("SIGNOVA_TRAINING_DATA_DIR", str(Path(__file__).resolve().parent / "training_data")))
ACTIONS = ["Hello", "How are you", "thank you"]  # Original checkpoint class order
ACTION_LOOKUP = {" ".join(label.lower().split()): idx for idx, label in enumerate(ACTIONS)}
logger = logging.getLogger("uvicorn.error")
SEQUENCE_LENGTH = 45
FEATURE_COUNT = 258
MAX_VIDEO_BYTES = 50 * 1024 * 1024
PREDICT_EVERY_N_FRAMES = 6

app = FastAPI(title="SIGNOVA API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8001", "http://localhost:8001", "http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
model_error: Optional[str] = None
model_lock = threading.Lock()
mp_holistic = mp.solutions.holistic
sessions: Dict[str, dict] = {}
sessions_lock = threading.RLock()
mediapipe_lock = threading.Lock()


def build_model(weights_path: Optional[Path] = None):
    """Re-create the checkpoint architecture, loading personal weights if available."""
    network = Sequential()
    network.add(LSTM(64, return_sequences=True, activation="relu", input_shape=(SEQUENCE_LENGTH, FEATURE_COUNT)))
    network.add(LSTM(128, return_sequences=True, activation="relu"))
    network.add(LSTM(256, return_sequences=True, activation="relu"))
    network.add(LSTM(64, return_sequences=False, activation="relu"))
    network.add(Dense(64, activation="relu"))
    network.add(Dense(32, activation="relu"))
    network.add(Dense(len(ACTIONS), activation="softmax"))
    if weights_path is None:
        weights_path = PERSONALIZED_MODEL_PATH if PERSONALIZED_MODEL_PATH.is_file() else MODEL_PATH
    network.load_weights(str(weights_path))
    network.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
    return network


def get_model():
    global model, model_error
    if model is not None:
        return model
    with model_lock:
        if model is not None:
            return model
        if not MODEL_PATH.is_file():
            model_error = f"Model weights not found at {MODEL_PATH}"
            raise HTTPException(status_code=503, detail=model_error)
        try:
            model = build_model()
            model_error = None
            logger.info("Loaded SIGNOVA weights from %s", MODEL_PATH)
        except Exception as exc:
            model_error = str(exc)
            logger.exception("Could not load SIGNOVA model weights")
            raise HTTPException(status_code=503, detail=f"Could not load model weights: {exc}") from exc
    return model


def new_holistic():
    return mp_holistic.Holistic(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def extract_keypoints_from_rgb(frame_rgb: np.ndarray, holistic) -> tuple[np.ndarray, bool]:
    """Extract pose+left-hand+right-hand landmarks in the original checkpoint format.

    The original source used skvideo frames and then cv2.COLOR_BGR2RGB in its helper.
    We intentionally retain that channel-conversion convention for checkpoint compatibility.
    """
    image = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    pose_landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None
    left_landmarks = results.left_hand_landmarks.landmark if results.left_hand_landmarks else None
    right_landmarks = results.right_hand_landmarks.landmark if results.right_hand_landmarks else None

    pose = np.asarray([[p.x, p.y, p.z, p.visibility] for p in pose_landmarks], dtype=np.float32).flatten() if pose_landmarks else np.zeros(33 * 4, dtype=np.float32)
    left = np.asarray([[p.x, p.y, p.z] for p in left_landmarks], dtype=np.float32).flatten() if left_landmarks else np.zeros(21 * 3, dtype=np.float32)
    right = np.asarray([[p.x, p.y, p.z] for p in right_landmarks], dtype=np.float32).flatten() if right_landmarks else np.zeros(21 * 3, dtype=np.float32)
    features = np.concatenate([pose, left, right]).astype(np.float32)
    detected = bool(pose_landmarks or left_landmarks or right_landmarks)
    if features.shape != (FEATURE_COUNT,):
        raise ValueError(f"Expected {FEATURE_COUNT} features, got {features.shape}")
    return features, detected


def decode_data_url(data_url: str) -> np.ndarray:
    try:
        encoded = data_url.split(",", 1)[1] if "," in data_url else data_url
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(status_code=400, detail="Camera frame was not valid base64 image data.") from exc
    image_array = np.frombuffer(raw, dtype=np.uint8)
    frame_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode camera frame.")
    # Convert to RGB first because the original extractor receives RGB frames from skvideo.
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def predict_scores(sequence: np.ndarray) -> np.ndarray:
    if sequence.shape != (SEQUENCE_LENGTH, FEATURE_COUNT):
        raise ValueError(f"Expected sequence shape {(SEQUENCE_LENGTH, FEATURE_COUNT)}, got {sequence.shape}")
    network = get_model()
    with model_lock:
        scores = network.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if scores.shape != (len(ACTIONS),):
        raise ValueError(f"Expected {len(ACTIONS)} class scores, got {scores.shape}")
    return scores


def prediction_payload(scores: np.ndarray) -> dict:
    index = int(np.argmax(scores))
    label = ACTIONS[index]
    display = label.title() if label.lower() != "thank you" else "Thank you"
    return {
        "ready": True,
        "model_ready": True,
        "class_index": index,
        "sign": label,
        "text": display,
        "confidence": round(float(scores[index]) * 100.0, 2),
        "scores": {name: round(float(value) * 100.0, 2) for name, value in zip(ACTIONS, scores)},
    }


def extract_video_sequence(video_path: str) -> tuple[np.ndarray, int]:
    """Extract exactly 45 landmark frames using the inference preprocessing pipeline."""
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError("Could not open this video. Try an MP4 or MOV file.")
    holistic = None
    try:
        frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        selected = []
        if frame_total > 0:
            indexes = np.rint(np.linspace(0, frame_total - 1, min(SEQUENCE_LENGTH, frame_total))).astype(int)
            for frame_index in indexes:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = capture.read()
                if ok:
                    selected.append(frame)
        else:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                selected.append(frame)
                if len(selected) >= 10000:
                    break
            if len(selected) > SEQUENCE_LENGTH:
                indexes = np.rint(np.linspace(0, len(selected) - 1, SEQUENCE_LENGTH)).astype(int)
                selected = [selected[int(i)] for i in indexes]
        if not selected:
            raise ValueError("No readable frames were found in the video.")

        holistic = new_holistic()
        features_list = []
        visible_count = 0
        for frame_bgr in selected[:SEQUENCE_LENGTH]:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            with mediapipe_lock:
                features, visible = extract_keypoints_from_rgb(frame_rgb, holistic)
            features_list.append(features)
            visible_count += int(visible)
        while len(features_list) < SEQUENCE_LENGTH:
            features_list.append(np.zeros(FEATURE_COUNT, dtype=np.float32))
        return np.asarray(features_list, dtype=np.float32), visible_count
    finally:
        capture.release()
        if holistic is not None:
            try:
                holistic.close()
            except Exception:
                pass


def _sample_paths(class_index: int) -> list[Path]:
    folder = TRAINING_DATA_DIR / str(class_index)
    return sorted(folder.glob("*.npy")) if folder.exists() else []


def _dataset_counts() -> dict:
    counts = {label: len(_sample_paths(index)) for index, label in enumerate(ACTIONS)}
    return {"counts": counts, "total_samples": sum(counts.values()), "minimum_per_class": 15, "recommended_per_class": 30}


def _make_session() -> dict:
    return {
        "holistic": new_holistic(),
        "frames": deque(maxlen=SEQUENCE_LENGTH),
        "visible": deque(maxlen=SEQUENCE_LENGTH),
        "frames_since_prediction": 0,
        "score_history": deque(maxlen=3),
        "last_scores": None,
    }


def _reset_session(session_id: str) -> None:
    with sessions_lock:
        state = sessions.pop(session_id, None)
    if state:
        try:
            state["holistic"].close()
        except Exception:
            pass


class FrameRequest(BaseModel):
    session_id: str
    image: str


class ResetRequest(BaseModel):
    session_id: str


@app.get("/api/health")
def health():
    try:
        get_model()
        ready = True
    except HTTPException:
        ready = False
    return {
        "service": "SIGNOVA API",
        "status": "ok",
        "model_ready": ready,
        "model_file_exists": MODEL_PATH.is_file(),
        "model_path": str(MODEL_PATH),
        "classes": ACTIONS,
        "sequence_shape": [SEQUENCE_LENGTH, FEATURE_COUNT],
        "message": "Model loaded and ready." if ready else (model_error or "Model is not ready."),
    }


@app.post("/api/recognize-frame")
def recognize_frame(payload: FrameRequest):
    session_id = payload.session_id.strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    frame_rgb = decode_data_url(payload.image)

    with sessions_lock:
        state = sessions.get(session_id)
        if state is None:
            state = _make_session()
            sessions[session_id] = state

    try:
        # MediaPipe's tracker should not be called concurrently.
        with mediapipe_lock:
            features, detected = extract_keypoints_from_rgb(frame_rgb, state["holistic"])
    except Exception as exc:
        logger.exception("Landmark extraction failed")
        raise HTTPException(status_code=500, detail=f"Could not extract landmarks: {exc}") from exc

    state["frames"].append(features)
    state["visible"].append(detected)
    state["frames_since_prediction"] += 1
    frames_collected = len(state["frames"])
    valid_frames = int(sum(state["visible"]))

    if frames_collected < SEQUENCE_LENGTH:
        return {
            "ready": False,
            "frames_collected": frames_collected,
            "required_frames": SEQUENCE_LENGTH,
            "visible_frames": valid_frames,
            "message": "Keep your upper body and both hands visible while performing one complete sign.",
        }

    if valid_frames < 10:
        return {
            "ready": False,
            "frames_collected": frames_collected,
            "required_frames": SEQUENCE_LENGTH,
            "visible_frames": valid_frames,
            "message": "No clear body or hand landmarks. Improve lighting and move both hands into view.",
        }

    if state["last_scores"] is None or state["frames_since_prediction"] >= PREDICT_EVERY_N_FRAMES:
        sequence = np.asarray(state["frames"], dtype=np.float32)
        try:
            scores = predict_scores(sequence)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Live prediction failed")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
        state["score_history"].append(scores)
        # Average a few adjacent windows to avoid flicker; this doesn't change model weights.
        smoothed = np.mean(np.stack(list(state["score_history"]), axis=0), axis=0)
        state["last_scores"] = smoothed
        state["frames_since_prediction"] = 0

    result = prediction_payload(state["last_scores"])
    result.update({"frames_collected": frames_collected, "required_frames": SEQUENCE_LENGTH, "visible_frames": valid_frames})
    return result


@app.post("/api/recognize-reset")
def recognize_reset(payload: ResetRequest):
    _reset_session(payload.session_id.strip())
    return {"ok": True, "message": "Recognition sequence reset."}


@app.post("/api/predict")
@app.post("/upload-video/")
async def predict_video(file: UploadFile = File(...)):
    """Predict one sign from an uploaded video using original 45-frame preprocessing."""
    suffix = Path(file.filename or "upload.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".m4v", ".webm"}:
        raise HTTPException(status_code=400, detail="Upload an MP4, MOV, AVI, M4V, or WebM video.")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded video is empty.")
    if len(content) > MAX_VIDEO_BYTES:
        raise HTTPException(status_code=413, detail="Video is too large. Maximum size is 50 MB.")

    temp_path = None
    holistic = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(content)
            temp_path = temp.name

        capture = cv2.VideoCapture(temp_path)
        if not capture.isOpened():
            raise HTTPException(status_code=400, detail="Could not open the uploaded video. Try an MP4 file.")
        frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_total <= 0:
            raw_frames = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                raw_frames.append(frame)
            capture.release()
            if not raw_frames:
                raise HTTPException(status_code=400, detail="The uploaded file contains no readable video frames.")
            indexes = np.rint(np.linspace(0, len(raw_frames) - 1, min(SEQUENCE_LENGTH, len(raw_frames)))).astype(int)
            selected = [raw_frames[int(i)] for i in indexes]
        else:
            indexes = np.rint(np.linspace(0, frame_total - 1, min(SEQUENCE_LENGTH, frame_total))).astype(int)
            selected = []
            for frame_index in indexes:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = capture.read()
                if ok:
                    selected.append(frame)
            capture.release()
            if not selected:
                raise HTTPException(status_code=400, detail="Could not read frames from this video.")

        holistic = new_holistic()
        features_list = []
        visible_list = []
        for frame_bgr in selected:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            with mediapipe_lock:
                features, visible = extract_keypoints_from_rgb(frame_rgb, holistic)
            features_list.append(features)
            visible_list.append(visible)

        while len(features_list) < SEQUENCE_LENGTH:
            features_list.append(np.zeros(FEATURE_COUNT, dtype=np.float32))
            visible_list.append(False)
        sequence = np.asarray(features_list[:SEQUENCE_LENGTH], dtype=np.float32)
        scores = predict_scores(sequence)
        result = prediction_payload(scores)
        result.update({
            "frames_used": SEQUENCE_LENGTH,
            "video_frames": frame_total or len(selected),
            "visible_frames": int(sum(visible_list[:SEQUENCE_LENGTH])),
        })
        return result
    finally:
        if holistic is not None:
            try:
                holistic.close()
            except Exception:
                pass
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        await file.close()



@app.get("/trainer")
def trainer_page():
    return FileResponse(FRONTEND_DIR / "trainer.html")


@app.get("/api/dataset/counts")
def dataset_counts():
    return _dataset_counts()


@app.post("/api/dataset/add-video")
async def add_training_video(label: str = Form(...), file: UploadFile = File(...)):
    key = " ".join(label.lower().split())
    if key not in ACTION_LOOKUP:
        raise HTTPException(status_code=400, detail=f"Choose one of these labels: {', '.join(ACTIONS)}")
    suffix = Path(file.filename or "sample.webm").suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".m4v", ".webm"}:
        raise HTTPException(status_code=400, detail="Use an MP4, MOV, AVI, M4V, or WebM clip.")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The recorded video is empty.")
    if len(content) > MAX_VIDEO_BYTES:
        raise HTTPException(status_code=413, detail="Video is too large. Keep each sample under 50 MB.")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(content)
            temp_path = temp.name
        sequence, visible = extract_video_sequence(temp_path)
        if visible < 10:
            raise HTTPException(
                status_code=422,
                detail="SIGNOVA could not see enough landmarks. Keep your upper body and both hands visible, improve lighting, and record again.",
            )
        class_index = ACTION_LOOKUP[key]
        class_dir = TRAINING_DATA_DIR / str(class_index)
        class_dir.mkdir(parents=True, exist_ok=True)
        sample_path = class_dir / f"sample_{uuid.uuid4().hex}.npy"
        np.save(sample_path, sequence)
        result = _dataset_counts()
        return {
            "ok": True,
            "label": ACTIONS[class_index],
            "visible_frames": visible,
            "saved_count_for_label": result["counts"][ACTIONS[class_index]],
            **result,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Could not save training sample")
        raise HTTPException(status_code=500, detail=f"Could not process this training clip: {exc}") from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        await file.close()


@app.post("/api/dataset/clear")
def clear_training_dataset():
    import shutil
    if TRAINING_DATA_DIR.exists():
        shutil.rmtree(TRAINING_DATA_DIR, ignore_errors=True)
    return _dataset_counts()


@app.post("/api/train-personalized")
def train_personalized_model():
    """Fine-tune the last LSTM/Dense layers on user-labelled examples."""
    global model, model_error
    counts = _dataset_counts()["counts"]
    too_few = {label: count for label, count in counts.items() if count < 15}
    if too_few:
        details = ", ".join(f"{label}: {count}" for label, count in too_few.items())
        raise HTTPException(
            status_code=400,
            detail=f"Record at least 15 examples for every sign before training. Current counts — {details}. We recommend 30–50 examples per sign.",
        )

    rng = np.random.default_rng(2026)
    train_sequences, train_labels, val_sequences, val_labels = [], [], [], []
    for class_index, label in enumerate(ACTIONS):
        samples = []
        for path in _sample_paths(class_index):
            try:
                sample = np.load(path).astype(np.float32)
                if sample.shape == (SEQUENCE_LENGTH, FEATURE_COUNT) and np.isfinite(sample).all():
                    samples.append(sample)
            except Exception:
                logger.warning("Skipping unreadable sample %s", path)
        if len(samples) < 15:
            raise HTTPException(status_code=400, detail=f"Only {len(samples)} usable samples found for {label}; at least 15 are required.")
        order = rng.permutation(len(samples))
        val_count = max(2, int(round(len(samples) * 0.2)))
        val_ids = set(order[:val_count].tolist())
        for sample_id, sample in enumerate(samples):
            if sample_id in val_ids:
                val_sequences.append(sample)
                val_labels.append(class_index)
            else:
                train_sequences.append(sample)
                train_labels.append(class_index)

    train_order = rng.permutation(len(train_sequences))
    X_train = np.asarray(train_sequences, dtype=np.float32)[train_order]
    y_train = np.asarray(train_labels, dtype=np.int32)[train_order]
    X_val = np.asarray(val_sequences, dtype=np.float32)
    y_val = np.asarray(val_labels, dtype=np.int32)
    Y_train = to_categorical(y_train, num_classes=len(ACTIONS))
    Y_val = to_categorical(y_val, num_classes=len(ACTIONS))

    network = get_model()
    with model_lock:
        old_weights = [weight.copy() for weight in network.get_weights()]
        old_trainable = [layer.trainable for layer in network.layers]
        try:
            # Keep general motion features from the original model; adapt its final layers to your examples.
            for layer_index, layer in enumerate(network.layers):
                layer.trainable = layer_index >= 3
            network.compile(
                optimizer=__import__("tensorflow").keras.optimizers.Adam(learning_rate=5e-5),
                loss="categorical_crossentropy",
                metrics=["categorical_accuracy"],
            )
            callbacks = [
                EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=0),
            ]
            network.fit(
                X_train, Y_train,
                validation_data=(X_val, Y_val),
                epochs=35,
                batch_size=min(8, max(2, len(X_train) // 6)),
                callbacks=callbacks,
                shuffle=True,
                verbose=0,
            )
            eval_values = network.evaluate(X_val, Y_val, verbose=0)
            predictions = np.argmax(network.predict(X_val, verbose=0), axis=1)
            per_class = {}
            for class_index, label in enumerate(ACTIONS):
                selected = (y_val == class_index)
                per_class[label] = round(float(np.mean(predictions[selected] == y_val[selected]) * 100.0), 2) if np.any(selected) else None
            temp_weights = PERSONALIZED_MODEL_PATH.with_name("signova_personalized.tmp.h5")
            network.save_weights(str(temp_weights))
            os.replace(temp_weights, PERSONALIZED_MODEL_PATH)
            model = network
            model_error = None
            return {
                "ok": True,
                "message": "Personalized training finished. Test it with new videos that were not used for training.",
                "training_samples": len(X_train),
                "validation_samples": len(X_val),
                "validation_accuracy": round(float(eval_values[1] * 100.0), 2),
                "per_class_validation_accuracy": per_class,
                "personalized_model_saved": True,
                "counts": counts,
                "note": "Validation scores are estimates from your recorded clips, not a guarantee of real-world accuracy. More varied labelled examples usually help.",
            }
        except HTTPException:
            network.set_weights(old_weights)
            for layer, trainable in zip(network.layers, old_trainable):
                layer.trainable = trainable
            network.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
            raise
        except Exception as exc:
            network.set_weights(old_weights)
            for layer, trainable in zip(network.layers, old_trainable):
                layer.trainable = trainable
            network.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
            logger.exception("Personalized model training failed")
            raise HTTPException(status_code=500, detail=f"Training failed: {exc}") from exc


@app.post("/api/reset-model")
def reset_to_original_model():
    global model, model_error
    with model_lock:
        try:
            if PERSONALIZED_MODEL_PATH.exists():
                PERSONALIZED_MODEL_PATH.unlink()
            model = build_model(weights_path=MODEL_PATH)
            model_error = None
        except Exception as exc:
            model_error = str(exc)
            logger.exception("Could not restore original checkpoint")
            raise HTTPException(status_code=500, detail=f"Could not restore original model: {exc}") from exc
    return {"ok": True, "message": "The original pretrained checkpoint has been restored."}


@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/{asset_path:path}")
def frontend_assets(asset_path: str):
    candidate = (FRONTEND_DIR / asset_path).resolve()
    if FRONTEND_DIR.resolve() not in candidate.parents or not candidate.is_file():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(candidate)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8001, reload=False)
