"""DebiasCLIP: Gradio demo for fair glaucoma screening."""
import logging
from pathlib import Path

import gradio as gr
import torch

from model import DebiasCLIP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("debiasclip")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATED_CHECKPOINT = PROJECT_ROOT / "debiasclip_adapter_v7.pth"
LEGACY_CHECKPOINT = PROJECT_ROOT / "debiasclip_adapter_v4.pth"
CHECKPOINT = (
    CALIBRATED_CHECKPOINT if CALIBRATED_CHECKPOINT.is_file() else LEGACY_CHECKPOINT
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_THRESHOLD = 0.5
BORDERLINE_MARGIN = 0.10       # flag predictions this close to the threshold
MAX_NOTE_WORDS = 60            # CLIP text encoder has a 77-token limit
DISCLAIMER = (
    "*Research prototype for academic demonstration. It is not a medical device "
    "and must not be used for clinical decisions. Uploaded data is not stored.*"
)


def load_model() -> tuple[DebiasCLIP, float]:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT.resolve()}")
    log.info("Loading calibrated DebiasCLIP adapter on %s ...", DEVICE)
    m = DebiasCLIP().to(DEVICE)
    try:
        state = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    except TypeError:  # older PyTorch without weights_only
        state = torch.load(CHECKPOINT, map_location=DEVICE)
    adapter_state = state.get("adapter_state_dict", state)
    threshold = float(state.get("decision_threshold", DEFAULT_THRESHOLD))
    m.adapter.load_state_dict(adapter_state)
    log.info("Model loaded.")
    return m.eval(), threshold


model, DEFAULT_THRESHOLD = load_model()


@torch.inference_mode()
def predict(image, note, threshold):
    if image is None:
        raise gr.Error("Please upload a retinal fundus image.")
    note = (note or "").strip()

    try:
        x = model.preprocess(image.convert("RGB")).unsqueeze(0).to(DEVICE)
        prob = torch.sigmoid(model(x, [note])).reshape(-1)[0].item()
    except Exception as e:  # surface a readable message in the UI
        log.exception("Inference failed")
        raise gr.Error(f"Inference failed: {e}")

    positive = prob >= threshold
    confidence = prob if positive else 1 - prob
    headline = (
        f"### ⚠️ Glaucoma suspected ({confidence:.1%} confidence)"
        if positive
        else f"### ✅ No glaucoma suspected ({confidence:.1%} confidence)"
    )

    warnings = []
    if not note:
        warnings.append(
            "No clinical note given. The model was trained on image + text, "
            "so treat this result with extra caution."
        )
    elif len(note.split()) > MAX_NOTE_WORDS:
        warnings.append("Long note: CLIP may truncate text beyond 77 tokens.")
    if abs(prob - threshold) <= BORDERLINE_MARGIN:
        warnings.append("Borderline score near the decision threshold. Refer for clinical review.")

    verdict = headline + "".join(f"\n\n> ⚠️ {w}" for w in warnings)
    return verdict, {"Glaucoma": prob, "Normal": 1 - prob}


if CHECKPOINT == CALIBRATED_CHECKPOINT:
    RESULTS = """
| Metric | Value |
| :--- | :--- |
| Accuracy | 76.20% |
| AUC | 0.8492 |
| Demographic Parity Difference (DPD) | 0.1437 |
| Equalized Odds Difference (EOD) | 0.0789 |
"""
    CALIBRATION_NOTE = (
        "The adapter threshold is selected on validation data, never on the test split. "
        "Additional fine-tuning did not improve validation accuracy, so the existing adapter "
        "weights were retained. Metrics are from the fixed cached test split."
    )
else:
    RESULTS = ""
    CALIBRATION_NOTE = (
        "Using the existing V4 adapter because no calibrated checkpoint was found. "
        "Run `python src/train.py` to train/calibrate the adapter."
    )

ABOUT = f"""
### About DebiasCLIP
A lightweight, fairness-aware glaucoma classifier built on a **frozen CLIP (ViT-B/32)**.
Image and text features (512 + 512) are fused into a 1024-d vector and passed through a
**527K-parameter adapter** (Linear, LayerNorm, ReLU, Dropout 0.3, Linear, residual connection, sigmoid).

{RESULTS}

**Calibration.** {CALIBRATION_NOTE}

**Key finding.** A leakage check showed race is more predictable from image features
(80.77% macro-AUC) than from text (67.31%), which motivated modality-specific debiasing.

**Running on:** `{DEVICE}`

{DISCLAIMER}
"""

with gr.Blocks(title="DebiasCLIP", theme=gr.themes.Soft(), analytics_enabled=False) as demo:
    gr.Markdown("# 👁️ DebiasCLIP: Fair Glaucoma Diagnosis\n"
                "A lightweight dual-modal AI for fair and accurate glaucoma screening.")
    with gr.Tab("Screening"):
        with gr.Row():
            with gr.Column():
                img = gr.Image(type="pil", label="Retinal fundus image", height=320)
                note = gr.Textbox(
                    label="Clinical note (recommended)", lines=4,
                    placeholder="e.g. Patient history, IOP, cup-to-disc findings ...",
                )
                with gr.Accordion("Advanced", open=False):
                    thr = gr.Slider(0.1, 0.9, value=DEFAULT_THRESHOLD, step=0.05,
                                    label="Decision threshold")
                with gr.Row():
                    clear = gr.ClearButton(value="Clear")
                    run = gr.Button("Run screening", variant="primary")
            with gr.Column():
                verdict = gr.Markdown("*Upload an image and click **Run screening**.*")
                probs = gr.Label(num_top_classes=2, label="Class probabilities")
        clear.add([img, note, verdict, probs])
        run.click(predict, inputs=[img, note, thr], outputs=[verdict, probs])
    with gr.Tab("About"):
        gr.Markdown(ABOUT)
    gr.Markdown(DISCLAIMER)

if __name__ == "__main__":
    demo.queue().launch()