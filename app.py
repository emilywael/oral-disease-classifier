"""
Oral Disease Classification - Gradio App (Modern Dark UI)
يستخدم EfficientNetB0 المدرب لتصنيف صور أمراض الفم لـ 6 فئات
"""
import spaces
import gradio as gr
import numpy as np
import json
import os
from tensorflow.keras.models import load_model
from PIL import Image

# ==================== تحميل الموديل والفئات ====================
MODEL_PATH = "final_model.keras"
CLASSES_PATH = "class_names.json"
IMG_SIZE = (224, 224)

model = load_model(MODEL_PATH)

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

class_info = {
    "Calculus": {
        "desc": "Hardened plaque (tartar) built up on the teeth, usually near the gumline.",
        "rec": "Professional dental scaling is recommended to remove tartar buildup.",
    },
    "Data_Caries": {
        "desc": "Tooth decay caused by bacterial acid erosion of the enamel.",
        "rec": "Early treatment (filling) is recommended to prevent progression.",
    },
    "Gingivitis": {
        "desc": "Inflammation of the gums, typically caused by plaque accumulation.",
        "rec": "Improved oral hygiene and a dental cleaning are recommended.",
    },
    "Hypodontia": {
        "desc": "A congenital condition involving a missing number of teeth.",
        "rec": "Consultation with an orthodontist is recommended.",
    },
    "Mouth_Ulcer": {
        "desc": "A painful sore inside the mouth, usually healing within two weeks.",
        "rec": "See a doctor if the ulcer persists beyond 2 weeks or recurs frequently.",
    },
    "Tooth_Discoloration": {
        "desc": "A change in tooth color, which may be surface-level or internal.",
        "rec": "A dental evaluation is recommended to determine the cause.",
    },
}

MODEL_NAME = "EfficientNetB0"
TEST_ACCURACY = "80.07%"

# ==================== دالة التنبؤ ====================
@spaces.GPU
def predict_oral_disease(image):
    if image is None:
        return None, "### Upload an image to get started", "", ""

    img = Image.fromarray(image).convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)[0]
    results = {class_names[i]: float(predictions[i]) for i in range(len(class_names))}
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

    top_class = list(sorted_results.keys())[0]
    top_confidence = sorted_results[top_class]

    info = class_info.get(top_class, {"desc": "", "rec": ""})

    confidence_label = "High confidence" if top_confidence > 0.75 else \
                        "Moderate confidence" if top_confidence > 0.5 else \
                        "Low confidence — consider a clearer image"
    dot = "🟢" if top_confidence > 0.75 else "🟡" if top_confidence > 0.5 else "🔴"

    summary_md = f"""
### {top_class.replace('_', ' ')}
**Confidence: {top_confidence*100:.1f}%**

{dot} {confidence_label}
"""

    detail_md = f"""
**📖 Description**
{info['desc']}

**💊 Recommendation**
{info['rec']}
"""

    top3 = list(sorted_results.items())[:3]
    top3_md = "**🏆 Top 3 Predictions**\n\n" + " &nbsp;&nbsp;|&nbsp;&nbsp; ".join(
        [f"{name.replace('_',' ')}: **{prob*100:.1f}%**" for name, prob in top3]
    )

    return sorted_results, summary_md, detail_md, top3_md


# ==================== CSS مخصص ====================
custom_css = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}
#title_box {
    text-align: center;
    padding: 8px 0 4px 0;
}
#title_box h1 {
    font-size: 2rem;
    margin-bottom: 4px;
}
#title_box p {
    color: var(--body-text-color-subdued);
    font-size: 0.95rem;
}
.card {
    border-radius: 16px !important;
    border: 1px solid var(--border-color-primary) !important;
}
#predict-btn {
    font-weight: 600;
    font-size: 1.05rem;
    border-radius: 10px !important;
}
#disclaimer {
    text-align: center;
    font-size: 0.82rem;
    color: var(--body-text-color-subdued);
    padding-top: 6px;
}
"""

theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="cyan",
    neutral_hue="slate",
).set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_500",
)

# ==================== واجهة Gradio ====================
with gr.Blocks(title="Oral Disease Classifier", css=custom_css, theme=theme) as demo:

    with gr.Column(elem_id="title_box"):
        gr.Markdown("# 🦷 Oral Disease Classifier")
        gr.Markdown(
            f"Upload a clear photo of the oral cavity and get an instant AI-assisted "
            f"prediction across **6 disease categories**."
        )

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group(elem_classes="card"):
                image_input = gr.Image(label="📤 Upload Image", type="numpy", height=320)
                predict_btn = gr.Button("🔍 Analyze Image", variant="primary", elem_id="predict-btn")

        with gr.Column(scale=1):
            with gr.Group(elem_classes="card"):
                label_output = gr.Label(label="📊 Prediction Probabilities", num_top_classes=6)
                summary_output = gr.Markdown()

    with gr.Row():
        with gr.Group(elem_classes="card"):
            detail_output = gr.Markdown()
            top3_output = gr.Markdown()

    with gr.Accordion("📋 Supported Disease Categories", open=False):
        gr.Markdown(
            "- **Calculus** — hardened tartar buildup\n"
            "- **Data_Caries** — tooth decay\n"
            "- **Gingivitis** — gum inflammation\n"
            "- **Hypodontia** — congenitally missing teeth\n"
            "- **Mouth_Ulcer** — oral sores\n"
            "- **Tooth_Discoloration** — color changes in teeth"
        )

    with gr.Accordion("🧠 Model Details", open=False):
        gr.Markdown(
            f"- **Architecture:** {MODEL_NAME} (Transfer Learning)\n"
            f"- **Test Accuracy:** {TEST_ACCURACY}\n"
            f"- **Input size:** 224×224\n"
            f"- **Trained on:** 5,563 oral disease images across 6 classes"
        )

    predict_btn.click(
        fn=predict_oral_disease,
        inputs=image_input,
        outputs=[label_output, summary_output, detail_output, top3_output]
    )

    gr.Markdown(
        "⚠️ **Disclaimer:** This tool is for educational purposes only and is **not** "
        "a substitute for professional dental diagnosis. Always consult a licensed "
        "dentist or physician for medical concerns.",
        elem_id="disclaimer"
    )

if __name__ == "__main__":
    demo.launch()