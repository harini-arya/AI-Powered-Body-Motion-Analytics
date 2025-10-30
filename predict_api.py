# predict_api.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import os, shutil, base64
from datetime import datetime
from scripts.convert_bvh_to_csv import bvh_to_csv
from api.utils import load_artifacts, run_inference, generate_analytics
from openai import OpenAI
import tensorflow as tf
from dotenv import load_dotenv
import os

load_dotenv() 

tf.config.set_visible_devices([], 'GPU')

# Directories
ARTIFACT_DIR = "artifacts"
OUTPUT_DIR = "outputs"

# Initialize FastAPI
app = FastAPI(title="Posture Analytics API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to restrict to specific domains if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load model and preprocessing artifacts
model, scaler, le, rot_cols = load_artifacts(ARTIFACT_DIR)


def generate_llm_feedback(analytics_data):
    """
    Uses OpenAI LLM to generate a posture coaching summary based on analytics data.
    Includes frame-level error insights.
    """
    try:
        error_details = []

        # 1️⃣ Extract frequency and frame-level error information
        class_freqs = analytics_data.get("class_frequencies", {})
        error_frames = analytics_data.get("error_frames", {})

        for posture_type, freq in class_freqs.items():
            if posture_type.lower() != "good" and freq > 0:
                frames = error_frames.get(posture_type, [])
                if len(frames) > 0:
                    frame_summary = f"at frames {frames[:10]}..." if len(frames) > 10 else f"at frames {frames}"
                else:
                    frame_summary = "with no specific frame data available"
                error_details.append(f"- {posture_type.capitalize()} issues observed {frame_summary} (frequency: {freq:.2f}).")

        # 2️⃣ Build prompt text
        prompt = f"""
        You are a professional posture correction coach analyzing a user's motion capture session.

        Below are the detected posture errors:
        {chr(10).join(error_details) if error_details else "No major posture issues detected. The user's posture was overall consistent."}

        Based on this data:
        1. Provide a concise summary of the overall posture quality.
        2. Identify which posture areas (spine, knee, etc.) need correction.
        3. Refer to the approximate frame numbers when possible.
        4. Offer 2–3 actionable, encouraging suggestions for improvement.
        5. Keep it under 5 sentences, conversational, and positive.
        """

        # 3️⃣ Call OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a skilled fitness and posture coach."},
                {"role": "user", "content": prompt.strip()}
            ],
            temperature=0.7,
            max_tokens=350,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error generating LLM feedback: {str(e)}"



def encode_image_base64(image_path):
    """
    Converts a PNG image to base64 string.
    """
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    filename = file.filename
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, filename)

    # Save uploaded file
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Convert BVH → CSV if needed
    if filename.endswith(".bvh"):
        csv_path = temp_path.replace(".bvh", ".csv")
        bvh_to_csv(temp_path, csv_path)
    elif filename.endswith(".csv"):
        csv_path = temp_path
    else:
        return JSONResponse({"error": "Unsupported file type"}, status_code=400)

    # Run inference
    df = pd.read_csv(csv_path)
    preds, pred_labels = run_inference(df, model, scaler, le, rot_cols)

    # Create timestamped output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_DIR, ts)
    os.makedirs(output_dir, exist_ok=True)

    # Generate analytics and filter out 'good'
    analytics = generate_analytics(pred_labels, preds, le, output_dir)
    analytics_filtered = {k: v for k, v in analytics.items() if k.lower() != "good"}

    # Encode generated PNG (assuming generate_analytics saves one in output_dir)
    graph_path = os.path.join(output_dir, "posture_graph.png")
    image_base64 = encode_image_base64(graph_path) if os.path.exists(graph_path) else None

    # Generate LLM-based feedback
    feedback = generate_llm_feedback(analytics_filtered)

    # Response
    return JSONResponse({
        "status": "success",
        "file": filename,
        "analytics": analytics_filtered,
        "feedback": feedback,
        "image_base64": image_base64
    })
