#!/usr/bin/env python3
"""
=============================================================================
AgriLite-Hybrid - Raspberry Pi Prediction Script
predict_pi.py - Standalone inference for edge deployment

Performs plant disease detection using TensorFlow Lite model on Raspberry Pi.
Supports image file input and camera capture.
=============================================================================
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Try to import TFLite runtime (lighter) or fall back to full TensorFlow
try:
    import tflite_runtime.interpreter as tflite
    USING_TFLITE = True
except ImportError:
    try:
        import tensorflow as tf
        tflite = tf.lite
        USING_TFLITE = False
    except ImportError:
        print("Error: Neither tflite_runtime nor tensorflow is installed.")
        print("Install with: pip3 install tflite-runtime")
        sys.exit(1)

# =============================================================================
# Configuration
# =============================================================================

# Paths
SCRIPT_DIR = Path(__file__).parent.absolute()
MODEL_PATH = SCRIPT_DIR / "models" / "agrilite_model.tflite"
LABELS_PATH = SCRIPT_DIR / "models" / "class_labels.json"

# Input image size
INPUT_SIZE = (224, 224)

# Severity thresholds
SEVERITY_THRESHOLDS = {
    "mild": 0.3,
    "moderate": 0.6,
    "severe": 0.8
}

# =============================================================================
# Helper Functions
# =============================================================================

def load_labels(labels_path: Path) -> dict:
    """Load class labels from JSON file."""
    if not labels_path.exists():
        print(f"Warning: Labels file not found at {labels_path}")
        return {}
    
    with open(labels_path, 'r') as f:
        return json.load(f)


def preprocess_image(image_path: str, target_size: tuple = INPUT_SIZE) -> np.ndarray:
    """
    Load and preprocess image for model input.
    
    Args:
        image_path: Path to image file
        target_size: Target size (width, height)
    
    Returns:
        Preprocessed image as numpy array
    """
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size, Image.BILINEAR)
    
    # Convert to numpy and normalize
    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def capture_image_from_camera(output_path: str = "capture.jpg") -> str:
    """
    Capture image from Raspberry Pi camera.
    
    Args:
        output_path: Path to save captured image
    
    Returns:
        Path to captured image
    """
    try:
        from picamera2 import Picamera2
        
        picam2 = Picamera2()
        config = picam2.create_still_configuration(
            main={"size": (1920, 1080)}
        )
        picam2.configure(config)
        picam2.start()
        
        # Wait for camera to warm up
        time.sleep(2)
        
        # Capture image
        picam2.capture_file(output_path)
        picam2.stop()
        
        print(f"Image captured and saved to: {output_path}")
        return output_path
        
    except ImportError:
        print("Error: picamera2 not installed. Install with: pip3 install picamera2")
        sys.exit(1)
    except Exception as e:
        print(f"Error capturing image: {e}")
        sys.exit(1)


def determine_severity(confidence: float, class_name: str) -> str:
    """
    Determine disease severity based on confidence and class.
    
    Args:
        confidence: Model confidence score
        class_name: Predicted class name
    
    Returns:
        Severity level string
    """
    # Check if healthy
    if "healthy" in class_name.lower():
        return "healthy"
    
    # Determine severity based on confidence
    if confidence >= SEVERITY_THRESHOLDS["severe"]:
        return "severe"
    elif confidence >= SEVERITY_THRESHOLDS["moderate"]:
        return "moderate"
    else:
        return "mild"


def get_treatment_recommendations(disease: str, crop: str) -> list:
    """
    Get treatment recommendations for detected disease.
    
    Args:
        disease: Detected disease name
        crop: Crop type
    
    Returns:
        List of treatment recommendations
    """
    # Basic treatment recommendations (can be expanded)
    treatments = {
        "cercospora": [
            {"type": "Chemical", "name": "Mancozeb 75% WP", "dosage": "2-3g/L water"},
            {"type": "Organic", "name": "Neem oil spray", "dosage": "5ml/L water"}
        ],
        "mosaic": [
            {"type": "Prevention", "name": "Remove infected plants", "dosage": "N/A"},
            {"type": "Prevention", "name": "Control aphid vectors", "dosage": "N/A"}
        ],
        "blight": [
            {"type": "Chemical", "name": "Copper oxychloride", "dosage": "3g/L water"},
            {"type": "Organic", "name": "Bordeaux mixture", "dosage": "1% solution"}
        ],
        "default": [
            {"type": "General", "name": "Consult local agricultural officer", "dosage": "N/A"}
        ]
    }
    
    # Find matching treatment
    disease_lower = disease.lower()
    for key, treatment in treatments.items():
        if key in disease_lower:
            return treatment
    
    return treatments["default"]


# =============================================================================
# Main Prediction Class
# =============================================================================

class AgriLitePredictor:
    """Handles plant disease prediction using TFLite model."""
    
    def __init__(self, model_path: Path = MODEL_PATH, labels_path: Path = LABELS_PATH):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to TFLite model file
            labels_path: Path to class labels JSON file
        """
        self.model_path = model_path
        self.labels = load_labels(labels_path)
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        
        self._load_model()
    
    def _load_model(self):
        """Load TFLite model and allocate tensors."""
        if not self.model_path.exists():
            print(f"Error: Model file not found at {self.model_path}")
            print("Please copy your trained model to the models/ directory")
            sys.exit(1)
        
        print(f"Loading model from: {self.model_path}")
        print(f"Using {'TFLite Runtime' if USING_TFLITE else 'TensorFlow'}")
        
        # Create interpreter
        self.interpreter = tflite.Interpreter(
            model_path=str(self.model_path),
            num_threads=4  # Use multiple cores on Pi 4
        )
        
        # Try to use XNNPACK delegate for ARM acceleration
        try:
            self.interpreter = tflite.Interpreter(
                model_path=str(self.model_path),
                num_threads=4,
                experimental_delegates=[
                    tflite.load_delegate('libXNNPACKDelegate.so')
                ]
            )
            print("XNNPACK delegate enabled for ARM acceleration")
        except Exception:
            pass  # Fall back to default interpreter
        
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print("Model loaded successfully")
    
    def predict(self, image_path: str, crop: str) -> dict:
        """
        Perform disease prediction on image.
        
        Args:
            image_path: Path to image file
            crop: Crop type (tomato, chilli)
        
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        print(f"Processing image: {image_path}")
        start_time = time.time()
        
        img_array = preprocess_image(image_path)
        
        # Set input tensor
        self.interpreter.set_tensor(
            self.input_details[0]['index'],
            img_array
        )
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output tensor
        output_data = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )
        
        inference_time = time.time() - start_time
        
        # Process predictions
        predictions = output_data[0]
        top_idx = np.argmax(predictions)
        confidence = float(predictions[top_idx])
        
        # Get class name from labels
        crop_labels = self.labels.get(crop.lower(), {}).get("classes", [])
        if top_idx < len(crop_labels):
            class_name = crop_labels[top_idx]
        else:
            class_name = f"class_{top_idx}"
        
        # Determine severity
        severity = determine_severity(confidence, class_name)
        
        # Get treatment recommendations
        treatments = get_treatment_recommendations(class_name, crop)
        
        result = {
            "success": True,
            "crop": crop,
            "disease": class_name,
            "confidence": round(confidence, 4),
            "severity": severity,
            "treatment": treatments,
            "inference_time_ms": round(inference_time * 1000, 2),
            "model": "AgriLite-Hybrid TFLite"
        }
        
        return result


def print_result(result: dict):
    """Pretty print prediction result."""
    print("\n" + "="*60)
    print("AGRILITE-HYBRID PREDICTION RESULT")
    print("="*60)
    
    if result.get("success"):
        print(f"Crop:           {result['crop']}")
        print(f"Disease:        {result['disease']}")
        print(f"Confidence:     {result['confidence']*100:.1f}%")
        print(f"Severity:       {result['severity'].upper()}")
        print(f"Inference Time: {result['inference_time_ms']:.2f} ms")
        
        print("\nTreatment Recommendations:")
        for i, treatment in enumerate(result.get('treatment', []), 1):
            print(f"  {i}. [{treatment['type']}] {treatment['name']}")
            if treatment.get('dosage') and treatment['dosage'] != 'N/A':
                print(f"     Dosage: {treatment['dosage']}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    print("="*60 + "\n")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AgriLite-Hybrid Plant Disease Detection for Raspberry Pi"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        help="Path to input image file"
    )
    parser.add_argument(
        "--camera", "-c",
        action="store_true",
        help="Capture image from Pi camera"
    )
    parser.add_argument(
        "--crop",
        type=str,
        required=True,
        choices=["tomato", "chilli"],
        help="Crop type for prediction"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(MODEL_PATH),
        help="Path to TFLite model file"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.image and not args.camera:
        parser.error("Either --image or --camera must be specified")
    
    # Capture image if using camera
    if args.camera:
        image_path = capture_image_from_camera()
    else:
        image_path = args.image
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            sys.exit(1)
    
    # Initialize predictor
    model_path = Path(args.model)
    predictor = AgriLitePredictor(model_path=model_path)
    
    # Run prediction
    result = predictor.predict(image_path, args.crop)
    
    # Output result
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_result(result)


if __name__ == "__main__":
    main()
