"""
AgriLite-Hybrid: Raspberry Pi Inference
========================================
Lightweight inference script for multi-crop disease detection
on Raspberry Pi 3/4/5 using TFLite models.

Supports:
- Single image inference
- Batch inference
- Camera capture with preview
- REST API server mode

Hardware Requirements:
- Raspberry Pi 3B+ / 4 / 5
- 2GB+ RAM recommended
- Camera module or USB webcam
- MicroSD 16GB+

Estimated Performance (Pi 4, 4GB):
- Float32: ~250ms/image
- Float16: ~180ms/image  
- INT8: ~100ms/image
- Full INT8: ~60ms/image

Author: AgriLite Hybrid Project
Date: February 2026
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from PIL import Image

# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class InferenceConfig:
    """Configuration for AgriLite-Hybrid inference."""
    
    # Model paths (relative to this script)
    MODEL_DIR: str = "models"
    MODEL_VARIANTS = {
        'float32': 'agrilite_hybrid.tflite',
        'float16': 'agrilite_hybrid_fp16.tflite',
        'int8': 'agrilite_hybrid_int8.tflite',
        'full_int8': 'agrilite_hybrid_full_int8.tflite'
    }
    
    # Default model
    DEFAULT_MODEL: str = 'int8'
    
    # Input specifications
    INPUT_SIZE: Tuple[int, int] = (224, 224)
    
    # Class labels file
    LABELS_FILE: str = "class_labels.json"
    
    # Confidence threshold
    CONFIDENCE_THRESHOLD: float = 0.5
    
    # Camera settings
    CAMERA_WIDTH: int = 640
    CAMERA_HEIGHT: int = 480
    CAMERA_FRAMERATE: int = 15


# ==============================================================================
# Logging Setup
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('AgriLite-Hybrid')


# ==============================================================================
# TFLite Interpreter
# ==============================================================================

class AgriLiteHybridInference:
    """
    TFLite inference engine for AgriLite-Hybrid model.
    
    Features:
    - Automatic model loading
    - Optimized for Raspberry Pi
    - Multiple model variant support
    - Batch inference
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 model_variant: str = 'int8',
                 labels_path: Optional[str] = None,
                 config: InferenceConfig = None):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to .tflite model (overrides variant)
            model_variant: Model variant ('float32', 'float16', 'int8', 'full_int8')
            labels_path: Path to class labels JSON
            config: Inference configuration
        """
        
        self.config = config or InferenceConfig()
        
        # Import TFLite runtime (prefer lite version for Pi)
        try:
            import tflite_runtime.interpreter as tflite  # type: ignore
            self.tflite = tflite
            logger.info("Using tflite-runtime")
        except ImportError:
            try:
                import tensorflow.lite as tflite
                self.tflite = tflite
                logger.info("Using tensorflow.lite")
            except ImportError:
                import tensorflow as tf
                self.tflite = tf.lite
                logger.info("Using full TensorFlow")
        
        # Resolve model path
        if model_path:
            self.model_path = model_path
        else:
            script_dir = Path(__file__).parent
            model_file = self.config.MODEL_VARIANTS.get(model_variant, 
                                                         self.config.MODEL_VARIANTS['int8'])
            self.model_path = script_dir / self.config.MODEL_DIR / model_file
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        logger.info(f"Loading model: {self.model_path}")
        
        # Load interpreter
        self.interpreter = self.tflite.Interpreter(
            model_path=str(self.model_path),
            num_threads=4  # Pi 4 has 4 cores
        )
        self.interpreter.allocate_tensors()
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Input shape and type
        self.input_shape = self.input_details[0]['shape']
        self.input_dtype = self.input_details[0]['dtype']
        self.input_index = self.input_details[0]['index']
        
        # Check for quantization
        self.is_quantized = self.input_dtype == np.uint8
        if self.is_quantized:
            self.input_scale = self.input_details[0].get('quantization', (0.0, 1))[0]
            self.input_zero_point = self.input_details[0].get('quantization', (0.0, 0))[1]
            logger.info(f"Quantized model: scale={self.input_scale}, zp={self.input_zero_point}")
        
        # Output details
        self.output_index = self.output_details[0]['index']
        self.output_dtype = self.output_details[0]['dtype']
        
        if self.output_dtype == np.uint8:
            self.output_scale = self.output_details[0].get('quantization', (0.0, 1))[0]
            self.output_zero_point = self.output_details[0].get('quantization', (0.0, 0))[1]
        else:
            self.output_scale = 1.0
            self.output_zero_point = 0
        
        # Load class labels
        self.class_labels = self._load_labels(labels_path)
        self.num_classes = len(self.class_labels)
        
        # Crop categorization
        self.crop_classes = self._categorize_classes()
        
        logger.info(f"Model loaded: {self.num_classes} classes")
        logger.info(f"Input shape: {self.input_shape}, dtype: {self.input_dtype}")
        
        # Warm up
        self._warmup()
    
    def _load_labels(self, labels_path: Optional[str] = None) -> List[str]:
        """Load class labels from JSON file."""
        
        if labels_path is None:
            script_dir = Path(__file__).parent
            labels_path = script_dir / self.config.MODEL_DIR / self.config.LABELS_FILE
        
        if os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                data = json.load(f)
                
            if 'class_names' in data:
                return data['class_names']
            elif 'class_indices' in data:
                indices = data['class_indices']
                return [k for k, v in sorted(indices.items(), key=lambda x: x[1])]
            elif 'classes' in data:
                return data['classes']
        
        logger.warning(f"Labels file not found: {labels_path}")
        return [f"Class_{i}" for i in range(19)]  # Default
    
    def _categorize_classes(self) -> Dict[str, List[int]]:
        """Categorize classes by crop type."""
        
        crops = {
            'tomato': [],
            'chilli': [],
            'unknown': []
        }
        
        for idx, name in enumerate(self.class_labels):
            name_lower = name.lower()
            if 'tomato' in name_lower:
                crops['tomato'].append(idx)
            elif 'chilli' in name_lower or 'pepper' in name_lower:
                crops['chilli'].append(idx)
            else:
                crops['unknown'].append(idx)
        
        return crops
    
    def _warmup(self, iterations: int = 3):
        """Warm up the interpreter."""
        
        dummy_input = np.zeros(self.input_shape, dtype=self.input_dtype)
        
        for _ in range(iterations):
            self.interpreter.set_tensor(self.input_index, dummy_input)
            self.interpreter.invoke()
        
        logger.info("Model warmed up")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for inference.
        
        Args:
            image: RGB image (H, W, 3) in range [0, 255]
        
        Returns:
            Preprocessed image tensor
        """
        
        # Resize
        if image.shape[:2] != self.config.INPUT_SIZE:
            pil_image = Image.fromarray(image.astype(np.uint8))
            pil_image = pil_image.resize(self.config.INPUT_SIZE, Image.BILINEAR)
            image = np.array(pil_image)
        
        # Normalize based on model type
        if self.is_quantized:
            # For quantized models, scale to uint8
            if image.max() > 1.0:
                processed = image.astype(np.uint8)
            else:
                processed = (image * 255).astype(np.uint8)
        else:
            # For float models, normalize to [0, 1]
            if image.max() > 1.0:
                processed = image.astype(np.float32) / 255.0
            else:
                processed = image.astype(np.float32)
        
        # Add batch dimension
        return np.expand_dims(processed, axis=0)
    
    def predict(self, image: np.ndarray) -> Dict:
        """
        Run inference on a single image.
        
        Args:
            image: RGB image (H, W, 3)
        
        Returns:
            Dictionary with predictions
        """
        
        start_time = time.perf_counter()
        
        # Preprocess
        input_data = self.preprocess(image)
        
        # Run inference
        self.interpreter.set_tensor(self.input_index, input_data)
        self.interpreter.invoke()
        
        # Get output
        output_data = self.interpreter.get_tensor(self.output_index)
        
        # Dequantize if needed
        if self.output_dtype == np.uint8:
            probs = (output_data.astype(np.float32) - self.output_zero_point) * self.output_scale
        else:
            probs = output_data[0]
        
        # Softmax if not already applied
        if probs.max() > 1.0 or probs.min() < 0.0:
            probs = np.exp(probs - np.max(probs))
            probs = probs / probs.sum()
        
        inference_time = (time.perf_counter() - start_time) * 1000
        
        # Get top prediction
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        class_name = self.class_labels[pred_idx]
        
        # Determine crop type
        crop = 'unknown'
        for crop_name, indices in self.crop_classes.items():
            if pred_idx in indices:
                crop = crop_name
                break
        
        # Get disease name (remove crop prefix)
        disease = class_name
        for prefix in ['tomato_', 'chilli_']:
            if disease.lower().startswith(prefix):
                disease = disease[len(prefix):]
                break
        
        # Check if healthy
        is_healthy = 'healthy' in disease.lower()
        
        # Top-5 predictions
        top_indices = np.argsort(probs)[-5:][::-1]
        top5 = [
            {
                'class': self.class_labels[i],
                'confidence': float(probs[i]),
                'index': int(i)
            }
            for i in top_indices
        ]
        
        return {
            'class_name': class_name,
            'class_index': pred_idx,
            'disease': disease,
            'crop': crop,
            'is_healthy': is_healthy,
            'confidence': confidence,
            'inference_time_ms': inference_time,
            'top5': top5,
            'all_probabilities': probs.tolist(),
            'status': 'high_confidence' if confidence >= self.config.CONFIDENCE_THRESHOLD 
                      else 'low_confidence'
        }
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """Run inference on multiple images."""
        return [self.predict(img) for img in images]
    
    def get_crop_predictions(self, image: np.ndarray) -> Dict:
        """
        Get predictions grouped by crop.
        
        Useful for identifying which crop type is most likely.
        """
        
        result = self.predict(image)
        probs = np.array(result['all_probabilities'])
        
        crop_scores = {}
        for crop_name, indices in self.crop_classes.items():
            if indices:
                crop_scores[crop_name] = float(probs[indices].sum())
        
        result['crop_scores'] = crop_scores
        return result
    
    def benchmark(self, iterations: int = 100) -> Dict:
        """Benchmark inference speed."""
        
        logger.info(f"Benchmarking with {iterations} iterations...")
        
        dummy_image = np.random.randint(0, 255, 
                                        (self.config.INPUT_SIZE[0], 
                                         self.config.INPUT_SIZE[1], 3), 
                                        dtype=np.uint8)
        
        times = []
        for _ in range(iterations):
            result = self.predict(dummy_image)
            times.append(result['inference_time_ms'])
        
        return {
            'iterations': iterations,
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'fps': 1000.0 / np.mean(times)
        }


# ==============================================================================
# Camera Capture
# ==============================================================================

def setup_camera(config: InferenceConfig):
    """Setup Raspberry Pi camera."""
    
    try:
        # Try picamera2 first (newer Pi OS)
        from picamera2 import Picamera2  # type: ignore
        
        camera = Picamera2()
        camera_config = camera.create_preview_configuration(
            main={"size": (config.CAMERA_WIDTH, config.CAMERA_HEIGHT)},
            buffer_count=4
        )
        camera.configure(camera_config)
        camera.start()
        
        logger.info("Using Picamera2")
        return camera, 'picamera2'
        
    except ImportError:
        try:
            # Fallback to legacy picamera
            from picamera import PiCamera  # type: ignore
            from picamera.array import PiRGBArray  # type: ignore
            
            camera = PiCamera()
            camera.resolution = (config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
            camera.framerate = config.CAMERA_FRAMERATE
            raw_capture = PiRGBArray(camera)
            
            logger.info("Using legacy PiCamera")
            return (camera, raw_capture), 'picamera'
            
        except ImportError:
            # Fallback to OpenCV
            import cv2
            
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            
            logger.info("Using OpenCV camera")
            return camera, 'opencv'


def capture_frame(camera, camera_type: str) -> np.ndarray:
    """Capture a frame from the camera."""
    
    if camera_type == 'picamera2':
        frame = camera.capture_array()
        return frame
    
    elif camera_type == 'picamera':
        cam, raw_capture = camera
        raw_capture.truncate(0)
        cam.capture(raw_capture, format='rgb')
        return raw_capture.array
    
    else:  # opencv
        import cv2
        ret, frame = camera.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None


# ==============================================================================
# REST API Server
# ==============================================================================

def create_flask_app(inference_engine: AgriLiteHybridInference):
    """Create Flask REST API for inference."""
    
    try:
        from flask import Flask, request, jsonify
        from werkzeug.utils import secure_filename
    except ImportError:
        logger.error("Flask not installed. Run: pip install flask")
        return None
    
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
    
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({
            'status': 'healthy',
            'model': str(inference_engine.model_path),
            'num_classes': inference_engine.num_classes
        })
    
    @app.route('/predict', methods=['POST'])
    def predict():
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        try:
            image = Image.open(file.stream).convert('RGB')
            image_array = np.array(image)
            
            result = inference_engine.predict(image_array)
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/classes', methods=['GET'])
    def get_classes():
        return jsonify({
            'classes': inference_engine.class_labels,
            'num_classes': inference_engine.num_classes,
            'crops': {
                crop: [inference_engine.class_labels[i] for i in indices]
                for crop, indices in inference_engine.crop_classes.items()
            }
        })
    
    @app.route('/benchmark', methods=['GET'])
    def benchmark():
        iterations = request.args.get('iterations', 50, type=int)
        return jsonify(inference_engine.benchmark(iterations))
    
    return app


# ==============================================================================
# CLI Interface
# ==============================================================================

def predict_image(args):
    """Predict on a single image file."""
    
    logger.info(f"Loading image: {args.image}")
    
    image = Image.open(args.image).convert('RGB')
    image_array = np.array(image)
    
    engine = AgriLiteHybridInference(
        model_path=args.model,
        model_variant=args.variant
    )
    
    result = engine.predict(image_array)
    
    # Display results
    print("\n" + "="*50)
    print("🌱 AGRILITE-HYBRID PREDICTION")
    print("="*50)
    
    crop_emoji = {'tomato': '🍅', 'chilli': '🌶️', 'unknown': '🌿'}
    emoji = crop_emoji.get(result['crop'], '🌿')
    
    print(f"\n{emoji} Crop: {result['crop'].upper()}")
    print(f"📋 Disease: {result['disease']}")
    print(f"🎯 Confidence: {result['confidence']:.2%}")
    health = "✅ Healthy" if result['is_healthy'] else "⚠️ Diseased"
    print(f"🏥 Status: {health}")
    print(f"⏱️ Inference: {result['inference_time_ms']:.1f}ms")
    
    print("\n📊 Top-5 Predictions:")
    for i, pred in enumerate(result['top5'], 1):
        print(f"   {i}. {pred['class']}: {pred['confidence']:.2%}")
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n✓ Results saved to: {args.output}")


def run_camera_demo(args):
    """Run live camera inference demo."""
    
    import cv2
    
    engine = AgriLiteHybridInference(
        model_path=args.model,
        model_variant=args.variant
    )
    
    config = InferenceConfig()
    camera, camera_type = setup_camera(config)
    
    logger.info("Starting camera demo. Press 'q' to quit, 's' to save.")
    
    try:
        while True:
            frame = capture_frame(camera, camera_type)
            if frame is None:
                continue
            
            result = engine.predict(frame)
            
            # Draw results on frame
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Background rectangle
            cv2.rectangle(frame_bgr, (10, 10), (400, 130), (0, 0, 0), -1)
            cv2.rectangle(frame_bgr, (10, 10), (400, 130), (0, 255, 0), 2)
            
            # Text
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (0, 255, 0) if result['is_healthy'] else (0, 0, 255)
            
            cv2.putText(frame_bgr, f"Crop: {result['crop'].upper()}", 
                       (20, 40), font, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_bgr, f"Disease: {result['disease']}", 
                       (20, 70), font, 0.7, color, 2)
            cv2.putText(frame_bgr, f"Conf: {result['confidence']:.1%} | {result['inference_time_ms']:.0f}ms", 
                       (20, 100), font, 0.6, (255, 255, 0), 2)
            
            cv2.imshow('AgriLite-Hybrid', frame_bgr)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"capture_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame_bgr)
                logger.info(f"Saved: {filename}")
    
    finally:
        cv2.destroyAllWindows()
        if camera_type == 'opencv':
            camera.release()


def run_server(args):
    """Run REST API server."""
    
    engine = AgriLiteHybridInference(
        model_path=args.model,
        model_variant=args.variant
    )
    
    app = create_flask_app(engine)
    
    if app:
        logger.info(f"Starting server on http://0.0.0.0:{args.port}")
        app.run(host='0.0.0.0', port=args.port, debug=args.debug)


def run_benchmark(args):
    """Run inference benchmark."""
    
    engine = AgriLiteHybridInference(
        model_path=args.model,
        model_variant=args.variant
    )
    
    results = engine.benchmark(args.iterations)
    
    print("\n" + "="*50)
    print("⚡ BENCHMARK RESULTS")
    print("="*50)
    print(f"Model: {engine.model_path}")
    print(f"Iterations: {results['iterations']}")
    print(f"\nInference Time:")
    print(f"  Mean: {results['mean_ms']:.2f} ms")
    print(f"  Std:  {results['std_ms']:.2f} ms")
    print(f"  Min:  {results['min_ms']:.2f} ms")
    print(f"  Max:  {results['max_ms']:.2f} ms")
    print(f"\nThroughput: {results['fps']:.1f} FPS")


def main():
    """CLI entry point."""
    
    parser = argparse.ArgumentParser(
        description="AgriLite-Hybrid: Multi-Crop Disease Detection for Raspberry Pi",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model', '-m', type=str, default=None,
                       help='Path to TFLite model')
    parser.add_argument('--variant', '-v', type=str, default='int8',
                       choices=['float32', 'float16', 'int8', 'full_int8'],
                       help='Model variant (default: int8)')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict on an image')
    predict_parser.add_argument('image', type=str, help='Path to image')
    predict_parser.add_argument('--output', '-o', type=str, help='Output JSON file')
    
    # Camera command
    camera_parser = subparsers.add_parser('camera', help='Live camera demo')
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Run REST API server')
    server_parser.add_argument('--port', '-p', type=int, default=5000)
    server_parser.add_argument('--debug', '-d', action='store_true')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run benchmark')
    bench_parser.add_argument('--iterations', '-n', type=int, default=100)
    
    args = parser.parse_args()
    
    if args.command == 'predict':
        predict_image(args)
    elif args.command == 'camera':
        run_camera_demo(args)
    elif args.command == 'server':
        run_server(args)
    elif args.command == 'benchmark':
        run_benchmark(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
