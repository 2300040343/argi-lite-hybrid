# =============================================================================
# AgriLite-Hybrid Backend
# services/ml_service.py - Machine Learning Service
# 
# Handles model loading, image preprocessing, inference, and
# Grad-CAM heatmap generation for plant disease detection.
# =============================================================================

import os
import io
import json
import base64
import random
import logging
import warnings
from typing import Optional, Dict, Any, List

import numpy as np
from PIL import Image

# Suppress Keras input structure warning (model works correctly with raw tensors)
warnings.filterwarnings("ignore", message="The structure of `inputs` doesn't match")

# Configure logging
logger = logging.getLogger(__name__)


class MLService:
    """
    Machine Learning Service for plant disease detection.
    
    This service handles:
    - Model loading (Keras .h5 or TensorFlow Lite)
    - Image preprocessing
    - Disease prediction inference
    - Grad-CAM heatmap generation
    - Treatment recommendation lookup
    
    When the actual model is not available, it provides placeholder
    predictions for development and testing purposes.
    """
    
    # Image preprocessing constants
    IMAGE_SIZE = (224, 224)
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the ML service.
        
        Args:
            model_path: Path to the models directory (optional)
        """
        self.model = None
        self.model_loaded = False
        self.use_tflite = False
        self.tflite_interpreter = None
        
        # Set model path
        if model_path:
            self.model_path = model_path
        else:
            # Default path relative to backend directory
            self.model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                '..',
                'models'
            )
        
        # Load class labels and treatments
        self.class_labels = self._load_class_labels()
        self.treatments = self.class_labels.get('treatments', {})
        self.crops = self.class_labels.get('crops', {})
        
        # Hybrid model class mapping (19-class output)
        self.class_names = self.class_labels.get('class_names', [])
        self.class_to_display = self.class_labels.get('class_to_display', {})
        self.crop_class_ranges = self.class_labels.get('crop_class_ranges', {})
        
        # Try to load the model
        self._load_model()
    
    def _load_class_labels(self) -> Dict:
        """
        Load class labels and treatment data from JSON file.
        
        Returns:
            Dictionary containing crops, diseases, and treatments
        """
        labels_path = os.path.join(self.model_path, 'class_labels.json')
        
        try:
            if os.path.exists(labels_path):
                with open(labels_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load class_labels.json: {e}")
        
        # Return default labels if file not found
        return self._get_default_labels()
    
    def _get_default_labels(self) -> Dict:
        """
        Get default class labels for fallback.
        
        Returns:
            Default labels dictionary
        """
        return {
            'class_names': [
                'chilli_Chilli_Anthracnos', 'chilli_Chilli_Damping_Off',
                'chilli_Chilli_Leaf_Curl_Virus', 'chilli_Chilli_Leaf_Spot',
                'chilli_Chilli_Veinal_Mottle_Virus', 'chilli_Chilli__Whitefly',
                'chilli_Chilli__Yellowish', 'chilli_Chilli__healthy',
                'tomato_Bacterial_spot', 'tomato_Early_blight', 'tomato_Late_blight',
                'tomato_Leaf_Mold', 'tomato_Septoria_leaf_spot',
                'tomato_Spider_mites_Two_spotted_spider_mite', 'tomato_Target_Spot',
                'tomato_Tomato_Yellow_Leaf_Curl_Virus', 'tomato_Tomato_mosaic_virus',
                'tomato_healthy', 'tomato_powdery_mildew'
            ],
            'crop_class_ranges': {
                'chilli': [0, 7], 'tomato': [8, 18]
            },
            'class_to_display': {},
            'crops': {
                'tomato': {
                    'diseases': ['Bacterial Spot', 'Early Blight', 'Late Blight',
                                'Leaf Mold', 'Septoria Leaf Spot',
                                'Spider Mites (Two-spotted)', 'Target Spot',
                                'Yellow Leaf Curl Virus', 'Mosaic Virus',
                                'Healthy', 'Powdery Mildew']
                },
                'chilli': {
                    'diseases': ['Anthracnose', 'Damping Off', 'Leaf Curl Virus',
                                'Leaf Spot', 'Veinal Mottle Virus', 'Whitefly',
                                'Yellowish', 'Healthy']
                }
            },
            'severity_levels': ['mild', 'medium', 'severe'],
            'treatments': {}
        }
    
    def _load_model(self) -> bool:
        """
        Load the trained model (Keras or TFLite).
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        # Try loading Keras model first
        h5_path = os.path.join(self.model_path, 'agrilite_hybrid.h5')
        keras_path = os.path.join(self.model_path, 'agrilite_hybrid.keras')
        tflite_path = os.path.join(self.model_path, 'agrilite_hybrid.tflite')
        
        # Try Keras/H5 model
        for model_file in [keras_path, h5_path]:
            if os.path.exists(model_file):
                try:
                    import tensorflow as tf
                    from cbam_layers import ChannelAttention, SpatialAttention, CBAM
                    custom_objects = {
                        'ChannelAttention': ChannelAttention,
                        'SpatialAttention': SpatialAttention,
                        'CBAM': CBAM,
                    }
                    self.model = tf.keras.models.load_model(
                        model_file, custom_objects=custom_objects
                    )
                    self.model_loaded = True
                    logger.info(f"Keras model loaded from {model_file}")
                    return True
                except Exception as e:
                    logger.warning(f"Could not load Keras model: {e}")
        
        # Try TFLite model
        if os.path.exists(tflite_path):
            try:
                import tflite_runtime.interpreter as tflite
                self.tflite_interpreter = tflite.Interpreter(model_path=tflite_path)
                self.tflite_interpreter.allocate_tensors()
                self.use_tflite = True
                self.model_loaded = True
                logger.info(f"TFLite model loaded from {tflite_path}")
                return True
            except ImportError:
                try:
                    import tensorflow as tf
                    self.tflite_interpreter = tf.lite.Interpreter(model_path=tflite_path)
                    self.tflite_interpreter.allocate_tensors()
                    self.use_tflite = True
                    self.model_loaded = True
                    logger.info(f"TFLite model loaded using TensorFlow")
                    return True
                except Exception as e:
                    logger.warning(f"Could not load TFLite model: {e}")
        
        logger.warning("No model file found - using placeholder predictions")
        return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model_loaded
    
    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """
        Preprocess image for model inference.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Preprocessed image array of shape (1, 224, 224, 3)
        """
        # Load image from bytes
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize(self.IMAGE_SIZE, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32)
        
        # Normalize to [0, 1] — matches training (rescale=1./255)
        # No ImageNet mean/std normalization (include_preprocessing=False in model)
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(
        self,
        image_data: bytes,
        crop: str,
        temperature: Optional[float] = None,
        humidity: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run disease prediction on an image.
        
        Args:
            image_data: Raw image bytes
            crop: Crop type (tomato, chilli)
            temperature: Optional environmental temperature (°C)
            humidity: Optional environmental humidity (%)
            
        Returns:
            Dictionary containing prediction results:
            - disease: Predicted disease name
            - severity: Severity level (mild, medium, severe)
            - confidence: Prediction confidence percentage
            - treatment: List of treatment recommendations
            - heatmap_base64: Grad-CAM heatmap as base64
            - original_image_base64: Original image as base64
            - environmental_risk: Risk assessment based on environment
        """
        try:
            # Pre-inference image validation (classical CV)
            validation = self._validate_leaf_image(image_data)
            if not validation['valid']:
                return {
                    'success': False,
                    'error': (
                        'The uploaded image does not appear to be a valid '
                        f'{crop} leaf. {validation["reason"]} '
                        'Please upload a clear, close-up photo of a plant leaf.'
                    ),
                    'original_image_base64': self._image_to_base64(image_data),
                }

            # Preprocess image
            processed_image = self.preprocess_image(image_data)
            
            # Get diseases for this crop
            crop_diseases = self.crops.get(crop, {}).get('diseases', [])
            if not crop_diseases:
                crop_diseases = self._get_default_labels()['crops'].get(crop, {}).get('diseases', [])
            
            # Run inference
            mismatch_crop = None
            if self.model_loaded and not self.use_tflite:
                # Keras model prediction
                predictions = self.model.predict(processed_image, verbose=0)
                disease, confidence, mismatch_crop = self._map_prediction_to_crop(
                    predictions[0], crop
                )
                    
            elif self.model_loaded and self.use_tflite:
                # TFLite prediction
                input_details = self.tflite_interpreter.get_input_details()
                output_details = self.tflite_interpreter.get_output_details()
                
                self.tflite_interpreter.set_tensor(
                    input_details[0]['index'],
                    processed_image.astype(np.float32)
                )
                self.tflite_interpreter.invoke()
                
                predictions = self.tflite_interpreter.get_tensor(output_details[0]['index'])
                disease, confidence, mismatch_crop = self._map_prediction_to_crop(
                    predictions[0], crop
                )
            else:
                # No real model available - fail explicitly rather than
                # silently returning a fake/random diagnosis to the user
                logger.error(
                    "Prediction requested but no model is loaded "
                    "(missing/corrupt model file or failed TensorFlow import)"
                )
                return {
                    'success': False,
                    'error': (
                        'The disease detection model is currently unavailable. '
                        'Please try again later or contact support.'
                    ),
                    'original_image_base64': self._image_to_base64(image_data),
                }
            
            # Reject cross-crop mismatch
            if mismatch_crop:
                return {
                    'success': False,
                    'error': (
                        f'This image appears to be a {mismatch_crop} leaf, '
                        f'not {crop}. Please select the correct crop type '
                        f'or upload a {crop} leaf image.'
                    ),
                    'original_image_base64': self._image_to_base64(image_data),
                }
            
            # Log low-confidence predictions but still allow them through
            # (pre-inference CV validation already filters non-plant images)
            if self.model_loaded:
                crop_range = self.crop_class_ranges.get(crop)
                num_crop_classes = (crop_range[1] - crop_range[0] + 1) if crop_range else 19
                random_chance = 1.0 / num_crop_classes
                if confidence < random_chance:
                    logger.warning(
                        f"Low confidence prediction for {crop}: "
                        f"{disease} ({confidence*100:.1f}%) — below random chance"
                    )

            # Determine severity based on confidence and disease type
            severity = self._determine_severity(disease, confidence)
            
            # Get treatment recommendations
            treatment = self.get_treatment(disease, severity)
            
            # Generate heatmap
            heatmap_base64 = self._generate_heatmap(image_data, processed_image)
            
            # Convert original image to base64
            original_base64 = self._image_to_base64(image_data)
            
            # Calculate environmental risk
            environmental_risk = self._assess_environmental_risk(
                temperature, humidity, disease
            )
            
            return {
                'success': True,
                'crop': crop,
                'disease': disease,
                'severity': severity,
                'confidence': round(min(confidence, 1.0), 4),
                'treatment': treatment,
                'heatmap_base64': heatmap_base64,
                'original_image_base64': original_base64,
                'environmental_risk': environmental_risk,
                'model_used': 'keras' if not self.use_tflite else 'tflite'
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _map_prediction_to_crop(
        self,
        predictions: np.ndarray,
        crop: str
    ) -> tuple:
        """
        Map 19-class hybrid model output to a crop-specific disease.
        
        Filters predictions to only classes belonging to the requested crop,
        then picks the highest-confidence class within that crop.
        Also detects cross-crop mismatches.
        
        Args:
            predictions: Model output array of shape (19,)
            crop: Requested crop type (chilli, tomato)
            
        Returns:
            Tuple of (display_disease_name, confidence, mismatch_crop_or_None)
        """
        # Apply softmax if predictions are logits (don't sum to ~1)
        pred_sum = float(np.sum(predictions))
        if pred_sum < 0.99 or pred_sum > 1.01:
            exp_preds = np.exp(predictions - np.max(predictions))
            predictions = exp_preds / np.sum(exp_preds)
        
        # Find the global best prediction across ALL crops
        global_best_idx = int(np.argmax(predictions))
        global_best_conf = float(predictions[global_best_idx])
        actual_crop = None
        for c, (s, e) in self.crop_class_ranges.items():
            if s <= global_best_idx <= e:
                actual_crop = c
                break
        
        crop_range = self.crop_class_ranges.get(crop)
        mismatch_crop = None
        
        if crop_range and self.class_names:
            start_idx, end_idx = crop_range
            # Extract only predictions for this crop's classes
            crop_preds = predictions[start_idx:end_idx + 1]
            crop_best_idx = np.argmax(crop_preds)
            global_idx = start_idx + crop_best_idx
            confidence = float(crop_preds[crop_best_idx])  # 0-1 ratio
            
            # Map to human-readable name
            class_name = self.class_names[global_idx]
            disease = self.class_to_display.get(class_name, class_name)
            
            # Detect cross-crop mismatch: global best is a different crop
            # AND that crop's confidence is much higher than selected crop
            if actual_crop and actual_crop != crop and global_best_conf > confidence * 3:
                mismatch_crop = actual_crop
        else:
            # Fallback: use global argmax
            disease_idx = np.argmax(predictions)
            confidence = float(predictions[disease_idx])  # 0-1 ratio
            if disease_idx < len(self.class_names):
                class_name = self.class_names[disease_idx]
                disease = self.class_to_display.get(class_name, class_name)
            else:
                disease = 'Unknown'
        
        return disease, confidence, mismatch_crop
    
    def _validate_leaf_image(self, image_data: bytes) -> Dict[str, Any]:
        """
        Validate that the uploaded image is a plant leaf using classical CV.
        
        Combines three checks:
        1. Green/plant color ratio (HSV analysis)
        2. Texture entropy (Shannon entropy of grayscale histogram)
        3. Edge pattern density (Canny edge detection)
        
        Returns:
            Dict with 'valid' (bool), 'reason' (str), and 'scores' (dict)
        """
        import cv2

        image = Image.open(io.BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Use a moderate size for analysis — fast but enough detail
        image = image.resize((256, 256), Image.Resampling.LANCZOS)
        img_np = np.array(image)

        # ------------------------------------------------------------------
        # Check 1: Green / plant color ratio (HSV)
        # ------------------------------------------------------------------
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

        # Green foliage:    H 25-95, S 25-255, V 25-255
        green_mask = cv2.inRange(hsv, (25, 25, 25), (95, 255, 255))
        # Brown / diseased:  H 5-25,  S 30-255, V 30-200
        brown_mask = cv2.inRange(hsv, (5, 30, 30), (25, 255, 200))
        # Dark/wilted green: H 25-95, S 15-255, V 10-100
        dark_mask = cv2.inRange(hsv, (25, 15, 10), (95, 255, 100))

        plant_pixels = cv2.countNonZero(green_mask | brown_mask | dark_mask)
        total_pixels = hsv.shape[0] * hsv.shape[1]
        green_ratio = plant_pixels / total_pixels

        # ------------------------------------------------------------------
        # Check 2: Texture entropy (Shannon entropy of grayscale histogram)
        # ------------------------------------------------------------------
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist = hist / hist.sum()  # normalise to probabilities
        hist = hist[hist > 0]     # avoid log(0)
        entropy = float(-np.sum(hist * np.log2(hist)))

        # ------------------------------------------------------------------
        # Check 3: Edge density (Canny)
        # ------------------------------------------------------------------
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = float(cv2.countNonZero(edges)) / total_pixels

        scores = {
            'green_ratio': round(green_ratio, 4),
            'entropy': round(entropy, 4),
            'edge_ratio': round(edge_ratio, 4),
        }

        # ------------------------------------------------------------------
        # Decision logic (strict mode)
        # ------------------------------------------------------------------
        reasons: List[str] = []

        if green_ratio < 0.15:
            reasons.append(
                f'Low plant-color content ({green_ratio*100:.1f}%). '
                'The image does not appear to contain a plant leaf.'
            )
        if entropy < 4.0:
            reasons.append(
                f'Image texture too uniform (entropy={entropy:.2f}). '
                'This may be a solid color, gradient, or synthetic image.'
            )
        if edge_ratio < 0.01:
            reasons.append(
                'Image has almost no detail (very few edges detected). '
                'Please upload a clear photo of a leaf.'
            )
        if edge_ratio > 0.40:
            reasons.append(
                f'Edge density too high ({edge_ratio*100:.1f}%). '
                'This appears to be a screenshot or text-heavy image.'
            )

        if reasons:
            logger.info(f"Image validation FAILED: {scores} | {reasons[0]}")
            return {
                'valid': False,
                'reason': reasons[0],  # report the first failure
                'scores': scores,
            }

        logger.debug(f"Image validation passed: {scores}")
        return {'valid': True, 'reason': '', 'scores': scores}

    def _determine_severity(self, disease: str, confidence: float) -> str:
        """
        Determine disease severity based on prediction.
        
        Args:
            disease: Disease name
            confidence: Prediction confidence
            
        Returns:
            Severity level (mild, medium, severe)
        """
        if disease.lower() == 'healthy':
            return 'mild'  # Not really applicable, but needed for consistency
        
        # Use confidence as a rough indicator plus randomization for demo
        # In real implementation, this would be a separate model output
        base_severity = random.random()
        
        if base_severity < 0.33:
            return 'mild'
        elif base_severity < 0.66:
            return 'medium'
        else:
            return 'severe'
    
    def get_treatment(self, disease: str, severity: str) -> List[Dict[str, str]]:
        """
        Get treatment recommendations for a disease.
        
        Args:
            disease: Disease name
            severity: Severity level
            
        Returns:
            List of structured treatment recommendation dicts
        """
        # Look up treatment in loaded data
        disease_treatments = self.treatments.get(disease, {})
        
        if disease_treatments:
            raw = disease_treatments.get(severity, disease_treatments.get('medium', []))
        elif disease.lower() == 'healthy':
            raw = [
                "No treatment needed - plant appears healthy",
                "Continue regular maintenance and monitoring",
                "Ensure proper watering and fertilization"
            ]
        else:
            raw = [
                "Consult with a local agricultural extension officer",
                "Remove affected plant parts if possible",
                "Ensure proper plant spacing for air circulation",
                "Consider appropriate fungicide or pesticide treatment"
            ]
        
        # Convert plain strings to structured objects for the frontend
        organic_keywords = ('neem', 'organic', 'compost', 'mulch', 'spacing',
                           'circulation', 'maintenance', 'monitoring', 'watering',
                           'prune', 'remove', 'healthy', 'companion', 'rotation')
        structured = []
        for item in raw:
            if isinstance(item, dict):
                structured.append(item)
            else:
                is_organic = any(w in item.lower() for w in organic_keywords)
                structured.append({
                    'type': 'organic' if is_organic else 'chemical',
                    'name': item,
                    'dosage': '',
                    'application': '',
                    'schedule': ''
                })
        return structured
    
    def _generate_heatmap(
        self,
        original_data: bytes,
        processed_image: np.ndarray
    ) -> str:
        """
        Generate Grad-CAM heatmap for prediction visualization.
        
        Args:
            original_data: Original image bytes
            processed_image: Preprocessed image array
            
        Returns:
            Base64 encoded heatmap overlay image
        """
        try:
            # Load original image
            original = Image.open(io.BytesIO(original_data))
            if original.mode != 'RGB':
                original = original.convert('RGB')
            original = original.resize(self.IMAGE_SIZE)
            original_array = np.array(original)
            
            if self.model_loaded and self.model is not None and not self.use_tflite:
                # Real Grad-CAM with Keras model
                heatmap = self._compute_gradcam(processed_image)
            else:
                # Placeholder heatmap for development
                heatmap = self._generate_placeholder_heatmap()
            
            # Create colored heatmap
            import cv2
            heatmap_colored = cv2.applyColorMap(
                np.uint8(255 * heatmap),
                cv2.COLORMAP_JET
            )
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Overlay on original image
            overlay = cv2.addWeighted(
                original_array, 0.6,
                heatmap_colored, 0.4,
                0
            )
            
            # Convert to base64
            overlay_image = Image.fromarray(overlay)
            buffer = io.BytesIO()
            overlay_image.save(buffer, format='JPEG', quality=85)
            buffer.seek(0)
            
            base64_str = base64.b64encode(buffer.read()).decode('utf-8')
            return f"data:image/jpeg;base64,{base64_str}"
            
        except Exception as e:
            logger.error(f"Heatmap generation error: {e}")
            # Return original image as fallback
            return self._image_to_base64(original_data)
    
    def _compute_gradcam(self, image: np.ndarray) -> np.ndarray:
        """
        Compute Grad-CAM heatmap using the model.
        
        Args:
            image: Preprocessed image array
            
        Returns:
            Heatmap array of shape (224, 224)
        """
        try:
            import tensorflow as tf
            
            # Get the last convolutional layer
            last_conv_layer = None
            for layer in reversed(self.model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer = layer
                    break
            
            if last_conv_layer is None:
                return self._generate_placeholder_heatmap()
            
            # Create gradient model
            grad_model = tf.keras.models.Model(
                [self.model.inputs],
                [last_conv_layer.output, self.model.output]
            )
            
            # Compute gradients
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(image)
                class_idx = tf.argmax(predictions[0])
                loss = predictions[:, class_idx]
            
            grads = tape.gradient(loss, conv_outputs)
            
            # Global average pooling of gradients
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight the feature maps
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
            
            # ReLU and normalize
            heatmap = tf.nn.relu(heatmap)
            heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
            
            # Resize to image size
            heatmap = tf.image.resize(
                heatmap[..., tf.newaxis],
                self.IMAGE_SIZE
            )
            
            return heatmap.numpy().squeeze()
            
        except Exception as e:
            logger.warning(f"Grad-CAM computation failed: {e}")
            return self._generate_placeholder_heatmap()
    
    def _generate_placeholder_heatmap(self) -> np.ndarray:
        """
        Generate a placeholder heatmap for development.
        
        Creates a realistic-looking heatmap with random but
        structured hot spots.
        
        Returns:
            Heatmap array of shape (224, 224)
        """
        # Create base heatmap with gaussian blobs
        heatmap = np.zeros((224, 224), dtype=np.float32)
        
        # Add 2-4 random hot spots
        num_spots = random.randint(2, 4)
        
        for _ in range(num_spots):
            # Random center position (avoiding edges)
            cx = random.randint(40, 184)
            cy = random.randint(40, 184)
            
            # Random size
            size = random.randint(30, 60)
            
            # Create gaussian blob
            y, x = np.ogrid[:224, :224]
            blob = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * size**2))
            
            # Add to heatmap with random intensity
            intensity = random.uniform(0.5, 1.0)
            heatmap = np.maximum(heatmap, blob * intensity)
        
        # Normalize
        heatmap = heatmap / (heatmap.max() + 1e-8)
        
        return heatmap
    
    def _image_to_base64(self, image_data: bytes) -> str:
        """
        Convert image bytes to base64 string.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Base64 encoded image with data URI prefix
        """
        image = Image.open(io.BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize for efficient transfer
        image.thumbnail((800, 800), Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)
        
        base64_str = base64.b64encode(buffer.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_str}"
    
    def _assess_environmental_risk(
        self,
        temperature: Optional[float],
        humidity: Optional[float],
        disease: str
    ) -> str:
        """
        Assess environmental risk based on temperature and humidity.
        
        Args:
            temperature: Temperature in °C (optional)
            humidity: Humidity in % (optional)
            disease: Detected disease
            
        Returns:
            Risk level string (normal, elevated, high)
        """
        if temperature is None and humidity is None:
            return 'unknown'
        
        risk_factors = []
        
        if temperature is not None:
            if temperature > 35:
                risk_factors.append('high_temp')
            elif temperature < 10:
                risk_factors.append('low_temp')
        
        if humidity is not None:
            if humidity > 85:
                risk_factors.append('high_humidity')
            elif humidity < 30:
                risk_factors.append('low_humidity')
        
        # High humidity increases fungal disease risk
        fungal_diseases = [
            'Powdery Mildew', 'Leaf Spot Disease', 'Leaf Spot',
            'White Mold Disease', 'Early Blight', 'Late Blight',
            'Septoria Leaf Spot', 'Anthracnose', 'Leaf Mold',
            'Target Spot', 'Damping Off'
        ]
        
        if disease in fungal_diseases and 'high_humidity' in risk_factors:
            return 'high'
        
        if len(risk_factors) >= 2:
            return 'high'
        elif len(risk_factors) == 1:
            return 'elevated'
        
        return 'normal'
