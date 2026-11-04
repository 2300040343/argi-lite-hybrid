# AgriLite-Hybrid Methodology and Technical Reference

## 1. Purpose and Scope
This document captures the end-to-end methodology and technical implementation of AgriLite-Hybrid, including:
- Training methodology
- Inference methodology
- Backend and frontend technical stack
- Concrete configuration values
- API request and response samples
- Deployment values

This is based on current project implementation files.

## 2. System Overview
AgriLite-Hybrid is a multi-crop disease detection platform with:
- Flask backend API
- React + Vite frontend
- Hybrid deep learning model (MobileNetV3-Small + EfficientNetV2-B0 + CBAM)
- Optional edge deployment path (TFLite-ready)

Supported crops and class counts:
- Chilli: 8 classes
- Tomato: 11 classes
- Total: 19 classes

(Brinjal was trained and evaluated separately but excluded from the final
system - its standalone model collapsed to 15.82% accuracy, 0.061 F1-macro,
likely due to data leakage from splitting an already-augmented dataset after
the fact, combined with severe class imbalance. See Section 6 for details.)

## 3. Methodology

### 3.1 Data Methodology
Primary dataset layout expected:
- DataSets/combined/train
- DataSets/combined/val
- DataSets/combined/test

Input preprocessing for model training and inference:
- Target image size: 224 x 224 x 3
- Pixel scaling: rescale = 1/255

Training-time augmentation values:
- rotation_range: 30
- width_shift_range: 0.2
- height_shift_range: 0.2
- shear_range: 0.1
- zoom_range: [0.8, 1.2]
- horizontal_flip: True
- vertical_flip: True
- brightness_range: [0.8, 1.2]
- channel_shift_range: 15.0
- fill_mode: nearest

### 3.2 Model Methodology
Hybrid architecture flow:
1. MobileNetV3-Small feature extractor
2. EfficientNetV2-B0 feature extractor
3. Feature concatenation
4. CBAM attention (ChannelAttention + SpatialAttention)
5. Dense classification head
5. Softmax output over 19 classes

Core model/training values:
- Input shape: (224, 224, 3)
- Num classes: 19
- Batch size: 32
- Total epochs: 100
- Initial training epochs: 25
- Fine-tune epochs: 75
- Initial learning rate: 1e-4
- Fine-tune learning rate: 1e-5
- Weight decay (AdamW): 1e-5
- Dropout rate: 0.4
- CBAM reduction ratio: 16
- MobileNet trainable layers during fine-tuning: 40
- EfficientNet trainable layers during fine-tuning: 30

Loss and metrics:
- Optimizer: AdamW
- Loss: CategoricalCrossentropy(label_smoothing=0.1)
- Metrics: accuracy, precision, recall

Callbacks values:
- EarlyStopping:
  - monitor: val_loss
  - patience: 15
  - restore_best_weights: True
- ModelCheckpoint:
  - monitor: val_accuracy
  - save_best_only: True
- ReduceLROnPlateau:
  - monitor: val_loss
  - factor: 0.5
  - patience: 5
  - min_lr: 1e-7
- TensorBoard enabled with histogram_freq: 1

### 3.3 Inference Methodology
Inference service behavior:
1. Validate uploaded image quality using classical CV checks
2. Preprocess image to 224 x 224, RGB, float32, rescale by 1/255
3. Run prediction with loaded Keras or TFLite model
4. Map 26-class prediction to requested crop class range
5. Detect cross-crop mismatch and reject if strong mismatch found
6. Derive severity
7. Fetch treatment recommendations
8. Generate Grad-CAM heatmap (or fallback placeholder)
9. Return structured JSON response

Crop ranges used in mapping:
- chilli: [0, 7]
- tomato: [8, 18]

Image validation thresholds:
- green_ratio must be >= 0.15
- entropy must be >= 4.0
- edge_ratio must be >= 0.01 and <= 0.40

Cross-crop mismatch rule:
- If global best class belongs to another crop and
- global_best_confidence > 3 x selected_crop_confidence
- then request is rejected as crop mismatch

Model file loading order:
1. models/agrilite_hybrid.keras
2. models/agrilite_hybrid.h5
3. models/agrilite_hybrid.tflite
4. fallback to placeholder prediction if none load

## 4. Backend Technical Stack and Values

### 4.1 Core Dependencies
Key backend dependencies:
- Flask==3.0.0
- Flask-SQLAlchemy==3.1.1
- Flask-Migrate==4.0.5
- Flask-JWT-Extended==4.6.0
- Flask-Bcrypt==1.0.1
- Flask-CORS==4.0.0
- Flask-Limiter==3.5.0
- tensorflow>=2.16.0
- numpy>=1.26.0
- Pillow>=10.2.0
- opencv-python-headless>=4.9.0

### 4.2 Configuration Defaults
Environment/config values from implementation:
- FLASK_ENV default: development
- SQLALCHEMY_DATABASE_URI fallback: sqlite:///agrilite.db
- Development DB fallback: sqlite:///agrilite_dev.db
- JWT access token expiry: 30 days
- JWT refresh token expiry: 60 days
- Default rate limit: 200 per hour
- Development rate limit: 1000 per hour
- Production rate limit: 100 per hour
- Max upload size: 16 MB (16 x 1024 x 1024)
- Allowed upload extensions: png, jpg, jpeg, gif, webp
- Rate limit strategy: fixed-window

Connection pool values:
- pool_size: 10
- pool_recycle: 3600
- pool_pre_ping: True
- max_overflow: 20
- pool_timeout: 30

### 4.3 API Blueprint Prefixes
- /api/auth
- /api/crops
- /api/predict
- /api/history
- /api/admin
- /health

### 4.4 Route-Level Rate Limits (Samples)
- POST /api/predict/: 30 per minute
- POST /api/predict/base64: 30 per minute
- GET /api/predict/treatment: 100 per minute
- POST /api/auth/register: 5 per minute
- POST /api/auth/login: 10 per minute
- GET /api/history/: 60 per minute

## 5. Frontend Technical Stack and Values

Core frontend dependencies:
- react ^18.3.1
- react-dom ^18.3.1
- react-router-dom ^7.9.4
- axios ^1.12.2
- react-hot-toast ^2.6.0
- lucide-react ^0.344.0
- html2canvas ^1.4.1
- vite ^5.4.2
- typescript ^5.5.3
- tailwindcss ^3.4.1

Frontend environment values:
- VITE_API_BASE_URL=http://localhost:5000
- VITE_APP_NAME=AgriLite-Hybrid

## 6. Data Model Technical Reference

### 6.1 User Model
Main fields:
- id (PK)
- email (unique, indexed)
- password_hash
- first_name, last_name, phone
- is_active (default True)
- role (default user)
- created_at, updated_at, last_login

### 6.2 DetectionHistory Model
Main fields:
- id (PK)
- user_id (nullable FK to users)
- crop
- disease
- severity
- confidence (float)
- treatment (JSON string in text)
- original_image, heatmap_image (text)
- temperature, humidity, environmental_risk
- device_type (default web)
- location, notes
- created_at

Indexes:
- idx_history_user_created(user_id, created_at)
- idx_history_crop_disease(crop, disease)

## 7. API Samples

### 7.1 Register User
Request:
{
  "email": "farmer@example.com",
  "password": "SecurePass123",
  "first_name": "Ravi",
  "last_name": "Kumar",
  "phone": "+91-9999999999"
}

Success response (201):
{
  "success": true,
  "message": "Registration successful",
  "access_token": "<jwt_access_token>",
  "refresh_token": "<jwt_refresh_token>",
  "user": {
    "id": 1,
    "email": "farmer@example.com",
    "first_name": "Ravi",
    "last_name": "Kumar",
    "role": "user",
    "is_active": true
  }
}

### 7.2 Login
Request:
{
  "email": "farmer@example.com",
  "password": "SecurePass123"
}

Success response (200):
{
  "success": true,
  "message": "Login successful",
  "access_token": "<jwt_access_token>",
  "refresh_token": "<jwt_refresh_token>",
  "user": {
    "id": 1,
    "email": "farmer@example.com"
  }
}

### 7.3 Predict (Multipart)
cURL sample:
curl -X POST http://localhost:5000/api/predict/ \
  -F "crop=tomato" \
  -F "image=@sample_leaf.jpg" \
  -F "temperature=29.5" \
  -F "humidity=72.0"

Success response (200) sample:
{
  "success": true,
  "data": {
    "crop": "tomato",
    "disease": "Early Blight",
    "severity": "medium",
    "confidence": 0.9321,
    "treatment": [
      {
        "type": "chemical",
        "name": "Apply copper-based fungicide every 7-10 days",
        "dosage": "",
        "application": "",
        "schedule": ""
      }
    ],
    "heatmap_base64": "data:image/jpeg;base64,<...>",
    "original_image_base64": "data:image/jpeg;base64,<...>",
    "environmental_risk": "normal",
    "model_used": "keras"
  }
}

Failure response (400) sample for crop mismatch:
{
  "success": false,
  "error": "This image appears to be a chilli leaf, not tomato. Please select the correct crop type or upload a tomato leaf image."
}

### 7.4 Predict (Base64)
Request:
{
  "crop": "tomato",
  "image_base64": "data:image/jpeg;base64,<...>",
  "temperature": 28.0,
  "humidity": 65.0
}

### 7.5 Save History
Request:
{
  "crop": "tomato",
  "disease": "Late Blight",
  "severity": "severe",
  "confidence": 0.8812,
  "treatment": [
    "Remove infected leaves",
    "Apply recommended fungicide"
  ],
  "temperature": 26.4,
  "humidity": 84.0,
  "environmental_risk": "high",
  "device_type": "web",
  "notes": "Observed after rainfall"
}

## 8. Class Label and Treatment Mapping Samples

Model metadata values:
- model: AgriLite-Hybrid
- num_classes: 19
- severity_levels: mild, medium, severe

Class mapping samples:
- chilli_Chilli_Leaf_Curl_Virus -> Leaf Curl Virus
- tomato_Septoria_leaf_spot -> Septoria Leaf Spot

Treatment sample for Leaf Curl Virus (severe):
- Remove all infected plants immediately
- Treat entire field for whitefly vectors
- Plant only resistant varieties (Ty gene)
- Install insect-proof nets for nursery

## 9. Deployment and Runtime Values

### 9.1 Docker Compose Values
Services and mapped ports:
- db: 5432:5432
- backend: 5000:5000
- frontend: 5173:5173

Database container values:
- POSTGRES_USER=agrilite
- POSTGRES_PASSWORD=agrilite_dev_password
- POSTGRES_DB=agrilite

Backend container environment sample:
- FLASK_ENV=development
- DATABASE_URL=postgresql://agrilite:agrilite_dev_password@db:5432/agrilite
- SECRET_KEY=dev-secret-key-change-in-production
- JWT_SECRET_KEY=dev-jwt-secret-change-in-production

### 9.2 Local Environment Templates
Backend .env example key values:
- FLASK_ENV=development
- FLASK_APP=app.py
- DATABASE_URL=sqlite:///agrilite_dev.db
- RATELIMIT_DEFAULT=200 per hour
- CORS_ORIGINS=http://localhost:5173,http://localhost:3000
- PORT=5000
- HOST=0.0.0.0

Frontend .env example key values:
- VITE_API_BASE_URL=http://localhost:5000
- VITE_APP_NAME=AgriLite-Hybrid

## 10. Evaluation Methodology and Benchmarks
Evaluation script behavior:
- Loads trained hybrid model
- Runs predictions on DataSets/combined/test
- Computes overall metrics: accuracy, f1_macro, f1_weighted
- Generates classification reports and confusion matrices
- Computes per-crop metrics using fixed crop index ranges
- Builds standalone vs hybrid comparison JSON

Embedded standalone baseline samples:
- Chilli standalone accuracy: 0.9292
- Tomato standalone accuracy: 0.9634

(Brinjal standalone accuracy was 0.1582 - excluded from the final system;
see Section 6 and the Limitations note above.)

## 11. Reproducibility Checklist
- Set Python, TensorFlow, and Node versions according to requirements
- Use the same input resolution: 224 x 224
- Keep rescale and augmentation values unchanged
- Preserve class index ordering from class_labels.json
- Use same crop class ranges for inference filtering
- Keep optimizer and loss settings unchanged during retraining
- Validate model with evaluate_hybrid.py on combined test set

## 12. Quick Run Commands
Backend:
- cd backend
- pip install -r requirements.txt
- python app.py

Frontend:
- cd frontend
- npm install
- npm run dev

Training:
- cd scripts
- python train_hybrid.py

Evaluation:
- cd scripts
- python evaluate_hybrid.py
