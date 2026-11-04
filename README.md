# AgriLite-Hybrid 🌱

## Multi-Crop Plant Disease Detection Web + Edge System

A production-ready plant disease detection system using a hybrid deep learning model (MobileNetV3 + EfficientNetV2-B0 with CBAM attention) for accurate disease identification in **Tomato and Chilli** crops.

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![React](https://img.shields.io/badge/react-18.3.1-61DAFB)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.16+-orange)

---

## 🎯 Features

- **Multi-Crop Support**: Detect diseases in Tomato and Chilli
  (Brinjal was evaluated but excluded - its standalone model collapsed to
  15.8% accuracy from data leakage/class imbalance; see Limitations below)
- **Hybrid AI Model**: MobileNetV3 + EfficientNetV2-B0 fused with CBAM attention
- **Severity Assessment**: Classifies disease severity (Mild / Medium / Severe)
- **Grad-CAM Heatmaps**: Visual explanation of model predictions
- **Treatment Recommendations**: Actionable treatment advice for each disease
- **Edge Deployment**: Raspberry Pi compatible with TensorFlow Lite
- **Environmental Monitoring**: Optional DHT22 sensor integration for risk alerts
- **User Authentication**: Secure login with JWT tokens
- **Detection History**: Save and review past diagnoses

---

## 📁 Project Structure

```
agri-lite-hybrid/
├── backend/                  # Flask REST API server
│   ├── routes/               # API blueprints
│   ├── services/             # Business logic & ML inference
│   ├── migrations/           # Database migrations (Alembic)
│   ├── static/uploads/       # Uploaded images
│   ├── app.py                # Application factory
│   ├── config.py             # Configuration management
│   ├── extensions.py         # Flask extensions
│   ├── models.py             # Database models
│   └── requirements.txt      # Python dependencies
│
├── frontend/                 # React + Vite + TypeScript SPA
│   ├── src/
│   │   ├── components/       # Reusable UI components
│   │   ├── pages/            # Page components
│   │   ├── contexts/         # React Context providers
│   │   ├── services/         # API service layer
│   │   ├── hooks/            # Custom React hooks
│   │   └── types/            # TypeScript definitions
│   ├── package.json
│   └── vite.config.ts
│
├── models/                   # Trained ML models (.h5, .tflite)
├── scripts/                  # Deployment & utility scripts
│   └── pi_deploy/            # Raspberry Pi edge deployment
├── shared/                   # Shared constants & types
├── docker-compose.yml        # Local development setup
└── README.md                 # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL (or SQLite for development)
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/agri-lite-hybrid.git
cd agri-lite-hybrid
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env
# Edit .env with your configuration

# Initialize database
flask db init
flask db migrate -m "Initial migration"
flask db upgrade

# Run development server
python app.py
```

Backend will be available at: `http://localhost:5000`

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Create environment file
cp .env.example .env
# Edit .env with your API URL

# Run development server
npm run dev
```

Frontend will be available at: `http://localhost:5173`

### 4. Docker Setup (Alternative)

```bash
# Build and run all services
docker-compose up --build

# Stop services
docker-compose down
```

---

## 🔌 API Endpoints

### Public Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/crops` | List all supported crops |
| POST | `/api/predict` | Analyze image for disease detection |
| GET | `/health` | Health check endpoint |

### Protected Endpoints (JWT Required)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register` | Register new user |
| POST | `/api/auth/login` | User login |
| GET | `/api/auth/profile` | Get user profile |
| POST | `/api/history` | Save detection result |
| GET | `/api/history` | Get user's detection history |

### Prediction Request Example

```bash
curl -X POST http://localhost:5000/api/predict \
  -F "crop=tomato" \
  -F "image=@path/to/image.jpg"
```

### Prediction Response

```json
{
  "success": true,
  "data": {
    "crop": "tomato",
    "disease": "Cercospora Leaf Spot",
    "severity": "medium",
    "confidence": 98.7,
    "treatment": [
      "Remove infected leaves immediately",
      "Apply copper-based fungicide",
      "Ensure proper plant spacing for air circulation",
      "Water at the base, avoid wetting leaves"
    ],
    "heatmap_base64": "data:image/jpeg;base64,...",
    "original_image_base64": "data:image/jpeg;base64,..."
  }
}
```

---

## 🌿 Supported Crops & Diseases

### Tomato
- Early Blight
- Late Blight
- Bacterial Spot
- Septoria Leaf Spot
- Healthy

### Chilli
- Leaf Curl Virus
- Powdery Mildew
- Anthracnose
- Bacterial Leaf Spot
- Healthy

### Excluded: Brinjal (Eggplant)
Brinjal was trained and evaluated but excluded from the final system - its
standalone model collapsed to 15.82% accuracy (predicting "Healthy" for
nearly all inputs), likely due to data leakage from splitting an
already-augmented dataset after the fact. See
METHODOLOGY_AND_TECHNICAL_REFERENCE.md for details and future work.

---

## 🤖 Model Architecture

The AgriLite-Hybrid model combines:

1. **MobileNetV3-Small**: Lightweight feature extraction
2. **EfficientNetV2-B0**: Accurate feature extraction
3. **CBAM Attention**: Channel and Spatial attention mechanism
4. **Fusion Layer**: Concatenation of both streams
5. **Classification Head**: Dense layers for disease + severity prediction

**Input**: 224×224 RGB image
**Output**: Disease class + Severity level + Confidence score

---

## 🍇 Raspberry Pi Deployment

For edge deployment on Raspberry Pi:

```bash
cd scripts/pi_deploy

# Follow the setup guide
cat README.md

# Install dependencies
pip install -r requirements_pi.txt

# Run inference server
python web_server_pi.py
```

Features:
- TensorFlow Lite optimized inference
- DHT22 temperature/humidity sensor support
- Local network web interface
- Offline operation capability

---

## 🧪 Testing

### Backend Tests

```bash
cd backend
pytest tests/ -v
```

### Frontend Tests

```bash
cd frontend
npm run test
```

---

## 📝 Environment Variables

### Backend (.env)

```env
FLASK_ENV=development
SECRET_KEY=your-secret-key
JWT_SECRET_KEY=your-jwt-secret
DATABASE_URL=postgresql://user:pass@localhost:5432/agrilite
# For development: DATABASE_URL=sqlite:///agrilite.db
SUPABASE_URL=your-supabase-url
SUPABASE_KEY=your-supabase-key
```

### Frontend (.env)

```env
VITE_API_BASE_URL=http://localhost:5000
VITE_APP_NAME=AgriLite-Hybrid
```

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Your Name**
- Final Year Research Project
- University Name
- Email: your.email@university.edu

---

## 🙏 Acknowledgements

- TensorFlow Team for the deep learning framework
- Flask Team for the web framework
- React Team for the frontend library
- Research advisors and mentors

---

**Version**: 1.0.0
**Last Updated**: February 2026
