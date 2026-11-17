# Semantic Segmentation System for Autonomous Robot Environment Perception

A computer vision system for semantic understanding of the environment of an autonomous robot. The project combines semantic segmentation, depth estimation, uncertainty estimation, and scene analysis to support environment perception and navigation-related tasks.

## Overview

The system processes images and generates pixel-level semantic predictions, helping an autonomous robot identify and interpret elements of its surroundings.

The application provides a web-based interface for image analysis, dataset management, model evaluation, and visualization of prediction results.

## Key Features

### Semantic Segmentation
- Pixel-level classification of images using **SegFormer-B3**.
- Pretrained model: `nvidia/segformer-b3-finetuned-cityscapes-1024-1024`.
- 19 semantic classes from the Cityscapes dataset.
- Segmentation mask and overlay visualization.
- Pixel-level confidence estimation.
- Confidence maps and downloadable prediction results.

### Environment Analysis
- Traversability estimation.
- Obstacle identification and risk assessment.
- Risk Map generation.
- Scene Analysis for interpreting the spatial arrangement of objects.
- Safe Corridor Detection for identifying potentially safe movement areas.
- Monocular depth estimation.
- Uncertainty estimation for segmentation predictions.

### Dataset Management
- Image upload and dataset organization.
- Dataset splits for training, validation, and testing.
- Image and prediction management.
- Dataset access control based on user roles.

### Model Evaluation and Training
- SegFormer-B3 evaluation on dataset splits.
- Model performance analysis.
- Training and fine-tuning support.
- Evaluation scripts for quantitative experiments.

### User Management and Security
- User registration and authentication.
- JWT-based authorization.
- Role-based access control (RBAC).
- Three roles:
  - **User** — access to personal uploaded images and their results.
  - **Technician** — dataset and analysis management permissions.
  - **Administrator** — administrative access and user management.

## Technology Stack

| Component | Technologies |
|---|---|
| Backend | Python, FastAPI |
| Machine Learning | PyTorch, Transformers |
| Segmentation Model | SegFormer-B3 |
| Image Processing | OpenCV, NumPy |
| Database | SQLite, SQLAlchemy |
| Authentication | JWT, pwdlib |
| Frontend | HTML, CSS, JavaScript |
| API | REST |

## Architecture

The application follows a client-server architecture.

- **Frontend:** provides the user interface for image uploads, visualization, dataset management, and analysis.
- **Backend:** handles image processing, model inference, database operations, authentication, and API requests.
- **Machine Learning:** performs semantic segmentation and additional environment perception tasks.
- **Database:** stores user accounts, roles, datasets, image metadata, predictions, and detected classes.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/villenuevve/MastersDiploma.git
cd MastersDiploma/MastersDiploma
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the backend

```bash
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

The backend API will be available at:

`http://127.0.0.1:8000`

### 5. Start the frontend

Open another terminal:

```bash
cd frontend
python3 -m http.server 5500
```

Open the application at:

`http://127.0.0.1:5500`

## API Documentation

FastAPI provides interactive API documentation:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Project Structure

```text
MastersDiploma/
├── backend/
│   ├── main.py
│   ├── models.py
│   ├── auth.py
│   ├── security.py
│   ├── risk_map.py
│   ├── init_db.py
│   ├── analyze_all_images.py
│   ├── train_segformer_b3.py
│   └── evaluate_segformer_b3_splits.py
│
├── frontend/
│   ├── index.html
│   └── js/
│       ├── app.js
│       ├── segmentationView.js
│       ├── segmentationModel.js
│       ├── rolePanels.js
│       ├── inspector.js
│       ├── obstacleView.js
│       ├── analytics.js
│       ├── history.js
│       └── language-switch.js
│
├── data/
│   └── segmentation.db
│
└── README.md
```

## Experimental Evaluation

The project includes experiments focused on semantic segmentation for autonomous robot environment perception.

The experimental workflow includes:

- Evaluation of SegFormer models.
- Analysis of segmentation predictions on Cityscapes and additional images.
- Dataset split management.
- Quantitative model evaluation.
- Analysis of prediction confidence and uncertainty.
- Investigation of segmentation results for navigation-related tasks.

## Hardware Considerations

The system uses PyTorch and supports hardware acceleration when available.

- NVIDIA GPU acceleration through CUDA, when available.
- Apple Silicon acceleration through PyTorch MPS, when supported.
- CPU execution as an alternative.

Inference performance depends on the hardware, input resolution, and model configuration.

## Project Goal

The goal of this project is to develop an integrated computer vision system that enables an autonomous robot to perceive and interpret its surroundings through semantic segmentation and complementary visual analysis methods.

The system is developed as part of a Master's diploma project.
