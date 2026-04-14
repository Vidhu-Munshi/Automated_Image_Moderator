# AIM - Automated Image Moderator

An AI-powered image moderation system that detects **human emotions** and **unsafe content (weapons, violence, etc.)** from images using Deep Learning.

Built using **YOLO + CNN + FastAPI**, this project provides a complete pipeline for **content filtering, safety detection, and intelligent moderation**.

---

## Features

*  Emotion Detection

  * Happy, Sad, Angry, Fear, Disgust, Surprise, Neutral, Contempt

*  Unsafe Content Detection

  * Gun, Knife, Blood, Violence, Drugs, Alcohol, Cigarette, Nudity, Gore

*  Smart Moderation System

  * Automatically blocks unsafe or restricted content

*  Detailed Output

  * Prediction label
  * Confidence score
  * Full probability distribution

*  Dynamic Filter System (Frontend)

  * Enable/Disable emotions & objects manually

*  Interactive Web UI

  * Drag & Drop Image Upload
  * Real-time Preview
  * Visual Result Display

*  Fast Backend API

  * Built with FastAPI

---

## Model Architecture

### Emotion Detection

* CNN model trained on FER2013 dataset
* Input size: **48x48 RGB**
* Output: Emotion probabilities

### Object Detection (Moderation Logic)

* YOLO-based dataset usage
* Keyword-based filtering for unsafe classes

### Accuracy

*  Overall Accuracy: **72%**

---

## Datasets Used

### Weapons Detection

* https://www.kaggle.com/datasets/issaisasank/guns-object-detection
* https://www.kaggle.com/datasets/shank885/knife-dataset

### Emotion Detection

* https://www.kaggle.com/datasets/msambare/fer2013

---

## 🛠️ Tech Stack

### Backend

* Python
* FastAPI
* TensorFlow / Keras
* NumPy
* OpenCV

### Frontend

* HTML
* CSS
* JavaScript

---

## 📦 Requirements

Install all dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies Used

* fastapi
* uvicorn
* tensorflow
* pillow
* numpy
* python-multipart
* opencv-python
* scikit-learn

---

##  Project Structure

```
├── app.py
├── emotion_best.keras
├── emotion_labels.json
├── requirements.txt
├── templates (optional)
├── static (optional)
└── README.md
```

---

##  How to Run

### Step 1: Clone Repository

```bash
git clone https://github.com/your-username/AIM-Image-Moderator.git
cd AIM-Image-Moderator
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run Server

```bash
python app.py
```

### Step 4: Open Browser

```
http://localhost:8000
```

---

## 📡 API Documentation

### POST `/predict`

Upload an image and receive prediction:

```json
{
  "status": "success",
  "prediction": "happy",
  "confidence": 0.87,
  "all_probs": {
    "happy": 0.87,
    "sad": 0.05
  },
  "detected_objects": []
}
```

---

##  Working Pipeline

1. Image uploaded by user
2. Image resized to **48x48**
3. Normalization applied
4. Passed into trained CNN model
5. Emotion predicted
6. Checked against unsafe object keywords
7. Output:

   *  Allowed
   *  Blocked

---

##  Frontend Features

* Drag & Drop Upload
* Live Image Preview
* Toggle Filters (Emotions & Objects)
* Confidence Bars Visualization
* Blocked Content Warning UI

---

##  Use Cases

* Social Media Moderation
* Content Filtering Platforms
* AI Safety Systems
* Upload Validation Tools
* Educational AI Projects

---

## Limitations

* Accuracy limited to **72%**
* YOLO not fully integrated in real-time pipeline
* Possible false positives/negatives
* Not suitable for production-level moderation

---

##  Future Improvements

* Improve accuracy (>85%)
* Integrate YOLOv8 fully
* Add real-time video moderation
* Deploy on cloud (AWS / GCP)
* Add multi-model ensemble

---

##  Author

**Vidhu Anand Munshi**

 LinkedIn:
https://www.linkedin.com/in/vidhu-anand-munshi-a272b028a

---

## Note

This project is developed for **learning, experimentation, and demonstration purposes**.

---

## Support

If you like this project:

* Star ⭐ the repo
* Share it
* Improve it

---
