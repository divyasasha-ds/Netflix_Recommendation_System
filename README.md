
# 🎬 Netflix Recommendation System

## 📌 Project Overview
This project is a Netflix Movie Recommendation System developed using Machine Learning (KNN) and Deep Learning (Autoencoder).  
The objective is to understand how recommendation systems work and to compare ML and DL models based on performance metrics.


---

## 🚀 Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- TensorFlow / Keras  
- Matplotlib  

---

## 📊 Dataset Information
- Dataset size: 500 rows  
- Dataset type: Self-created (not taken from Google or Kaggle)  
- Number of users: 99  
- Number of movies: 50  

### Columns
- UserID  
- MovieID  
- Rating (1–5)  
- Genre  
- ReleaseYear  
- Timestamp  

---

## 🧠 Models Implemented

### Machine Learning
- K-Nearest Neighbors (KNN)
- Collaborative Filtering

### Deep Learning
- Autoencoder Neural Network

---

## 📈 Model Performance

| Model | RMSE | Training Time |
|------|------|---------------|
| ML (KNN) | 3.2278 | 0.20 sec |
| DL (Autoencoder) | 3.2752 | 5.45 sec |

---

## 🖥 Sample Output

- Dataset loaded successfully  
- Shape: (500, 6)  
- Users: 99  
- Movies: 50  

### Top-5 Movie Recommendations

**ML (KNN)**
- User 1: [38, 43, 20, 30, 42]  
- User 2: [44, 25, 37, 10, 18]  
- User 3: [4, 15, 25, 8, 36]  

**DL (Autoencoder)**
- User 1: [43, 35, 24, 38, 23]  
- User 2: [25, 44, 37, 10, 32]  
- User 3: [4, 15, 25, 5, 48]  

---

## 🎯 Key Features
- Netflix-style recommendation system  
- ML vs DL comparison  
- RMSE evaluation  
- Training time comparison  
- Top-N movie recommendations  
- Genre-based analysis  
- Data visualization

---

## ▶️ How to Run the Project

```bash
pip install -r requirements.txt
python Netflix_Recommendation_System.py

👩‍💻 Author

C. Divya Sasha
BSc Computer Science (2025)
