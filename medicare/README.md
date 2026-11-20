# MediCare AI — Healthcare Recommendation System

MediCare AI is a Flask-based web application that uses machine learning models (`SVC`, `Random Forest`, `Gradient Boosting`) and `medicine_database.pkl` to estimate general health risk levels, surface safety warnings, and display top probable conditions from user-selected symptoms and patient details.

> **Disclaimer:** This project is for educational and informational purposes only and is not a`requirements.txt` verify ho gayi hai, toh ab `app.py` wale usi `medicare` folder ke andar `README.md` file banayein.

### `README.md` Ka Content

Is text ko copy karke `README.md` mein paste karein aur `Ctrl + S` se save karein:

```markdown
# MediCare AI

MediCare AI ek Flask-based healthcare risk and condition screening web application hai jo user ke symptoms aur basic health inputs ke basis par risk level aur top probable conditions ka estimate dikhati hai.

## Key Features

- **Interactive Web Interface:** Symptoms, age, gender, blood pressure, aur cholesterol inputs ke liye `templates/index.html` interface.
- **Flask Backend API:** `/predict` endpoint jo input data process karke structured JSON response return karta hai.
- **ML & Recommendation Pipeline:** Scikit-learn models (`SVC`, `Random Forest`, `Gradient Boosting`) aur `medicine_database.pkl` ke saath risk warnings aur general guidance generate karta hai.

## Project Structure

```text
medicare/
├── app.py
├── medicine_database.pkl
├── requirements.txt
├── README.md
└── templates/
    ├── index.html
    ├── about.html
    └── contact.html