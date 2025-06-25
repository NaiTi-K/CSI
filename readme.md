# CSI_SOC_AI/ML – Environmental Risk Monitoring

🌍 This project uses **Remote Sensing** + **Machine Learning** to monitor land change and predict environmental risk zones.

## 🔗 Live Demo
👉 [Streamlit App](https://atw8vyxjpqxwh5fq4yhrqx.streamlit.app/)

---

## 📄 Final Report
You can read the full technical write-up and interpretation here:

👉 [📘 Download Report (PDF)](Final_report.pdf)


## 📌 Project Overview

- **Remote Sensing Analysis** (Task 1):
  - NDVI Change Map (2013–2023)
  - LULC Classification and Transition
  - VIIRS Nightlights (Urban Growth)

- **Risk Prediction Model** (Task 2):
  - Decision Tree trained on proxy data (forest loss, illness, mine distance)
  - Predicts High-Risk vs Low-Risk zones

---

## 📁 Folders

| Folder | Description |
|--------|-------------|
| `MI_model_decision_tree/` | Jupyter notebook to train Decision Tree |
| `app/` | Streamlit app source code |
| `assets/` | Model visualizations (heatmaps, tree, etc.) |
| `remote_sensing_images/` | NDVI, LULC, Transition, and VIIRS maps |
| `requirements.txt` | Python packages for running the app |

---

## 📦 How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
