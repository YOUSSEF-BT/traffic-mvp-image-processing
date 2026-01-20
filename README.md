# Traffic MVP — Vision par ordinateur & IA (Traitement d’images)

Projet de fin de cours (Traitement d’images) : détection et analyse du trafic routier à partir d’une vidéo, avec génération de métriques et visualisation via un dashboard Streamlit.

## 🎯 Objectif
- Détecter les véhicules (voitures / motos / bus / camions) sur une vidéo
- Suivre l’évolution du trafic (densité, comptage total, score de congestion)
- Exporter les résultats dans un fichier `metrics.csv`
- Visualiser les métriques dans un dashboard web (Streamlit)

## 🧠 Fonctionnalités
- Détection d’objets en temps réel (YOLOv8 / Ultralytics)
- Superposition des bounding boxes sur la vidéo
- Export CSV : `t_video_s, vehicles_in_frame, count_total, avg_speed_kmh, congestion_score, ...`
- Dashboard Streamlit : tableau + courbes

## 🧰 Technologies utilisées
- Python 3
- OpenCV
- Ultralytics (YOLOv8)
- Streamlit
- Pandas / Numpy
- FFmpeg (conversion vidéo)

## 📁 Structure du projet
- `traffic_mvp.py` : script principal (détection + export CSV)
- `dashboard.py` : dashboard Streamlit (lecture du CSV + visualisation)
- `config.yaml` : configuration (classes, seuils, etc.)
- `presentation/` : slides de présentation (PDF/PPT)

## ✅ Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 🎬 Préparer la vidéo (si .MOV / HEVC)

Exemple de conversion en H264 compatible OpenCV :

ffmpeg -y -i "IMG_5473.MOV" -c:v libx264 -pix_fmt yuv420p -movflags +faststart "IMG_5473_h264.mp4"

## ▶️ Lancer la détection + générer le CSV
source .venv/bin/activate
python traffic_mvp.py --source "/chemin/video.mp4" --show --csv metrics.csv --conf 0.50

## 📊 Lancer le dashboard

⚠️ Toujours lancer Streamlit via le python de l’environnement :

source .venv/bin/activate
python -m streamlit run dashboard.py

## 🧪 Démo rapide (preuve)

Fenêtre OpenCV : détection en direct

metrics.csv : fichier généré automatiquement

Dashboard : affichage des dernières mesures + courbes

## 📌 Limites

La vitesse en km/h est une estimation (sans calibration réelle caméra → mètres)

Les faux positifs peuvent apparaître selon l’angle et la qualité vidéo

## 🚀 Perspectives

Calibration caméra (homographie / mètres par pixel)

Suivi multi-objets (DeepSORT / ByteTrack)

Comptage précis par ligne/zone (ROI)

Amélioration du filtrage (confiance, taille min, classes)

👤 Auteur

Youssef BT
