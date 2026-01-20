# Traffic Project Zero — Traitement d'images (Sujet 1)

MVP complet **from scratch** :
- Détection + tracking véhicules (YOLOv8 via `ultralytics`)
- Comptage par ligne virtuelle
- Estimation vitesse (approx) + densité + score de congestion
- Sauvegarde `metrics.csv`
- Dashboard Streamlit (optionnel)

## 1) Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Convertir une vidéo DJI en H.264 (recommandé)
OpenCV lit mieux H.264 que H.265/HEVC.

```bash
ffmpeg -y \
  -i "/path/to/input.mp4" \
  -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
  "/path/to/output_h264.mp4"
```

## 3) Configurer la ligne de comptage
Éditer `config.yaml` :
- `count_line.p1` et `count_line.p2` = 2 points en pixels sur l'image (ligne à travers la voie)
- `pixels_per_meter` = calibration grossière (à ajuster)

## 4) Lancer le script
```bash
python traffic_mvp.py --source "/path/to/video_h264.mp4" --show --csv metrics.csv
```
- ESC pour quitter.

## 5) Dashboard (optionnel)
Dans un autre terminal :
```bash
streamlit run dashboard.py
```

## Notes
- Pour accélérer: `--fps 10`
- Pour webcam: `--source 0` (macOS: autoriser Caméra pour Terminal/Python)
