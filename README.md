# K-League Set-Piece Danger Overlay

Entry for **Track 2 of the K-League data competition (Dacon, Jan 2026)**.

The idea: make set-piece broadcasts smarter. An XGBoost-based **VAEP** model scores every action in K-League event data, set-piece danger zones are rendered as heatmaps, and **OpenCV homography** projects those heatmaps onto real broadcast footage — wrapped in an MVP dashboard that lets a viewer toggle the AI overlay on and off during playback.

```
K-League event CSV → SPADL → XGBoost (P_score / P_concede) → VAEP per action
      → set-piece danger heatmaps → homography overlay on broadcast video → MVP dashboard
```

## Repository Layout

```text
k-track2/
├── 히트맵/                          # "heatmap" — modeling & visualization
│   └── 히트맵.ipynb                 #   Colab notebook: SPADL → VAEP → set-piece heatmaps
├── 영상(기준점)/                    # "video (reference points)" — broadcast overlay scripts
│   ├── 코너킥.py                    #   corner kick: heatmap projected onto the pitch
│   ├── 스로인 및 프리킥(크로스).py  #   long throw-in & free-kick cross: heatmap projection
│   └── 프리킥(슛).py                #   direct free kick: Near/Center/Far-post VAEP course cards
└── mvp/
    ├── index.html                   # dashboard MVP (raw ↔ AI-overlay toggle, 6 categories)
    ├── alert_raw.mp4                # demo: cross-match "switch broadcast" alert (raw clip)
    └── alert_final.mp4              # demo: same clip with the alert overlay
```

## 1. VAEP Model & Heatmaps — `히트맵/히트맵.ipynb`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mself8/k-track2/blob/main/%ED%9E%88%ED%8A%B8%EB%A7%B5/%ED%9E%88%ED%8A%B8%EB%A7%B5.ipynb)

Runs on Google Colab with the competition data mounted from Google Drive.

- **Input**: the competition-provided K-League raw event CSV (`raw_data.csv`, not redistributed here — adjust `file_path` in the notebook to your own Drive path).
- **Preprocessing**: raw events are mapped into the [SPADL](https://socceraction.readthedocs.io/) action format via `socceraction`.
- **Model**: two XGBoost classifiers predict scoring / conceding probability (`P_score`, `P_concede`) with a strict labeling window (only the few actions immediately preceding a goal count), then the VAEP formula turns probability deltas into per-action values.
- **Visualizations**:
  - corner kick landing-zone analysis with strict position filtering, plus a high-resolution (25×20) landing heatmap
  - free-kick 3-zone danger plot and location-parameterized free-kick / long throw-in heatmaps (`plot_heatmap_at_location`)
  - tactics-board style corner kick trajectory arrows weighted by real VAEP values
  - dark-background variants that export the overlay PNGs (e.g. `heatmap_overlay.png`, `heatmap_throwin.png`) consumed by the video scripts below

## 2. Broadcast Overlay — `영상(기준점)/`

Each script overlays analysis graphics onto a broadcast clip using a **4-point homography calibration**:

1. A window shows the heatmap image — click the 4 reference points (e.g. penalty-box corners).
2. A second window shows a video frame — click the same 4 points on the pitch.
3. The script computes the homography, warps the heatmap onto the pitch surface, and writes the result video.

Robustness touches: a green-pitch HSV mask keeps the overlay on grass only (players/logos stay clean), broadcast top/bottom bands and the station logo area are ignored, camera motion is smoothed (`SMOOTH_FACTOR`), and the overlay fades out just before the kick (`FADE_SPEED`) to clear the viewer's sight line.

| Script | Set piece | Overlay |
|---|---|---|
| `코너킥.py` | Corner kick | Danger heatmap projected on the pitch, fade-out before the kick |
| `스로인 및 프리킥(크로스).py` | Long throw-in / free-kick cross | Landing-zone ("red zone") heatmap projection |
| `프리킥(슛).py` | Direct free kick | HUD-style cards per shot course (Near Post / Center / Far Post) showing attempt counts and VAEP values, read from `freekick_stats.json` |

Edit the `VIDEO_FILE` / `HEATMAP_FILE` / `SAVE_FILE` constants at the top of each script, then run with `python <script>.py` (requires `opencv-python`, `numpy`).

## 3. MVP Dashboard — `mvp/index.html`

A static, dark-themed "K-League Tactical Analysis Hub" page — open it directly in a browser.

- Six analysis categories: corner kick, free-kick cross, direct free kick, long throw-in, pass success probability, and a **cross-match broadcast-switch alert** (an xT-style "this other match just got interesting" toast).
- The **AI toggle** swaps between the raw clip and the AI-overlay clip while preserving the playback position, so the effect of the overlay is directly comparable.
- Only the alert demo videos are committed. The other categories expect `<category>_raw.mp4` / `<category>_final.mp4` files (produced by the overlay scripts) placed next to `index.html`.

## Requirements

- Notebook: `socceraction`, `matplotsoccer`, `pandera`, `statsbombpy`, `xgboost`, `numpy<2.0`, `pandas>=2.0` (installed in the first notebook cell)
- Overlay scripts: `opencv-python`, `numpy`

## Data

The K-League event data (`raw_data.csv`) was provided by the competition and is **not redistributed** in this repository.
