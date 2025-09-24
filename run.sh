#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Base env'i kullan
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

# Ilk deneme: az sayida sembol ve küçük batch
export RUN_MAX_SYMBOLS="${RUN_MAX_SYMBOLS:-5}"   # ilk kosuda 5 sembolle dene
export RUN_BATCH="${RUN_BATCH:-3}"               # batch boyutu
export RUN_PAUSE="${RUN_PAUSE:-2.5}"             # batch arasi bekleme (sn)

python fetch/prices.py
python fetch/compute.py

streamlit run ui/app.py
