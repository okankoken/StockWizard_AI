#!/usr/bin/env bash

python fetch/prices.py
python fetch/compute.py
streamlit run ui/app.py
