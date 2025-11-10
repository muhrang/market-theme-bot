name: Run Bot

on:
  schedule:
    - cron: "*/5 0-8 * * 1-5"   # 평일 09:00~15:00 한국시간 기준 5분마다
  workflow_dispatch:

jobs:
  run:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install System Dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y tesseract-ocr tesseract-ocr-kor libgl1

      - name: Install Python Packages
        run: |
          pip install --upgrade pip
          pip install -r requirements.txt
          pip install playwright
          python -m playwright install chromium

      - name: Run Bot
        run: python main.py
