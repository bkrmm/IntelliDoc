# -*- coding: utf-8 -*-
"""IntelliDoc.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1PTvIN8HtOMN4c3w-Cu4Zg6fGtPQTeDUd
"""

!pip install easyocr

import tensorflow
import pytesseract as pyt
import easyocr as eo
import numpy as np
import pandas as pd
from PIL import Image
from transformers import pipeline

#files
img1 = Image.open("/content/image.png")
img2 = Image.open("/content/img2.png")

#OCR
reader = eo.Reader(['en'])
eximg1 = reader.readtext("/content/image.png")
eximg2 = reader.readtext("/content/img2.png")

for detection in eximg2:
    print(f"Text: {detection[1]}, Confidence: {detection[2]:.2f}")

preimg1 = " ".join([text[1] for text in eximg1])
preimg1

#NLP
from transformers import pipeline
def summarize(preimg1):
    try:
        summarizer = pipeline("summarization")
        summary = summarizer(preimg1, max_length=54, min_length=5, do_sample=False)
        return summary[0]['summary_text']
    except:
        return "Invalid Image"

try:
  result = summarize(preimg1)
  result
  print(result)
except:
  print("Invalid Image")