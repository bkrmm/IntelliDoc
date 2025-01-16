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

"""
ORIGINAL TEXT:
Abstract Sentiment Analysis (SA) is an ongoing field of research in text mining field. SA is the computational treatment of opinions, sentiments and subjectivity of text_ This survey paper tackles comprehensive overview of the last update in this field. recently proposed algorithms' enhancements and various SA applications are investigated and presented briefly in this survey: These articles are categorized according to their contributions in the various SA techniques_ The related fields to SA (transfer learning; emotion detection; and building resources) that attracted researchers recently are discussed. The main target of this survey is to give nearly full image of SA techniques and the related fields with brief details. The main contributions of this paper include the sophisticated categorizations of number of recent articles and the illustration of the recent trend of research in the sentiment analysis and its related areas_ 2014 Production and hosting by Elsevier BV on behalf of A

SUMMARIZED TEXT:
Abstract Sentiment Analysis (SA) is an ongoing field of research in text mining field . SA is the computational treatment of opinions, sentiments and subjectivity of text . This survey paper tackles comprehensive overview of last update in this field .
"""
