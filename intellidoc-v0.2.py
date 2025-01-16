!pip install easyocr
!pip install pytesseract transformers Pillow

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
preimg2 = " ".join([text[1] for text in eximg2])
preimg1 = " ".join([text[1] for text in eximg1])

#for detection in eximg2:
 #   print(f"Text: {detection[1]}, Confidence: {detection[2]:.2f}")
#Summarize Function
def summarize(preimg2):
    try:
        summarizer = pipeline("summarization")
        summary = summarizer(preimg2, max_length=170, min_length=5, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return str(e)

#Sentiment Analysis Function
def sentiment_analysis(preimg1):
    try:
        sentiment_analyzer = pipeline("sentiment-analysis")
        sentiment = sentiment_analyzer(preimg1)[0]
        return sentiment['label'], sentiment['score']
    except Exception as e:
        return str(e)

result_sa = sentiment_analysis(preimg2)
  result_s = summarize(preimg2)

#User Input
user_input = input("Enter a sentence, enter (s) for summarization or (sa) for SA: ")
if user_input == "s":
  print("Summarized Test: ",result_s)
elif user_input == "sa":
  print("Sentiment Analysis Score: ",result_sa)
else:
  print("Invalid input. Please enter 's' for summarization or 'sa' for sentiment analysis.")

