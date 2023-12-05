import os
import json
import time
import requests as rq
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import combinations
from glob import glob as gb, glob as gbz
from collections import Counter
from string import ascii_uppercase as afb
from scipy.stats import zscore
from scipy.stats import linregress as lr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from bs4 import BeautifulSoup as bs
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from scipy.optimize import curve_fit
from scipy.stats import linregress as lr
from pymannkendall import original_test as pmk
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
import fasttext

DATA_PATH = '/home/rb/Documents/ProjectData/antonym-data/' ## change to data path

VD_DATA_PATH = os.path.join(DATA_PATH,'vd-antonym-html')

PAIR_CHUNK_PATH = os.path.join(DATA_PATH,'pair-chunks')

MODEL_PATH = os.path.join(DATA_PATH,'fastext_articles.bin')

## COLORS

PRIM_DARK = (0.13725490196078433, 0.21568627450980393, 0.23137254901960785)
SEC_DARK = (0.3764705882352941, 0.2980392156862745, 0.2196078431372549)

PRIM_LIGHT = (0.9215686274509803, 0.5058823529411764, 0.10588235294117647)
SEC_LIGHT = (0.7843137254901961, 0.47843137254901963, 0.1843137254901961)
