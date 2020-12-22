
import pandas as pd
import numpy as np
import re

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt

# Gensim
import gensim
from gensim.utils import simple_preprocess

# NLTK
import nltk
from nltk.corpus import stopwords

from collections import Counter
from wordcloud import WordCloud

#from util.util import *
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import spacy
import re
from pprint import pprint

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

from collections import Counter
from wordcloud import WordCloud


# Gensim
import gensim
from gensim.utils import simple_preprocess


from collections import Counter
from wordcloud import WordCloud
import datetime

import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)



from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


stop_words = stopwords.words('english')
stop_words.extend(['from', 'https', 'twitter', 'esmo', 'pic','twitt','congress','breast','cancer'])
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

print("lib import successful")