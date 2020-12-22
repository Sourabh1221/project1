#!/usr/bin/env python
# coding: utf-8

from utils.lib import *
from utils.util import *

input_file='C:\\Users\\sourakumar\\Documents\\learning\\NLP\\Senti_Topic\\final\\data\\raw\\feedback_file.csv'
output_file='C:\\Users\\sourakumar\\Documents\\learning\\NLP\\Senti_Topic\\final\\data\\output\\detailed_report.csv'

df =pd.read_csv(input_file)
#df=feedback_df
df.drop_duplicates(subset=['feedback_txt'], keep='first',inplace=True)
df['feedback_txt'] = df['feedback_txt'].apply(str)
#df.shape


df['feedback_txt_re'] = np.vectorize(remove_users)(df['feedback_txt'], "@ [\w]*", "@[\w]*")
df['feedback_txt_re'] = df['feedback_txt_re'].str.lower()

df['feedback_txt_re'] = np.vectorize(remove_hashtags)(df['feedback_txt_re'], "# [\w]*", "#[\w]*")
df['feedback_txt_re'] = np.vectorize(remove_links)(df['feedback_txt_re'])

df['feedback_txt_re'] = df['feedback_txt_re'].str.replace("[^a-zA-Z#]", " ")
df['feedback_txt_re'] = df['feedback_txt_re'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))


df['feedback_txt_re_tokens'] = list(tokenize(df['feedback_txt_re']))
df['tokens_no_stop'] = remove_stopwords(df['feedback_txt_re_tokens'])


df['length'] = df['tokens_no_stop'].apply(len)
df = df.drop(df[df['length']<3].index)
df = df.drop(['length'], axis=1)
df['Creation_Date'] = pd.to_datetime(df['Creation_Date'])
df["tweet_length"] = df["feedback_txt"].str.len()
df['no_stop_joined'] = df.apply(rejoin_words, axis=1)

df["cleaned_tweet_length"] = df["no_stop_joined"].str.len()
df["tweet_length"] = df["feedback_txt"].str.len()
df["cleaned_tweet_length"] = df["no_stop_joined"].str.len()
df['no_stop_joined'] = df.apply(rejoin_words, axis=1)


df.drop_duplicates(subset=['feedback_txt'], keep='first', inplace=True)
count = df['feedback_txt'].str.split().str.len()
count.index = count.index.astype(str) + ' words:'
count.sort_index(inplace=True)
print("Total number of words:", count.sum(), "words")
print("Mean number of words per feedback:", round(count.mean(),2), "words")
df["feedback_length"] = df["feedback_txt"].str.len()
print("Total length of the dataset is:", df.feedback_length.sum(), "characters")

print("Mean Length of a feedback is:", round(df.feedback_length.mean(),0), "characters")
df = df.drop(['feedback_length'], axis=1)


word_freq = pd.Series(np.concatenate([x.split() for x in df.no_stop_joined])).value_counts()
word_df = pd.Series.to_frame(word_freq)
word_df['word'] = list(word_df.index)
word_df.reset_index(drop=True, inplace=True)
word_df.columns = ['freq', 'word']



##Uncomment below line for generating plot
#label = word_df['word'].head(25)
#freq = word_df['freq'].head(25)
#index = np.arange(len(freq))

#print("Unique words:", len(word_df))
#plt.figure(figsize=(12,9))
#plt.bar(index, freq, alpha=0.8, color= 'black')
#plt.xlabel('Words', fontsize=13)
#plt.ylabel('Frequency', fontsize=13)
#plt.xticks(index, label, fontsize=11, rotation=90, fontweight="bold") 
#plt.title('Top 25 Words after preprocessing', fontsize=12, fontweight="bold")
#plt.show()


df = df[['feedback_txt_re','User','Creation_Date','no_stop_joined','tokens_no_stop']]


vader_analyzer = SentimentIntensityAnalyzer()

### sentiment analysis
negative = []
neutral = []
positive = []
compound = []


def sentiment_scores(df, negative, neutral, positive, compound):
    for i in df['feedback_txt_re']:
        sentiment_dict = vader_analyzer.polarity_scores(i)
        negative.append(sentiment_dict['neg'])
        neutral.append(sentiment_dict['neu'])
        positive.append(sentiment_dict['pos'])
        compound.append(sentiment_dict['compound'])


sentiment_scores(df, negative, neutral, positive, compound)

# Prepare columns to add the scores later
df["negative"] = negative
df["neutral"] = neutral
df["positive"] = positive
df["compound"] = compound


# Fill the overall sentiment with encoding:
# (-1)Negative, (0)Neutral, (1)Positive
sentiment = []
for i in df['compound']:
    if i >= 0.05 : 
        sentiment.append('positive')
  
    elif i <= - 0.05 : 
        sentiment.append('negative') 
        
    else : 
        sentiment.append('neutral')
df['sentiment'] = sentiment

df= df[['User','Creation_Date','feedback_txt_re','compound', 'sentiment','no_stop_joined','tokens_no_stop']]
df


###################

cv = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')

dtm = cv.fit_transform(df['feedback_txt_re'])
LDA = LatentDirichletAllocation(n_components=7,random_state=42)

LDA.fit(dtm)

topic_results = LDA.transform(dtm)

#final_df = df[['User','Creation_Date']] 
df['Topic_number'] = topic_results.argmax(axis=1)

df1=[]
df2=[]
for index,topic in enumerate(LDA.components_):
    df1 = [cv.get_feature_names()[i] for i in topic.argsort()[-2:]]
    df1.append(list(map(int, str(index))))
    df2.append(df1)
dim_tbl=pd.DataFrame(df2)

dim_tbl.columns = [
  'Title_1',
  'Title_2',
   'Topic_number'
]
dim_tbl['Topic_number'] = dim_tbl['Topic_number'].apply(pd.Series)


final_df = pd.merge(df,
                 dim_tbl,
                 on='Topic_number')
final_df = final_df.drop(['tokens_no_stop','Topic_number','no_stop_joined'], axis=1)

final_df.to_csv(output_file,index=False)



















