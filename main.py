# Librerie da tenere a portata di mano
# pandas
# datetime
# sklearn
# matplotlib

###############################################
"""

# Primo step - Data integration

"""
###############################################

# Leggi da Excel
import pandas as pd
df_excel = pd.read_excel('dataset/Employee.xlsx', index_col=0)

print(df_excel.head())

# Leggi da JSON
df_json = pd.read_json('dataset/QR_Essence.json')

print(df_json.head())

# errore: il dataframe ha dei valori ripetuti!

# df_excel.reset_index(inplace=True)
# df_excel.to_json('dataset/Employee.json')

print(df_excel.describe())

print(df_excel.duplicated())

# .to_json()
# .to_html()
# .to_sql()
# ...

'''
###############################################
'''

# Secondo step - Data cleansing

'''
###############################################
'''

import pandas as pd

df = pd.read_excel("dataset/HR_Employee_data.xlsx")
print("################## First 5 records")
print(df.head())

print("################## Descrivi il dataset")
print(df.describe())

 # Dati mancanti - colonna age e average_montly_hours
print("################## Dati mancanti")
# colonna salary
column_salary = df['salary']
for var in df.columns:
    if df[var].isnull().sum()/len(df) > 0:
        print(var, df[var].isnull().mean().round(3))

 # Dati rumorosi : colonna left (inutile) e promotion (outliers)

print("################## Outliers values")
for var in df.columns:
    print(var, "min: ", df[var].min(), " max: ", df[var].max())

# Dati errati - unici

# HR Emp Dataset.xlsx - emp_id: ci serve?
print("################## Unique values")
for col in df.columns:
    print(col, df[col].nunique(), len(df))

 # Dati errati - duplicati
print("################## Duplicated values")
bool_series = df.duplicated()
print(bool_series)

# Terzo step - Data Transformation

# tutto in minuscolo - liste

texts=["pizza","Pizza","PIZZA","PiZzA"]
lower_words=[word.lower() for word in texts]
print(lower_words)

# tutto in minuscolo - dataframe

data = {'Fruits': ['BANANA', 'APPLE', 'MANGO', 'WATERMELON', 'PEAR'],
        'Color': ['Yellow', 'Red', 'Orange', 'Pink', 'Green']
        }

df = pd.DataFrame(data, columns=['Fruits', 'Color'])

print(df)

for column in df:
    df[column] = df[column].str.lower()

print(df)

# discretizzazione - one-hot encoding and countvectorizer

# Link completo: https://github.com/serenasensini/FZTH-Python-movieRS/
# Per il calcolo della similarità tra due film, andremo ad utilizzare un approccio molto semplice, ovvero la funzione di coseno-similarità: si tratta di una misura della somiglianza tra due vettori diversi da zero di uno spazio interno del prodotto che misura il coseno dell’angolo tra di loro. Applicando la definizione di similarità, questa sarà di fatto uguale a 1 se i due vettori sono identici e sarà 0 se i due sono ortogonali. In altre parole, la somiglianza è un numero limitato tra 0 e 1 che ci dice quanto i due vettori sono simili. In altre parole, il nostro modello non può funzionare con le sole stringhe che abbiamo prodotto, ma è necessario vettorializzare quelle singole stringhe in numeri; a questo scopo, si utilizza la feature di scikit-learn CountVectorizer che converte una raccolta di documenti di testo in una matrice di frequenze di token; si è scelto in questo caso di utilizzare CountVectorizer anziché TfIdfVecorizer per un semplice motivo: si aveva bisogno di un semplice contatore di frequenza per ogni parola nella colonna “bag\_of\_words”.
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

pd.set_option('display.max_columns', 100)
df = pd.read_csv('dataset/movie_metadata.csv')
print(df.head())

df = df[['director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name', 'plot_keywords', 'genres', 'movie_title']]

if not df['actor_1_name'].empty or not df['actor_2_name'].empty or not df['actor_3_name'].empty:
    df['actors'] = df['actor_1_name'] + "," + df['actor_2_name'] + "," + df['actor_3_name']

df = df[['director_name', 'plot_keywords', 'genres', 'movie_title', 'actors']]
df.dropna()
print(df.head())

df1 = df.where((pd.notnull(df)), 'REMOVE')
print(df1.head())

df.replace(["NaN"], np.nan, inplace=True)
df = df.dropna()
print(df.head())


for index, row in df.iterrows():
    # process actors names
    app = row['actors'].lower().replace(' ', '')
    app = app.replace(',', ' ')
    row['actors'] = app

    # process director_name
    app = row['director_name'].lower().replace(' ', '')
    row['director_name'] = app

    # process genres
    app = row['genres'].lower().replace('|', ' ')
    row['genres'] = app

    # process plot_keywords
    app = row['plot_keywords'].lower().replace('|', ' ')
    row['plot_keywords'] = app

print(df.head())

df.set_index('movie_title', inplace=True)
print(df.head())

df['bag_of_words'] = ''
columns = df.columns
for index, row in df.iterrows():
    words = ''
    for col in columns:
        words = words + row[col] + ' '
    row['bag_of_words'] = words

df.drop(columns=[col for col in df.columns if col != 'bag_of_words'], inplace=True)

print(df.head())

# instantiating and generating the count matrix
count = CountVectorizer()
count_matrix = count.fit_transform(df['bag_of_words'])

indices = pd.Series(df.index)
indices[:5]

cosine_sim = cosine_similarity(count_matrix, count_matrix)
print(cosine_sim)

'''




# BONUS: NLP pre-processing
'''
# Porter stemmer

from nltk.stem import PorterStemmer

porter_stemmer=PorterStemmer()

words=["trouble","troubles","troubling","troubled","troubleshooting"]
stemmed_words=[porter_stemmer.stem(word=word) for word in words]

from nltk.stem import SnowballStemmer


stemmer_snowball = SnowballStemmer('italian')


eg1 = ['andare', 'andai', 'andiamo', 'andarono']

eg_list = []

eg_list.extend(eg1)

print('Parole: {}\nRadici:'.format(eg_list))

for word in eg_list:

    print('\t- {}'.format(stemmer_snowball.stem(word)))

# lemmatization

import simplemma

mywords = ['migliore', 'camminiamo', 'mangiassimo', 'promisi', 'derivante']

langdata = simplemma.load_data('it')

result = []
for word in mywords:
    result.append(simplemma.lemmatize(word, langdata))

print(result)

# stopwords

stopwords=['this','that','and','a','we','it','to','is','of']
text="this is a text full of content and we need to clean it up"

words=text.split(" ")
shortlisted_words=[]

for w in words:
    if w not in stopwords:
        shortlisted_words.append(w)
    else:
        shortlisted_words.append("W")

print("original sentence = ",text)
print("sentence with stopwords removed= ",' '.join(shortlisted_words))

# rimuovere simboli di punteggiatura

import re
import string
testo = "Lorem, ipsum; testo! di -prova."
testo = re.sub('[%s]' % re.escape(string.punctuation), '' , testo)
print(testo)