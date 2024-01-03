import numpy as np
# import matplotlib.pyplot as plt
# import spacy
# from spacy import displacy
# import nltk
# import pandas as pd
# from sklearn.pipeline import Pipeline
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.svm import LinearSVC
# from sklearn.model_selection import train_test_split
# from sklearn import metrics
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from nltk.sentiment.vader import SentimentIntensityAnalyzer



# nlp = spacy.load('en_core_web_sm')

# print(nlp.pipe_names)
# doc = nlp(u'it is a $10 doller gift card from Google.you can get it from "www.fgm.com" website! like seriously!!!')

# for t in doc:
#   print(t,t.pos_,t.lemma_)

# displacy.render(doc,style = 'dep',jupyter = True,options={'distance':80})

# pd.DataFrame(nlp.Defaults.stop_words)

# df = pd.read_csv('DATA\smsspamcollection.tsv',sep='\t')
# df.head(10)



# X = df['message']
# y = df['label']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# vectorizer = CountVectorizer()

# # Fit and transform the text data
# X_vect = vectorizer.fit_transform(X_train)

# # feature names (words)
# feature_names = vectorizer.get_feature_names_out()

# #dense matrix
# dense_matrix = X_vect.toarray()

# print("Feature Names (Words):", feature_names)
# print("Feature Matrix:\n", dense_matrix)

# tfidf = TfidfVectorizer()


# # Fit and transform the text data
# X_tfidf = tfidf.fit_transform(X_train)

# # Get feature names (words)
# feature_names_tfidf = tfidf.get_feature_names_out()

# # Convert to dense matrix (for demonstration purposes)
# dense_matrix_tfidf = X_tfidf.toarray()

# print("Feature Names (Words):", feature_names_tfidf)
# print("TF-IDF Feature Matrix:\n", dense_matrix_tfidf)

# movie_df = pd.read_csv('DATA\moviereviews.tsv',sep='\t')
# movie_df

# movie_df.isnull().sum()

# movie_df.dropna(inplace = True)

# X = movie_df['review']
# y = movie_df['label']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# # Naïve Bayes:
# text_clf_nb = Pipeline([('tfidf', TfidfVectorizer()),
#                      ('clf', MultinomialNB()),
# ])

# # Linear SVC:
# text_clf_lsvc = Pipeline([('tfidf', TfidfVectorizer()),
#                      ('clf', LinearSVC()),
# ])

# text_clf_nb.fit(X_train, y_train)

# predictions = text_clf_nb.predict(X_test)
# # predictions

# print(metrics.confusion_matrix(y_test,predictions))

# print(metrics.classification_report(y_test,predictions))

# print(metrics.accuracy_score(y_test,predictions))

# text_clf_lsvc.fit(X_train, y_train)

# predictions = text_clf_lsvc.predict(X_test)

# print(metrics.confusion_matrix(y_test,predictions))

# print(metrics.accuracy_score(y_test,predictions))

# stopwords = ['a', 'about', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'but', 'by', 'can', \
#              'even', 'ever', 'for', 'from', 'get', 'had', 'has', 'have', 'he', 'her', 'hers', 'his', \
#              'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'just', 'me', 'my', 'of', 'on', 'or', \
#              'see', 'seen', 'she', 'so', 'than', 'that', 'the', 'their', 'there', 'they', 'this', \
#              'to', 'was', 'we', 'were', 'what', 'when', 'which', 'who', 'will', 'with', 'you']

# """### adding stop words to imporve calssification"""

# text_clf_lsvc2 = Pipeline([('tfidf', TfidfVectorizer(stop_words=stopwords)),
#                      ('clf', LinearSVC()),
# ])
# text_clf_lsvc2.fit(X_train, y_train)

# predictions = text_clf_lsvc2.predict(X_test)
# print(metrics.confusion_matrix(y_test,predictions))

# print(metrics.classification_report(y_test,predictions))

# print(metrics.accuracy_score(y_test,predictions))

# myreview1 = "A movie I really wanted to love was terrible. \
# I'm sure the producers had the best intentions, but the execution was lacking."

# myreview2 = "as a thriller movie fan i was looking for a movie that is full of thrill and this movie \
# did not dissapoint me. though i thought it won't meet my expection but i was wrong."

# print(text_clf_nb.predict([myreview1]))
# print(text_clf_lsvc2.predict([myreview2]))

# ### Using lemmatization for performence improvement

# # Load Spacy's English model
# nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# # Lemmatization and stopwords
# def spacy_lemmatize(text):
#     return ' '.join([token.lemma_ for token in nlp(text) if not token.is_stop])

# # Applying Lemmatization and custom stopwords
# X_train_lemmatized = [spacy_lemmatize(text) for text in X_train]
# X_test_lemmatized = [spacy_lemmatize(text) for text in X_test]

# # Updating the model pipeline with Lemmatized text
# text_clf_lsvc3 = Pipeline([
#     ('tfidf', TfidfVectorizer()),
#     ('clf', LinearSVC())
# ])

# text_clf_lsvc3.fit(X_train_lemmatized, y_train)

# predictions = text_clf_lsvc3.predict(X_test_lemmatized)

# print(metrics.confusion_matrix(y_test, predictions))
# print(metrics.classification_report(y_test, predictions))
# print(metrics.accuracy_score(y_test, predictions))

# print(text_clf_lsvc3.predict([myreview1]))
# print(text_clf_lsvc3.predict([myreview2])) #it should be posative




# # ...................................Using VADER SENTIMENT ANALYSIS.................
# sid = SentimentIntensityAnalyzer()
# print(sid.polarity_scores(myreview2))

# movie_df['scores'] = movie_df['review'].apply(lambda r : sid.polarity_scores(r))
# print(movie_df.head(10))
# movie_df['compound'] = movie_df['scores'].apply(lambda d : d['compound'])
# movie_df['compound_score'] = movie_df['compound'].apply(lambda s : 'pos' if s > 0 else 'neg' )
# movie_df.head(10)