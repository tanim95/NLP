import spacy

nlp = spacy.load('en_core_web_lg')
# print(nlp(u'tiger').vector)
tokens = nlp(u'lion cat pet')
for i in tokens:
  for j in tokens:
    print(i.text,j.text,i.similarity(j))