from krwordrank.sentence import summarize_with_sentences
import pandas as pd
import re
# with open('stopword.txt','r') as f:
#     words= f.read()
# stopwords = words.split('\n')
# # print(words.split('\n'), type(words.split('\n')))
df = pd.read_csv('/opt/ml/input/final-project-level3-nlp-07/utils/테스트.csv')
inputs = [re.sub('\n','',sent) for sent in df.loc[:100, 'Message'].tolist()]
def key_word_extraction(inputs,penalty = None):
    text =  [re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]|[\n]|(@[A-Za-z0-9]+)|(\w+:\/\/\S+)", " ",sent) for sent in inputs]
    PATH = '/opt/ml/input/final-project-level3-nlp-07/utils/stopword.txt'
    with open(PATH,'r') as f:
        words= f.read()
    stopwords = words.split('\n')
    stopwords = {words for words in stopwords}
    # pos = ['채용', '취업', '코테', '인공지능', 'AI','알고리즘','면접','대기업', 'IT기업', 'IT', 'ML','DL','CNN', 'RNN', 'CV', 'NLP','Recsys']
    pnlty = lambda x:0 if any(word in x.split() for word in penalty) else 1
    key_word,sent= summarize_with_sentences(inputs, min_count=3, max_length=10,stopwords = stopwords,penalty=pnlty,
                                            beta=0.85, max_iter=10, verbose=False)
    return ' #' + ' #'.join(list(key_word.keys())[:3])

pos = ['채용', '취업', '코테', '인공지능', 'AI','알고리즘','면접','대기업', 'IT기업', 'IT', 'ML','DL','CNN', 'RNN', 'CV', 'NLP','Recsys']
# print(len(pos))
try:
    ans = key_word_extraction(inputs,pos)
except:
    ans = inputs[0]
print(ans)