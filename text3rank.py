from dataLoader import data_loader
from data_clean import data_cleaner
import data_clean
import numpy as np
import math
import pandas as pd


class snatextrank3key(object):
    def __init__(self,window_size,d,cleaner):
        self.MAX_ITERATIONS = 50
        self.d =d # damping factor
        self.threshold = 0.0001 #convergence threshold
        self.phrase_inout = None
        self.score = None
        self.window_size = window_size
        self.cleaner = cleaner
        self.k = None
    def build_vocabulary(self,subsents_list,txtpath):
        # 1. remove none printable characters
        # subsents_str = cleaner.remove_non_printable_characters(d.subsents_list)
        # 2. word lemmatization
        self.cleaner.word_Lemmatization(subsents_list,txtpath)
        # 3. removed stopwords
        self.cleaner.remove_stopwords()
        # 4. build vocabulary
        # vocabulary = list(set(cleaner.processed_data))
        vocabulary = self.cleaner.remove_non_word_vocabulary(txtpath)
        # print("---"*20)
        # print(len(cleaner.processed_data))
        self.vocabulary=vocabulary
        # 5. build unique_phrases
        phrases = []
        phrase = " "
        # print(len(cleaner.lemmatized_words))
        # print(cleaner.lemmatized_words)
        # print("***"*20)
        for word in self.cleaner.lemmatized_words:
            if word in self.cleaner.stopwords and word not in self.vocabulary:
                if phrase!= " ":
                    phrases.append(str(phrase).strip().split())
                phrase = " "
            elif word not in self.cleaner.stopwords and word in self.vocabulary:
                phrase+=str(word)
                phrase+=" "
        unique_phrases = []
        [unique_phrases.append(x) for x in phrases if x not in unique_phrases]
        # 6. Removing single word keyphrases-candidates that are present multi-word alternatives.
        for word in vocabulary:
            for phrase in unique_phrases:
                if (word in phrase) and ([word] in unique_phrases) and (len(phrase)>1):
                    #if len(phrase)>1 then the current phrase is multi-worded.
                    #if the word in vocabulary is present in unique_phrases as a single-word-phrase
                    # and at the same time present as a word within a multi-worded phrase,
                    # then I will remove the single-word-phrase from the list.
                    unique_phrases.remove([word])
        self.unique_phrases = unique_phrases
        return vocabulary,unique_phrases
    def scoring_phrases(self):
        phrase_scores = []
        keywords = []
        for phrase in self.unique_phrases:
            phrase_score=0
            keyword = ''
            for word in phrase:
                keyword += str(word)
                keyword += " "
                phrase_score+=self.score[self.vocabulary.index(word)]
            phrase_scores.append(phrase_score)
            keywords.append(keyword.strip())
        return keywords,phrase_scores
    def ranking(self,num=5,flag=1):
        """
        flag=1: phrase ranking
        flag=0: word ranking
        """
        if flag:
            keywords,phrase_scores=self.scoring_phrases()
            sorted_index = np.flip(np.argsort(phrase_scores),0)
            phrase_ranking = np.concatenate((np.array(keywords).reshape(-1,1)[sorted_index,:],np.array(phrase_scores).reshape(-1,1)[sorted_index,:]),axis=1)
            np.savetxt("./results/"+self.k+"phrase_ranking",phrase_ranking,fmt='%s')
            print("Keywords:\n")
            for i in range(0,num):
                print(str(keywords[sorted_index[i]])+", ", phrase_scores[sorted_index[i]])
        else:
            sorted_index = np.flip(np.argsort(self.score),0)
            words_ranking = np.concatenate((np.array(self.vocabulary).reshape(-1,1)[sorted_index,:],self.score.reshape(-1,1)[sorted_index,:]),axis=1)
            np.savetxt("./results/"+self.k+"words_ranking",words_ranking,fmt='%s')
            print("Keywords:\n")
            for i in range(0,num):
                print(str(self.vocabulary[sorted_index[i]])+", ", self.score[sorted_index[i]])

    def build_matrix(self,vocabulary,normalized=False):
        vocab_len = len(vocabulary)
        weighted_edge = np.zeros((vocab_len,vocab_len),dtype=np.float32)
        score = np.zeros((vocab_len),dtype=np.float32)
        covered_coocurrences = []
        for i in range(0,vocab_len):
            score[i]=1
            for j in range(0,vocab_len):
                if j==i:
                    weighted_edge[i][j]=0
                else:
                    for window_start in range(0,(len(self.cleaner.processed_data)-self.window_size)):
                        
                        window_end = window_start+self.window_size
                        
                        window = self.cleaner.processed_data[window_start:window_end]
                        
                        if (vocabulary[i] in window) and (vocabulary[j] in window):
                            
                            index_of_i = window_start + window.index(vocabulary[i])
                            index_of_j = window_start + window.index(vocabulary[j])
                            
                            if [index_of_i,index_of_j] not in covered_coocurrences:
                                weighted_edge[i][j]+=1/math.fabs(index_of_i-index_of_j)
                                covered_coocurrences.append([index_of_i,index_of_j])
        # normalized matrix
        if normalized:
            norm = np.sum(weighted_edge,axis=1)
            weighted_edge=np.divide(weighted_edge,norm,where=norm!=0)
        self.score=score
        self.weighted_edge=weighted_edge

    def calc_inout(self):
        word_inout = np.zeros((len(self.vocabulary)),dtype=np.float32)
        for i in range(0,len(self.vocabulary)):
            for j in range(0,len(self.vocabulary)):
                word_inout[i]+=self.weighted_edge[i][j]
        self.word_inout = word_inout
        return word_inout
    def iteration(self,flag=1):
        MAX_ITERATIONS = self.MAX_ITERATIONS
        score = self.score
        vocab_len = len(self.vocabulary)
        weighted_edge = self.weighted_edge
        if flag:
            inout = self.word_inout
        else:
            inout =self.phrase_inout
        threshold = self.threshold
        for iter in range(0,MAX_ITERATIONS):
            prev_score = np.copy(score)
            
            for i in range(0,vocab_len):
                
                summation = 0
                for j in range(0,vocab_len):
                    if weighted_edge[i][j] != 0:
                        summation += (weighted_edge[i][j]/inout[j])*score[j]
                        
                score[i] = (1-self.d) + self.d*(summation)
            if np.sum(np.fabs(prev_score-score)) <= threshold: #convergence condition
                print("Converging at iteration "+str(iter)+"....")
                break

if __name__ == '__main__':

    stopwords_path = "./stopwords_en"
    glove_vec_path = "./data/glove_vocabulary.txt"
    data_path = './data/Ferguson-en.xlsx'

    d = data_loader(data_path,0)
    d.load_data()
    d.build_data_dic()
    d.get_all_text()
    all_texts=d.all_texts

    cleaner = data_cleaner(stopwords_path,glove_vec_path)
    cleaner.word_Lemmatization(sents_list=all_texts,txtpath=glove_vec_path)
    cleaner.stopwords_generation()
    cleaner.remove_stopwords()
    stopwords=cleaner.stopwords
    # print(len(stopwords))
    processed_wholedata=cleaner.processed_data
    new_dict = dict([(k,d.field_dict[k]) for k in sorted(d.field_dict.keys())])

    ranker = snatextrank3key(3,0.85,cleaner)
    for k,v in new_dict.items():
        v_list = v[:,1].tolist()
        v_list = pd.Series(v_list).str.replace("http.*", "", regex=True).tolist()
        v_list = pd.Series(v_list).str.replace("pic.twitter.com.*", "", regex=True).tolist()
        np.savetxt("./results/"+k,np.array(v_list),fmt="%s")
        # print(type(v_list))
        # for i in v_list:
        #     print(i)
        # d.subsents_list=v_list
        print(k,len(v_list))
        vocabulary,unique_phrases = ranker.build_vocabulary(v_list,txtpath=glove_vec_path)
        # print(len(vocabulary))
        # print("*"*20)
        # print(len(unique_phrases))
        # print(cleaner.processed_data)
        # print(processed_wholedata)
        print("---"*20+"Ranking"+"---"*20)
        ranker.build_matrix(vocabulary)
        ranker.calc_inout()
        ranker.iteration()
        print("For phrase:\n")
        ranker.ranking(10,1)
        print("For words:\n")
        ranker.ranking(10,0)
        break
    
    # cleaner = data_cleaner()
    # sents_str = cleaner.remove_non_printable_characters()
    # cleaner.word_Lemmatization()
    # cleaner.stopwords_generation(stopwords_path)
    # cleaner.remove_stopwords()
    # stopwords=cleaner.stopwords
    # processed_wholedata=cleaner.processed_data