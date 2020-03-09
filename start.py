from data_clean import data_cleaner
from dataLoader import data_loader
from text3rank import snatextrank3key
import numpy as np
import pandas as pd



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
    processed_wholedata=cleaner.processed_data
    new_dict = dict([(k,d.field_dict[k]) for k in sorted(d.field_dict.keys())])

    ranker = snatextrank3key(4,0.5,cleaner)
    '''
    test in dictionary
    '''
    # for k,v in new_dict.items():
    #     ranker.k=k
    #     v_list = v[:,1].tolist()
    #     v_list = pd.Series(v_list).str.replace("http.*", "", regex=True).tolist()
    #     v_list = pd.Series(v_list).str.replace("pic.twitter.com.*", "", regex=True).tolist()
    #     np.savetxt("./results/"+k,np.array(v_list),fmt="%s")
    #     print(k,len(v_list))
    #     vocabulary,unique_phrases = ranker.build_vocabulary(v_list,txtpath=glove_vec_path)
    #     print("---"*20+"Ranking"+"---"*20)
    #     ranker.build_matrix(vocabulary)
    #     ranker.calc_inout()
    #     ranker.iteration()
    #     print("For phrase:\n")
    #     ranker.ranking(10,1)
    #     print("For words:\n")
    #     ranker.ranking(10,0)
    #     break

    '''
    test for article
    '''
    ranker.k="test"
    Text = "Compatibility of systems of linear constraints over the set of natural numbers. \
            Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and \
            nonstrict inequations are considered. \
            Upper bounds for components of a minimal set of solutions and \
            algorithms of construction of minimal generating sets of solutions for all \
            types of systems are given. \
            These criteria and the corresponding algorithms for constructing \
            a minimal supporting set of solutions can be used in solving all the \
            considered types of systems and systems of mixed types."
    test_str_list = [Text]
    vocabulary,unique_phrases = ranker.build_vocabulary(test_str_list,txtpath=glove_vec_path)
    print("---"*20+"Ranking"+"---"*20)
    ranker.build_matrix(vocabulary)
    ranker.calc_inout()
    ranker.iteration()
    print("For phrase:\n")
    ranker.ranking(10,1)
    print("For words:\n")
    ranker.ranking(10,0)