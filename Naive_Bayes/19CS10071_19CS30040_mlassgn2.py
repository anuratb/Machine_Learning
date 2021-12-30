import numpy as np
from numpy.lib.arraysetops import unique
import pandas as pd
import re
from scipy.sparse import csr_matrix
#from nltk.corpus import stopwords
#sw = stopwords.words("english")
sw = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

def get_word_tokens(text):
    '''
    param : text : text to be tokenised
    Returns unique word tokens excluding stopwords     
    '''
    word_token = (re.findall("[a-z0-9]+", text.lower()))
    for stp_wrd in sw:
        if stp_wrd in word_token : word_token.remove(stp_wrd)
    return unique(word_token)

def train_test_split(tdm,y):
    '''    
    Does a 70 : 30 split and returns a tuple denoting train and test
    param : tdm : term document matrix(binary matrix)
            y : labels
    return : tuple of form ((train_x,train_y),(val_x,val_y))
    '''
    n = len(y)
    choice = np.random.choice([0, 1], size=(n,), p=[0.3, 0.7])   
    val_y = []
    train_y = []
    train_ind = []
    val_ind = []
    for i in range(n):
        if(choice[i]==1) : 
            train_ind.append(i)
            train_y.append(y[i])
        else : 
            val_ind.append(i)
            val_y.append(y[i])
    train_x = tdm[train_ind]
    val_x = tdm[val_ind]
    train_y = pd.DataFrame(train_y,columns=['author'])
    val_y = pd.DataFrame(val_y,columns=['author'])
    return ((train_x,train_y),(val_x,val_y))


def prep_data(data_):
    '''
    params: data_ : The dataset to be used
    returns : A tuple of the form (word_vector,((train_x,train_y),(val_x,val_y)))
    word_vector : The bag of unique word in total found over the dataset exclusing stopwords
    train_x ,train_y : Training set with labels
    val_x,val_y : Validation set with labels
    '''
    data = data_.copy()
    #first calculate word tokens from entire dataset (need to take entire data as else we may have missing words)
    word_tokens =[]
    for text in data['text']:
        #print(text)
        word_tokens+=(re.findall("[a-z0-9]+", text.lower()))
    
    for stp_wrd in sw:
        if stp_wrd in word_tokens : word_tokens.remove(stp_wrd)
    word_vector = {}
    #Helper variables to construct csr matrix
    ind = []
    ln = [0]
    arr = []
    for text in data['text']:
        curr_vector = get_word_tokens(text)
        for word in curr_vector:
            currInd = word_vector.setdefault(word,len(word_vector))
            ind.append(currInd)
            arr.append(1)
        ln.append(len(ind))
    tdm = csr_matrix((arr, ind, ln), dtype=int)
    y = data['author']
    (train_x,train_y),(val_x,val_y) = train_test_split(tdm,y)
    return (word_vector,((train_x,train_y),(val_x,val_y)))


def calc_likelihood(y:pd.DataFrame,word_vector:dict,tdm:csr_matrix,alpha = 0):
    '''
    parameters :
     data : the training data
     word_vector : the dictionary mapping word_tokens to their id
    return : The likelihood of each word vector given each class
    '''
   
    authors = dict(y.groupby('author').apply(lambda itr: len(itr)))
    likelihood = {}
    for author,frq in authors.items() : likelihood[author] = np.zeros((1,len(word_vector)))

    for ind,row in y.iterrows(): 
        #print(np.array(tdm[ind,:].toarray()).shape,ind)       
        currRow = np.array(tdm[ind,:].toarray())
        #print(currRow.shape)
        #print(likelihood[row.author].shape)
        likelihood[row.author]+=currRow
    for author,val in authors.items():
        likelihood[author]+=alpha#Laplace correction
        likelihood[author]/=(val+alpha*len(word_vector))
    return likelihood


#Calculate priors
def calc_priors(y:pd.DataFrame):
    '''
    Arguments : data
    Returns a dictionary containing the prior probabilities
    '''
    return dict(y.groupby('author').apply(lambda itr: len(itr))/len(y))


class model:
    def __init__(self,alpha=0):
        '''
        alpha : parameter for laplace correction = 0 for no laplace correction
        '''
        self.data = pd.read_csv('train.csv')
        self.authors = list(self.data['author'].unique())
        self.word_vector,((self.train_x,self.train_y),(self.val_x,self.val_y)) = prep_data(self.data)
        self.priors = calc_priors(self.train_y)
        self.alpha = alpha
        self.likelihood = calc_likelihood(self.train_y,self.word_vector,self.train_x,alpha)
        #print(self.word_vector)
        #print(self.train_x.toarray(),self.train_y)
    def predict(self,x_):
        '''
        parameter : x : a row 
        return author name as predicted
        '''
        posterior_list = []
        x = x_.reshape(len(self.word_vector))
        
        for author in self.authors:
            if self.alpha == 0:
                #-------------WITHOUT TAKING LOG------------
                posterior = (self.priors[author])
                posterior*=np.prod((x*self.likelihood[author]+(1-x)*(1.0-self.likelihood[author])))
            else:
                #-------------TAKING LOG--------------------
                posterior = np.log(self.priors[author])
                posterior+=np.sum(np.log(x*self.likelihood[author]+(1-x)*(1.0-self.likelihood[author])))
            '''
            for i in range(len(x)):
                attr = x[i]
                #print(attr)
                if attr==1:
                    posterior*=(self.likelihood[author][0,i])
                else:
                    posterior*=(1.0-self.likelihood[author][0,i])
            '''
            posterior_list.append(posterior)
        posterior_list = np.array(posterior_list) 
        
        #print(posterior_list,self.authors,self.val_y)
        #print(self.authors[np.argmax(posterior_list)])
        return self.authors[np.argmax(posterior_list)]
    def eval(self):
        '''
        For Model evaluation
        '''
        predictions = []
        for i in range(len(self.val_y)):
            #if(i%100==0): print("| |")
            predictions.append(self.predict(self.val_x[i,:].toarray()))
        
        y_val = np.array(self.val_y)
        predictions = np.array(predictions).reshape((len(y_val),1))
        #print(predictions)
        #print((predictions==y_val).shape,predictions.shape,y_val.shape)
        #print(np.sum(predictions==y_val),predictions.shape)
        mu = (np.sum(predictions==y_val))/predictions.shape[0]
        sigma = 1.96*np.sqrt(mu*(1-mu)/len(self.val_y))
        print('Accuracy: ',mu,' +/- ',sigma)
        #print("Confusion Matrix")
        confusion_matrix = {}
        for author in self.authors:
            confusion_matrix[author] = {}
            for auth in self.authors:
                confusion_matrix[author][auth] = 0
        for i in range(len(self.val_y)):
            confusion_matrix[self.val_y.author[i]][predictions[i,0]]+=1
        #confusion_matrix_ = pd.DataFrame(confusion_matrix,columns=self.authors)
        #print(confusion_matrix_)
        precision = {}
        tp = 0
        fn = 0
        fp = 0
        tn = 0
        f_score = {}
        sensitivity = {}
        specificity = {}
        
        #precision
        for author in self.authors:#Actual label    
            tp = fp = 0.0        
            for auth in self.authors:#Predicted
                if(auth==author) : tp+=confusion_matrix[author][auth]
                else:  fp+=confusion_matrix[author][auth]
            precision[author] = tp/(tp+fp)

        #sensitivity
        for auth in self.authors:#Predicted  
            tp = fn = 0.0        
            for author in self.authors:#Actual
                if(auth==author) : tp+=confusion_matrix[author][auth]
                else:  fn+=confusion_matrix[author][auth]
            sensitivity[auth] = tp/(tp+fn)
        
        #specificity
        for author in self.authors:#Actual label    
            tn = fp = 0.0
            for i in self.authors:
                for j in self.authors:
                    if i != author:
                        if j!=author :
                            tn+=confusion_matrix[i][j]
                    else :
                        if(j==author):
                            fp+=confusion_matrix[i][j]
            specificity[author] = tn/(tn+fp)
        #f score
        for author in self.authors:
            f_score[author] = 2/((1/precision[author])+(1/sensitivity[author]))
        print('Precision: ')
        for author in self.authors:
            print(author,' : ',precision[author])
        print('F-score: ')
        for author in self.authors:
            print(author,' : ',f_score[author])
        print('Sensitivity: ')
        for author in self.authors:
            print(author,' : ',sensitivity[author])
        print('Specificity: ')
        for author in self.authors:
            print(author,' : ',specificity[author])



if __name__ == '__main__':
    print('Without Laplace Correction')
    md = model()
    md.eval()
    print('With Laplace Correction')
    md = model(1.0)
    md.eval()

        



