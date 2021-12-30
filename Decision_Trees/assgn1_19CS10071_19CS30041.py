import numpy as np
import pandas as pd
from graphviz import Graph
import matplotlib.pyplot as plt

'''
data = pd.read_csv('diabetes.csv')
data = data.dropna()
bin = []
copy_ = data.copy()
copy_.drop('Outcome',axis='columns',inplace=True)
for col in copy_.columns:
    mean = copy_[col].mean()
    sd = copy_[col].std()
    row=[mean-3*sd,mean-2*sd,mean-sd,mean,mean+sd,mean+2*sd,mean+3*sd]
    bin.append(row)
    
def discretise_normal(X):
    for col in X.columns:
        #print(col)
        if(col=='Outcome'):  continue
        mean = X[col].mean()
        sd = X[col].std()
        mx = X[col].max()
        mn = X[col].min()
        #print([mean-3*sd,mean-2*sd,mean-sd,mean,mean+sd,mean+2*sd,mean+3*sd,mx])
        if mx>=mean+3*sd and mn<=mean-3*sd:X[col] = pd.cut(x = X[col],bins=[mn-1,mean-3*sd,mean-2*sd,mean-sd,mean,mean+sd,mean+2*sd,mean+3*sd,mx+1],labels=[0,1,2,3,4,5,6,7])
        elif mx>=mean+3*sd: X[col] = pd.cut(x = X[col],bins=[mean-3*sd,mean-2*sd,mean-sd,mean,mean+sd,mean+2*sd,mean+3*sd,mx+1],labels=[0,1,2,3,4,5,6])
        elif mn<=mean-3*sd: X[col] = pd.cut(x = X[col],bins=[mn-1,mean-3*sd,mean-2*sd,mean-sd,mean,mean+sd,mean+2*sd,mean+3*sd],labels=[0,1,2,3,4,5,6])
        else: X[col] = pd.cut(x = X[col],bins=[mean-3*sd,mean-2*sd,mean-sd,mean,mean+sd,mean+2*sd,mean+3*sd],labels=[0,1,2,3,4,5])   
        '''
#train_data = data.sample(frac = 0.8)
#val_data = data.drop(train_data.index)

'''
For reading and  preparing(discretising data)
'''
def read_prepare_data():
    data = pd.read_csv('diabetes.csv')
    data = data.dropna()
    bin = []
    #return (data,bin)
    for col in data.columns:
        #print(col)
        #if(col=='Pregrancies') : continue  #this also reduces accuracy
        
        if(col=='Outcome'):  continue
        mean = data[col].mean()
        sd = data[col].std()
        mx = data[col].max()
        mn = data[col].min()
        cbin = []
        if False:
            #this part is ommited empirically to increse accuracy
            intv = (mx-mn)/10
            prev = mn
            for i in range(10):
                if i==0:cbin.append(prev-1)
                else:cbin.append(prev)
                prev+=intv
            cbin.append(prev+1)
        else:
            if mx>=mean+2*sd and mn<=mean-2*sd:
                cbin = [mn-1,mean-2*sd,mean-sd,mean,mean+sd,mean+2*sd,mx+1]            
            elif mx>=mean+2*sd: 
                cbin = [mean-2*sd,mean-sd,mean,mean+sd,mean+2*sd,mx+1]            
            elif mn<=mean-2*sd: 
                cbin = [mn-1,mean-2*sd,mean-sd,mean,mean+sd,mean+2*sd]            
            else: 
                cbin = [mean-2*sd,mean-sd,mean,mean+sd,mean+2*sd]
        #print(cbin)
        #print([mean-3*sd,mean-2*sd,mean-sd,mean,mean+sd,mean+2*sd,mean+3*sd,mx])
        '''

        if mx>=mean+3*sd and mn<=mean-3*sd:
            cbin = [mn-1,mean-3*sd,mean-2*sd,mean-sd,mean,mean+sd,mean+2*sd,mean+3*sd,mx+1]            
        elif mx>=mean+3*sd: 
            cbin = [mean-3*sd,mean-2*sd,mean-sd,mean,mean+sd,mean+2*sd,mean+3*sd,mx+1]            
        elif mn<=mean-3*sd: 
            cbin = [mn-1,mean-3*sd,mean-2*sd,mean-sd,mean,mean+sd,mean+2*sd,mean+3*sd]            
        else: 
            cbin = [mean-3*sd,mean-2*sd,mean-sd,mean,mean+sd,mean+2*sd,mean+3*sd]            
        '''
        '''
        if mx>=mean+3*sd and mn<=mean-3*sd:
            cbin = [mn-1,mean-3*sd,mean-2*sd,mean-sd,mean-0.8*(sd), mean-0.6*(sd), mean-0.4*(sd),mean-0.2*(sd), mean, mean+0.2*(sd), mean+0.4*(sd), mean+0.6*(sd),mean+0.8*(sd),mean+sd,mean+2*sd,mean+3*sd,mx+1]            
        elif mx>=mean+3*sd: 
            cbin = [mean-3*sd,mean-2*sd,mean-sd,mean-0.8*(sd), mean-0.6*(sd), mean-0.4*(sd),mean-0.2*(sd), mean, mean+0.2*(sd), mean+0.4*(sd), mean+0.6*(sd),mean+0.8*(sd),mean+sd,mean+2*sd,mean+3*sd,mx+1]            
        elif mn<=mean-3*sd: 
            cbin = [mn-1,mean-3*sd,mean-2*sd,mean-sd,mean-0.8*(sd), mean-0.6*(sd), mean-0.4*(sd),mean-0.2*(sd), mean, mean+0.2*(sd), mean+0.4*(sd), mean+0.6*(sd),mean+0.8*(sd),mean+sd,mean+2*sd,mean+3*sd]            
        else: 
            cbin = [mean-3*sd,mean-2*sd,mean-sd,mean-0.8*(sd), mean-0.6*(sd), mean-0.4*(sd),mean-0.2*(sd), mean, mean+0.2*(sd), mean+0.4*(sd), mean+0.6*(sd),mean+0.8*(sd),mean+sd,mean+2*sd,mean+3*sd]
        '''
        '''
        Based on uniformly discrtising into 10 groups
        intv = (mx-mn)/10
        prev = mn
        for i in range(10):
            if i==0:cbin.append(prev-1)
            else:cbin.append(prev)
            prev+=intv
        cbin.append(prev+1)
        #print(cbin)
        '''
         
        bin.append(cbin)
        data[col] = pd.cut(x = data[col],bins=cbin,labels=[i for i in range(len(cbin)-1)])
       # print(data)
    return (data,bin)
'''
Makes a copy of the data splits it into test and train and returns the split data
'''
def train_test_split(data_):
    data = data_.copy()
    train_data = data.sample(frac = 0.8)
    val_data = data.drop(train_data.index)
    y = train_data['Outcome'].values
    X = train_data
    X.drop('Outcome',axis='columns',inplace=True)    
    X_train_tuple = (X,y)
    y_val = val_data['Outcome'].values
    X_val = val_data
    X_val.drop('Outcome',axis='columns',inplace=True)   
    X_test_tuple = (X_val,y_val)
    return(X_train_tuple,X_test_tuple)

def getEntropy(p,n):    
    '''
    Params :
    p : number of positive samples 
    n : number of negative samples
    returns entropy of a node with p pos and n neg examples
    '''
    if(p ==0 or n == 0) : return 0.0
    return -(p/(p+n))*np.log2(p/(p+n)) - (n/(n+p))*np.log2(n/(n+p))
   
def getGini(p,n):    
    '''
    Params :
    p : number of positive samples 
    n : number of negative samples
    returns entropy of a node with p pos and n neg examples
    '''
    return 1-(p**2+n**2)/((p+n)**2)
    



class Node:
    '''
    Class to store each node of the tree
    '''
    def __init__(self,attrList,data,y,func,func_nme,depth=0):
        '''
        attrList : List of attributes present in current node
        adj : Link to next nodes based on attribute value
        data : Data used in current node
        Func : Impurity function used
        attrChosen : Chosen attribute to split to next nodes None if leaf
        '''
        self.attrList = attrList #contains list of attributes present in current node
        self.adj = {}  #adjaceny link to descendent nodes
        self.data = data
        self.depth= depth
        self.Funcname = func_nme
        #self.dtype = dtype  #data type in format (type_name,type)
        self.Func = func  #impurity function
        self.attrChosen = None #none if leaf
        self.y = y
        self.isPruned = 0# denotes if the subtrees below it are pruned

    def getFunc(self):
        p =0 
        for y_ in self.y : 
            if y_==0:
                p+=1      
        curFunc = self.Func(p,len(self.y)-p)
        return curFunc
    def isLeaf(self):
        if self.isPruned :return 1
        return len(self.adj)==0
    def predictedVal(self):
        #return label of majority
        p =0 
        for y_ in self.y : 
            if y_==1:
                p+=1 
        n = len(self.y)-p
        if p>n:return 1
        return 0
    
    def __str__(self):
        #str1 = str("%s,\n,%s%s,\n predicted value = %s"%(self.attrList,self.Funcname,self.getFunc(),self.predictedVal()))
        #str2 = str("%s,\n,%s%s,\n"%(self.attrList,self.Funcname,self.getFunc()))
        if(self.isLeaf()):
            return str("%s,\n,%s = %s,\n predicted value = %s"%(self.attrList,self.Funcname,self.getFunc(),self.predictedVal()))
        return str("%s,\n,%s = %s,\nAttribute Chosen = %s"%(self.attrList,self.Funcname,self.getFunc(),self.attrChosen))

    



class DTClassifier:
    '''
    Class for the entire DTClassifier
    '''

    def __init__(self,attrDict:dict,attrInd:dict,X,y,Func,FuncName,maxDepth=None):
        '''
        X : input(np array)
        y : Labels
        Func : Impurity function to be used
        attrDict : Attribute Dictionary (attrName-> values)
        '''
        self.depth = 1
        self.attrDict = attrDict   #stores values corresponding to each attribute
        self.attrInd = attrInd
        self.X = X
        self.y = y    
        self.nodes = 0  #total nodes
        self.Func = Func
        self.FuncName = FuncName
        self.maxDepth = maxDepth
        self.root = Node(list(attrDict.keys()),X,y,self.Func,self.FuncName)
        
    def propagate(self,node:Node):    
        #print(node.data)
        #print(node.attrList)
        if self.maxDepth is not None and node.depth>=self.maxDepth : return 
        self.depth = max(self.depth,node.depth)        
        p =0 
        for y_ in node.y : 
            if y_==0:
                p+=1      
        curFunc = self.Func(p,len(node.y)-p)
        #print(curFunc)
        if curFunc == 0:
            return
        if(len(node.attrList)==0):return 
        attrChoose = None
        maxInfoG = 0 
        for attr in node.attrList:
            curDict = {}
            #id = self.attrInd[attr] #get the attribute index
            for i in range(len(node.data)):
                x = node.data[i]
                y = node.y[i]               
                attrVal = int(x[self.attrInd[attr]])
                if attrVal in curDict.keys() :
                    curDict[attrVal][y]+=1
                else:
                    curDict[attrVal] = [0,0]
                    curDict[attrVal][y]+=1
            newFunc = 0
            n = len(node.data)
            for key,val in curDict.items():
                newFunc+=(((val[0]+val[1])/n)*self.Func(val[0],val[1])) 
            curInfoGain = curFunc-newFunc
            if curInfoGain>maxInfoG:
                attrChoose = attr
                maxInfoG = curInfoGain
        #now attribute for split has ben chosen
       # id = self.attrInd[attrChoose]
        #print(attrChoose)
        if(attrChoose is None):return 
        divX = {}   #split and divide  : attrVal -> [data_x,data_y]
        for i in range(len(node.data)):
            x = node.data[i]
            y = node.y[i]
            attrVal = int(x[self.attrInd[attrChoose]])
            if attrVal in divX.keys() :
                divX[attrVal][0].append(x)
                divX[attrVal][1].append(y)
            else:
                divX[attrVal] = [[x],[y]]
        newAttrList = node.attrList.copy()
        newAttrList.remove(attrChoose)
        node.attrChosen = attrChoose

        for key,val in divX.items():     
            self.nodes+=1       
            cur = Node(newAttrList,val[0],val[1],self.Func,self.FuncName,node.depth+1)#crete next Node
            node.adj[key] = cur#add to adjacency list
            self.propagate(cur)
        
        
    def fit(self):
        self.nodes+=1
        self.propagate(self.root)
    def predict(self,a,currNode=None):
        if(currNode is None ):currNode = self.root
        if(currNode.isLeaf()==0):
            m=int(self.attrInd[currNode.attrChosen])
            k=a[m]
            if k not in currNode.adj.keys():
                return currNode.predictedVal   #since we cant proceed furthur
            return self.predict(a,currNode.adj[k])
        else:
            return currNode.predictedVal()
    
    def getAccuracy(self,val_X,val_y):
        cnt = 0
        for i in range(len(val_X)):
            x = val_X[i]
            y = val_y[i]
            y_hat = self.predict(x)
            if(y_hat==y): cnt+=1
        return cnt/len(val_X)
    def dfs(self,nd:Node):
        if nd.isLeaf(): return 1
        ans = 1
        for key,val in nd.adj.items():
            ans+=self.dfs(val)
        return ans
        
    def getNodes(self):   #returnes number of nodes
        return self.dfs(self.root)

    def reduced_errorPruning(self,val_X,val_y):
        prevAcc = self.getAccuracy(val_X,val_y)
        while True:
            mxAcc = 0.0 #set current max accuracy to 0
            nd = None #node which will be pruned at this stage
            queue = [self.root]
            #do a bfs
            while(len(queue)>0):
                cur = queue.pop(0)
                if cur.isLeaf() :continue
                cur.isPruned = True#temporarily prune
                curAcc = self.getAccuracy(val_X,val_y)
                if curAcc>mxAcc:
                    mxAcc = curAcc
                    nd = cur
                cur.isPruned = False#restore the pruning
                #traverse other nodes
                
                for key,val in cur.adj.items():
                    queue.append(val)
            if mxAcc>prevAcc:
                prevAcc = mxAcc
                nd.isPruned = True #permanently prune
               
            else: break   #validation accuracy not decreasing so break

    
'''   
#seprate train dataa
y = train_data['Outcome'].values
X = train_data
X.drop('Outcome',axis='columns',inplace=True)
#separate val data
y_val = val_data['Outcome'].values
X_val = val_data
X_val.drop('Outcome',axis='columns',inplace=True)
print(X_val)

discretise_normal(X)
discretise_normal(X_val)
'''
'''
attrDict = {}
for col in X.columns:
    attrDict[col] = X[col].unique()
attrInd = {
    'Pregnancies':0,

 'Glucose': 1,
 'BloodPressure': 2,
 'SkinThickness': 3,
 'Insulin': 4,
 'BMI': 5,
 'DiabetesPedigreeFunction': 6,
 'Age': 7
}
X = list(np.array(X))#change to list
model = DTClassifier(attrDict,attrInd,None,X,y,getEntropy)

model.fit()
'''
# pretty print the tree
def print_tree(root:Node,flname,bins,attrInd):
    g = Graph('Decision Tree',filename=flname)
    g.attr(size='1000,3000')
    g.attr('node',shape='rectangle')
    q = [root]
    while len(q)>0:
        curr = q.pop(0)
        if curr.isLeaf(): continue
        for key,val in curr.adj.items():
            q.append(val)#push to queue
            lbl = (bins[attrInd[curr.attrChosen]][key+1] + bins[attrInd[curr.attrChosen]][key])/2
            g.edge(str(val),str(curr),label=str(lbl))
    g.render('./'+flname,view=True)



#print_tree(model.root)


class model:
    def __init__(self,Func,FuncName,maxDepth = None):
        self.data ,self.bin = read_prepare_data()
        self.Func = Func
        self.maxDepth = maxDepth
        self.attrDict = {}
        self.FuncName = FuncName
        for col in self.data.columns:
            if col == 'Outcome' : continue
            self.attrDict[col] = self.data[col].unique()
        self.attrInd = {
        'Pregnancies':0,
        'Glucose': 1,
        'BloodPressure': 2,
        'SkinThickness': 3,
        'Insulin': 4,
        'BMI': 5,
        'DiabetesPedigreeFunction': 6,
        'Age': 7
        }
    def trainedModel(self):
        '''
        Does a random 80: 20 split and fits on the sample
        '''
        (train_X,train_y),(val_X,val_y) = train_test_split(self.data)     
        train_X = list(np.array(train_X))
        val_X = list(np.array(val_X))   
        DT = DTClassifier(self.attrDict,self.attrInd,train_X,train_y,self.Func,self.FuncName,self.maxDepth)
        DT.fit()
        return (DT,((train_X,train_y),(val_X,val_y)))
    def findMaxAccTree(self,noIter):
        acc = 0
        DT = None
        val_X = None
        val_y = None
        train_X = None
        train_y = None
        accAvg = 0
        for i in range(noIter):
            DT_c,((train_X_c,train_y_c),(val_X_c,val_y_c)) = self.trainedModel()
            currAcc = DT_c.getAccuracy(val_X_c,val_y_c)
            accAvg+=currAcc
            if currAcc > acc:
                acc = DT_c.getAccuracy(val_X_c,val_y_c)                
                DT = DT_c
                val_X = val_X_c
                val_y = val_y_c
                train_X,train_y = train_X_c,train_y_c
        accAvg/=noIter
        return ((DT,accAvg),((train_X,train_y),(val_X,val_y)))
    #pruning
            
if __name__ == "__main__":
   
    #First for Entropy as impurity function
    print('********Taking entropy as Impurity function*******\n')
    md = model(getEntropy,'Entropy')
    (DT,acc),((train_X,train_y),(val_X,val_y)) = md.findMaxAccTree(10)    
    #Ques 2
    print('Original Accuracy : {} \n'.format(acc))
    print_tree(DT.root,'Entropy_tree_original.gv',md.bin,md.attrInd)  #print original tree
    #Ques 3
    
    XAxis = []
    yAxis = []
    XNodes = []
    for depth in range(9):
        mdc = model(getEntropy,'Entropy',depth)
        (DT_c,acc_c),((_,_),(_,_)) = mdc.findMaxAccTree(10)
        yAxis.append(acc_c)
        XAxis.append(depth)
        XNodes.append(DT_c.getNodes())
    #depth vs test acc
    plt.plot(XAxis,yAxis)  
    plt.xlabel('Depth')  
    plt.ylabel('Test_Accuracy')
    plt.title('Test_Acc vs Depth')    
    plt.savefig('Acc_vs_depth_Entropy.pdf')
    plt.close()
    #nodes vs test acc
    plt.plot(XNodes,yAxis)  
    plt.xlabel('No of Nodes')  
    plt.ylabel('Test_Accuracy')
    plt.title('Test_Acc vs number of Nodes')    
    plt.savefig('Acc_vs_nodes_Entropy.pdf')
    plt.close()
    
    
    #Question 4 Prunings
    #reduced error pruning
    DT.reduced_errorPruning(val_X,val_y)
    print('Accuracy After Pruning : {} \n'.format(DT.getAccuracy(val_X,val_y)))    
    #Ques 5 print the final tree
    print_tree(DT.root,'Entropy_tree_after_pruning.gv',md.bin,md.attrInd)

    #For gini Index as impurity function
    print('********Taking Gini Index as Impurity function*******\n')
    md = model(getGini,'Gini Index')
    (DT,acc),((train_X,train_y),(val_X,val_y)) = md.findMaxAccTree(10)    
    #Ques 2
    print('Original Accuracy : {} \n'.format(acc))
    print_tree(DT.root,'Gini_tree_original.gv',md.bin,md.attrInd)#original gini tree
    #Ques 3
    
    XAxis = []
    yAxis = []
    XNodes = []
    for depth in range(9):
        mdc = model(getGini,'Gini index',depth)
        (DT_c,acc_c),((_,_),(_,_)) = mdc.findMaxAccTree(10)
        yAxis.append(acc_c)
        XAxis.append(depth)
        XNodes.append(DT_c.getNodes())
    #depth vs test acc
    plt.plot(XAxis,yAxis)  
    plt.xlabel('Depth')  
    plt.ylabel('Test_Accuracy')
    plt.title('Test_Acc vs Depth')
    
    plt.savefig('Acc_vs_depth_Gini.pdf')
    plt.close()
    #nodes vs test acc
    plt.plot(XNodes,yAxis)  
    plt.xlabel('No of Nodes')  
    plt.ylabel('Test_Accuracy')
    plt.title('Test_Acc vs number of Nodes')    
    plt.savefig('Acc_vs_nodes_Gini.pdf')
    plt.close()
    #Question 4 Prunings
    #reduced error pruning
    DT.reduced_errorPruning(val_X,val_y)
    print('Accuracy After Pruning : {} \n'.format(DT.getAccuracy(val_X,val_y)))    
    #Ques 5 print the final tree
    print_tree(DT.root,'Gini_tree_after_pruning.gv',md.bin,md.attrInd)
