A*
def aStarAlgo(start_node, stop_node):
 
 open_set = set(start_node) 
 closed_set = set()
 g = {} #store distance from starting node
 parents = {} # parents contains an adjacency map of all nodes
 #ditance of starting node from itself is zero
 g[start_node] = 0
 #start_node is root node i.e it has no parent nodes
 #so start_node is set to its own parent node
 parents[start_node] = start_node
 
 while len(open_set) > 0:
 n = None
 #node with lowest f() is found
 for v in open_set:
 if n == None or g[v] + heuristic(v) < g[n] + heuristic(n):
 n = v
 
 if n == stop_node or Graph_nodes[n] == None:
 pass
 else:
for (m, weight) in get_neighbors(n):
 #nodes 'm' not in first and last set are added to first
 #n is set its parent
 if m not in open_set and m not in closed_set:
 open_set.add(m)
 parents[m] = n
 g[m] = g[n] + weight
 
 
 #for each node m,compare its distance from start i.e g(m) to the
 #from start through n node
 else:
 if g[m] > g[n] + weight:
 #update g(m)
 g[m] = g[n] + weight
 #change parent of m to n
 parents[m] = n
 
 #if m in closed set,remove and add to open
 if m in closed_set:
 closed_set.remove(m)
 open_set.add(m)
 if n == None:
 print('Path does not exist!')
 return None
 # if the current node is the stop_node
 # then we begin reconstructin the path from it to the start_node
 if n == stop_node:
 path = []
 while parents[n] != n:
 path.append(n)
 n = parents[n]
 path.append(start_node)
 path.reverse()
 print('Path found: {}'.format(path))
 return path
 # remove n from the open_list, and add it to closed_list
 # because all of his neighbors were inspected
 open_set.remove(n)
 closed_set.add(n)
 print('Path does not exist!')
 return None
 
#define fuction to return neighbor and its distance
#from the passed node
def get_neighbors(v):
 if v in Graph_nodes:
 return Graph_nodes[v]
 else:
 return None
#for simplicity we ll consider heuristic distances given
#and this function returns heuristic distance for all nodes
def heuristic(n):
 H_dist = {
 'A': 10,
 'B': 8,
 'C': 5,
 'D': 7,
 'E': 3,
 'F': 6,
 'G': 5,
 'H': 3,
 'I': 1,
 'J': 0 
 }
 return H_dist[n]
#Describe your graph here 
Graph_nodes = {
 'A': [('B', 6), ('F', 3)],
 'B': [('C', 3), ('D', 2)],
 'C': [('D', 1), ('E', 5)],
 'D': [('C', 1), ('E', 8)],
 'E': [('I', 5), ('J', 5)],
 'F': [('G', 1),('H', 7)] ,
 'G': [('I', 3)],
 'H': [('I', 2)],
 'I': [('E', 5), ('J', 3)],
 
}
aStarAlgo('A', 'J')
6. Result/Output:
Path found: ['A', 'F', 'G', 'I', 'J']
['A', 'F', 'G', 'I', 'J']

2)    AO*
class Graph:
 def __init__(self, graph, heuristicNodeList, startNode): #instantiate graph object with 
 graph topology, heuristic values, start node
 self.graph = graph
 self.H=heuristicNodeList
 self.start=startNode
 self.parent={}
 self.status={}
 self.solutionGraph={}
def applyAOStar(self): # starts a recursive AO* algorithm
 self.aoStar(self.start, False)
 def getNeighbors(self, v): # gets the Neighbors of a given node
 return self.graph.get(v,'')
 
 def getStatus(self,v): # return the status of a given node
 return self.status.get(v,0)
 
 def setStatus(self,v, val): # set the status of a given node
 self.status[v]=val
 
 def getHeuristicNodeValue(self, n):
 return self.H.get(n,0) # always return the heuristic value of a given node
 def setHeuristicNodeValue(self, n, value):
 self.H[n]=value # set the revised heuristic value of a given node
 
 def printSolution(self):
 print("FOR GRAPH SOLUTION, TRAVERSE THE GRAPH FROM THE START 
NODE:",self.start)
 print("------------------------------------------------------------")
 print(self.solutionGraph)
 print("------------------------------------------------------------")
 
 def computeMinimumCostChildNodes(self, v): # Computes the Minimum Cost of child nodes of a given node v 
 minimumCost=0
 costToChildNodeListDict={}
 costToChildNodeListDict[minimumCost]=[]
 flag=True
 for nodeInfoTupleList in self.getNeighbors(v): # iterate over all the set of child node/s
 cost=0
 nodeList=[]
 for c, weight in nodeInfoTupleList:
 cost=cost+self.getHeuristicNodeValue(c)+weight
 nodeList.append(c)
 
 if flag==True: # initialize Minimum Cost with the cost of first set of child node/s 
 minimumCost=cost
 costToChildNodeListDict[minimumCost]=nodeList # set the Minimum Cost child node/s
 flag=False
 else: # checking the Minimum Cost nodes with the current Minimum Cost 
 if minimumCost>cost:
 minimumCost=cost
 costToChildNodeListDict[minimumCost]=nodeList # set the Minimum Cost child node/s
 
 return minimumCost, costToChildNodeListDict[minimumCost] # return Minimum Cost and 
Minimum Cost child node/s
def aoStar(self, v, backTracking): # AO* algorithm for a start node and backTracking status flag
 
 print("HEURISTIC VALUES :", self.H)
 print("SOLUTION GRAPH :", self.solutionGraph)
 print("PROCESSING NODE :", v)
 print("-----------------------------------------------------------------------------------------")
 
 if self.getStatus(v) >= 0: # if status node v >= 0, compute Minimum Cost nodes of v
 minimumCost, childNodeList = self.computeMinimumCostChildNodes(v)
 self.setHeuristicNodeValue(v, minimumCost)
 self.setStatus(v,len(childNodeList))
 
 solved=True # check the Minimum Cost nodes of v are solved 
 for childNode in childNodeList:
 self.parent[childNode]=v
 if self.getStatus(childNode)!=-1:
 solved=solved & False
 
 if solved==True: # if the Minimum Cost nodes of v are solved, set the current node status as solved(-1)
 self.setStatus(v,-1) 
 self.solutionGraph[v]=childNodeList # update the solution graph with the solved nodes which may be a part of 
solution 
 
 
 if v!=self.start: # check the current node is the start node for backtracking the current node value 
 self.aoStar(self.parent[v], True) # backtracking the current node value with backtracking status set to true
 
 if backTracking==False: # check the current call is not for backtracking
 for childNode in childNodeList: # for each Minimum Cost child node
 self.setStatus(childNode,0) # set the status of child node to 0(needs exploration)
 self.aoStar(childNode, False) # Minimum Cost child node is further explored with backtracking 
status as false
 
 
h1 = {'A': 1, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 7, 'I': 7, 'J': 1, 'T': 3}
graph1 = {
 'A': [[('B', 1), ('C', 1)], [('D', 1)]],
 'B': [[('G', 1)], [('H', 1)]],
 'C': [[('J', 1)]],
 'D': [[('E', 1), ('F', 1)]],
 'G': [[('I', 1)]] 
}
G1= Graph(graph1, h1, 'A')
G1.applyAOStar() 
G1.printSolution()
h2 = {'A': 1, 'B': 6, 'C': 12, 'D': 10, 'E': 4, 'F': 4, 'G': 5, 'H': 7} # Heuristic values of Nodes
graph2 = { # Graph of Nodes and Edges
 'A': [[('B', 1), ('C', 1)], [('D', 1)]], # Neighbors of Node 'A', B, C & D with repective weights 
 'B': [[('G', 1)], [('H', 1)]], # Neighbors are included in a list of lists
 'D': [[('E', 1), ('F', 1)]] # Each sublist indicate a "OR" node or "AND" nodes
}
G2 = Graph(graph2, h2, 'A') # Instantiate Graph object with graph, heuristic values and start 
Node
G2.applyAOStar() # Run the AO* algorithm
G2.printSolution() # Print the solution graph as output of the AO* algorithm search
6. Result/Output:
HEURISTIC VALUES : {'A': 1, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 7, 
'I': 7, 'J': 1, 'T': 3}
SOLUTION GRAPH : {}
PROCESSING NODE : A
--------------------------------------------------------------------------------------
HEURISTIC VALUES : {'A': 10, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 7, 
'I': 7, 'J': 1, 'T': 3}
SOLUTION GRAPH : {}
PROCESSING NODE : B
--------------------------------------------------------------------------------------
HEURISTIC VALUES : {'A': 10, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 7, 
'I': 7, 'J': 1, 'T': 3}
SOLUTION GRAPH : {}
PROCESSING NODE : A
--------------------------------------------------------------------------------------
HEURISTIC VALUES : {'A': 10, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 7, 
'I': 7, 'J': 1, 'T': 3}
SOLUTION GRAPH : {}
PROCESSING NODE : G
--------------------------------------------------------------------------------------
HEURISTIC VALUES : {'A': 10, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 8, 'H': 7, 
'I': 7, 'J': 1, 'T': 3}
SOLUTION GRAPH : {}
PROCESSING NODE : B
--------------------------------------------------------------------------------------
HEURISTIC VALUES : {'A': 10, 'B': 8, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 8, 'H': 7, 
'I': 7, 'J': 1, 'T': 3}
SOLUTION GRAPH : {}
PROCESSING NODE : A
--------------------------------------------------------------------------------------
HEURISTIC VALUES : {'A': 12, 'B': 8, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 8, 'H': 7, 
'I': 7, 'J': 1, 'T': 3}
SOLUTION GRAPH : {}



3) Candidate eimination
import csv
a=[]
with open("enjoysport.csv","r") as csvfile:
 fdata=csv.reader(csvfile)
 for row in fdata:
 a.append(row)
 print(row)
num_att=len(a[0])-1
S=['0']*num_att
G=['?']*num_att
print(S)
print(G)
temp=[]
for i in range(0,num_att):
 S[i]=a[0][i]
print("................................................")
for i in range(0,len(a)):
if a[i][num_att]=="Yes":
 for j in range(0,num_att):
 if S[j]!=a[i][j]:
 S[j]='?'
 for j in range(0,num_att):
 for k in range(0,len(temp)):
 if temp[k][j]!=S[j] and temp[k][j]!='?':
 del temp[k]
 if a[i][num_att]=='No':
 for j in range(0,num_att):
 if a[i][j]!=S[j] and S[j]!='?':
 G[j]=S[j]
 temp.append(G)
 G=['?']*num_att
 print(S)
 if len(temp)==0:
 print(G)
 else: 
 print(temp)
 print("......................................................................")


4)  ID3 algo
import pandas as pd
import math
df = pd.read_csv('/Users/Documents/Python Scripts/PlayTennis.csv')
print("\n Input Data Set is:\n", df)
t = df.keys()[-1]
print('Target Attribute is: ', t)
# Get the attribute names from input dataset
attribute_names = list(df.keys())
#Remove the target attribute from the attribute names list
attribute_names.remove(t) 
print('Predicting Attributes: ', attribute_names)
#Function to calculate the entropy of collection S
def entropy(probs): 
return sum( [-prob*math.log(prob, 2) for prob in probs])
#Function to calulate the entropy of the given Data Sets/List with 
#respect to target attributes
def entropy_of_list(ls,value): 
 from collections import Counter
 cnt = Counter(x for x in ls)# Counter calculates the propotion of class
 print('Target attribute class count(Yes/No)=',dict(cnt))
 total_instances = len(ls) 
 print("Total no of instances/records associated with {0} is: {1}".format(value,total_instances ))
 probs = [x / total_instances for x in cnt.values()] # x means no of YES/NO
 print("Probability of Class {0} is: {1:.4f}".format(min(cnt),min(probs)))
 print("Probability of Class {0} is: {1:.4f}".format(max(cnt),max(probs)))
 return entropy(probs) # Call Entropy 
def information_gain(df, split_attribute, target_attribute,battr):
 print("\n\n-----Information Gain Calculation of ",split_attribute, " --------") 
 df_split = df.groupby(split_attribute) # group the data based on attribute values
 glist=[]
 for gname,group in df_split:
 print('Grouped Attribute Values \n',group)
 glist.append(gname) 
 
 glist.reverse()
 nobs = len(df.index) * 1.0 
 df_agg1=df_split.agg({target_attribute:lambda x:entropy_of_list(x, glist.pop())})
 df_agg2=df_split.agg({target_attribute :lambda x:len(x)/nobs})
 
 df_agg1.columns=['Entropy']
 df_agg2.columns=['Proportion']
 
 # Calculate Information Gain:
 new_entropy = sum( df_agg1['Entropy'] * df_agg2['Proportion'])
 if battr !='S':
 old_entropy = entropy_of_list(df[target_attribute],'S-'+df.iloc[0][df.columns.get_loc(battr)])
 else:
 old_entropy = entropy_of_list(df[target_attribute],battr)
 return old_entropy - new_entropy
def id3(df, target_attribute, attribute_names, default_class=None,default_attr='S'):
 
 from collections import Counter
 cnt = Counter(x for x in df[target_attribute])# class of YES /NO
 
 ## First check: Is this split of the dataset homogeneous?
 if len(cnt) == 1:
 return next(iter(cnt)) # next input data set, or raises StopIteration when EOF is hit.
 
 ## Second check: Is this split of the dataset empty? if yes, return a default value
 elif df.empty or (not attribute_names):
 return default_class # Return None for Empty Data Set
 
 ## Otherwise: This dataset is ready to be devied up!
else:
 # Get Default Value for next recursive call of this function:
 default_class = max(cnt.keys()) #No of YES and NO Class
 # Compute the Information Gain of the attributes:
 gainz=[]
 for attr in attribute_names:
 ig= information_gain(df, attr, target_attribute,default_attr)
 gainz.append(ig)
 print('Information gain of ',attr,' is : ',ig)
 
 index_of_max = gainz.index(max(gainz)) 
 best_attr = attribute_names[index_of_max
 print("\nAttribute with the maximum gain is: ", best_attr)
 # Create an empty tree, to be populated in a moment
 tree = {best_attr:{}} # Initiate the tree with best attribute as a node 
 remaining_attribute_names =[i for i in attribute_names if i != best_attr]
 
 # Split dataset-On each split, recursively call this algorithm.Populate the empty tree with 
subtrees, which
 # are the result of the recursive call
 for attr_val, data_subset in df.groupby(best_attr):
 subtree = id3(data_subset,target_attribute, 
remaining_attribute_names,default_class,best_attr)
 tree[best_attr][attr_val] = subtree
 return tree
 
 from pprint import pprint
tree = id3(df,t,attribute_names)
print("\nThe Resultant Decision Tree is:")
print(tree)
def classify(instance, tree,default=None): # Instance of Play Tennis with Predicted 
 attribute = next(iter(tree)) # Outlook/Humidity/Wind 
 if instance[attribute] in tree[attribute].keys(): # Value of the attributs in set of Tree keys 
 result = tree[attribute][instance[attribute]]
 if isinstance(result, dict): # this is a tree, delve deeper
 return classify(instance, result)
 else:
 return result # this is a label
 else:
 return default
 
df_new=pd.read_csv('/Users/Documents/Python Scripts/PlayTennisTest.csv')
df_new['predicted'] = df_new.apply(classify, axis=1, args=(tree,'?')) 
print(df_new)


5) BACKPROPAGTION
import numpy as np
X = np.array(([2, 9], [1, 5], [3, 6])) 
y = np.array(([92], [86], [89])) 
y = y/100 
def sigmoid(x): 
 return 1/(1 + np.exp(-x))
def derivatives_sigmoid(x):
 return x * (1 - x)
epoch=10000
lr=0.1 
inputlayer_neurons = 2 
hiddenlayer_neurons = 3 
output_neurons = 1 
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bias_hidden=np.random.uniform(size=(1,hiddenlayer_neurons)) 
weight_hidden=np.random.uniform(size=(hiddenlayer_neurons,output_neurons)) 
bias_output=np.random.uniform(size=(1,output_neurons)) 
for i in range(epoch):
 hinp1=np.dot(X,wh)
 hinp= hinp1 + bias_hidden 
 hlayer_activation = sigmoid(hinp)
 
 outinp1=np.dot(hlayer_activation,weight_hidden)
 outinp= outinp1+ bias_output
 output = sigmoid(outinp)
 
 EO = y-output 
 outgrad = derivatives_sigmoid(output) 
 d_output = EO * outgrad 
 EH = d_output.dot(weight_hidden.T) 
 hiddengrad = derivatives_sigmoid(hlayer_activation) 
 d_hiddenlayer = EH * hiddengrad
 weight_hidden += hlayer_activation.T.dot(d_output) *lr
 bias_hidden += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr
 wh += X.T.dot(d_hiddenlayer) *lr
 bias_output += np.sum(d_output, axis=0,keepdims=True) *lr
print("Input: \n" + str(X))
print("Actual Output: \n" + str(y))
print("Predicted Output: \n" ,output)

6) NAIVE BAYESIAN
import numpy as np
import math
import csv
import pdb
def read_data(filename):
 with open(filename,'r') as csvfile:
 datareader = csv.reader(csvfile)
 metadata = next(datareader)
 traindata=[]
 for row in datareader:
 traindata.append(row)
 return (metadata, traindata)
def splitDataset(dataset, splitRatio):
 trainSize = int(len(dataset) * splitRatio)
 trainSet = []
 testset = list(dataset)
 i=0
 while len(trainSet) < trainSize:
 trainSet.append(testset.pop(i))
 return [trainSet, testset]
def classify(data,test):
 total_size = data.shape[0]
 print("\n")
 print("training data size=",total_size)
 print("test data size=",test.shape[0])
 countYes = 0
 countNo = 0
 probYes = 0
 probNo = 0
 print("\n")
 print("target count probability")
 for x in range(data.shape[0]):
 if data[x,data.shape[1]-1] == 'yes':
 countYes +=1
 if data[x,data.shape[1]-1] == 'no':
 countNo +=1
 probYes=countYes/total_size
 probNo= countNo / total_size
 print('Yes',"\t",countYes,"\t",probYes)
 print('No',"\t",countNo,"\t",probNo)
 prob0 =np.zeros((test.shape[1]-1))
 prob1 =np.zeros((test.shape[1]-1))
 accuracy=0
 print("\n")
 print("instance prediction target")
 for t in range(test.shape[0]):
 for k in range (test.shape[1]-1):
 count1=count0=0
 for j in range (data.shape[0]):
 #how many times appeared with no
 if test[t,k] == data[j,k] and data[j,data.shape[1]-1]=='no':
 count0+=1
 #how many times appeared with yes
 if test[t,k]==data[j,k] and data[j,data.shape[1]-1]=='yes':
 count1+=1
 prob0[k]=count0/countNo
 prob1[k]=count1/countYes
 probno=probNo
 probyes=probYes
 for i in range(test.shape[1]-1):
 probno=probno*prob0[i]
probyes=probyes*prob1[i]
 if probno>probyes:
 predict='no'
 else:
 predict='yes'
 print(t+1,"\t",predict,"\t ",test[t,test.shape[1]-1])
 if predict == test[t,test.shape[1]-1]:
 accuracy+=1
 final_accuracy=(accuracy/test.shape[0])*100
 print("accuracy",final_accuracy,"%")
 return
metadata,traindata= read_data("/Users/Chachu/Documents/Python Scripts/tennis.csv")
splitRatio=0.6
trainingset, testset=splitDataset(traindata, splitRatio)
training=np.array(trainingset)
print("\n The Training data set are:")
for x in trainingset:
 print(x)
 
testing=np.array(testset)
print("\n The Test data set are:")
for x in testing:
 print(x)
classify(training,testing)


7) CLUSTERING BASED ON EM ALGO
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.cluster import KMeans 
import pandas as pd
import numpy as np
 # import some data to play with 
iris = datasets.load_iris()
X = pd.DataFrame(iris.data)
X.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width'] 
y = pd.DataFrame(iris.target)
y.columns = ['Targets']
# Build the K Means Model
model = KMeans(n_clusters=3)
model.fit(X) # model.labels_ : Gives cluster no for which samples belongs to
# # Visualise the clustering results 
plt.figure(figsize=(14,14))
colormap = np.array(['red', 'lime', 'black'])
# Plot the Original Classifications using Petal features 
plt.subplot(2, 2, 1)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y.Targets], s=40) 
plt.title('Real Clusters')
plt.xlabel('Petal Length') 
plt.ylabel('Petal Width')
# Plot the Models Classifications
plt.subplot(2, 2, 2)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[model.labels_], s=40) 
plt.title('K-Means Clustering')
plt.xlabel('Petal Length') 
plt.ylabel('Petal Width')
# General EM for GMM
from sklearn import preprocessing
# transform your data such that its distribution will have a # mean value 0 and standard 
deviation of 1.
scaler = preprocessing.StandardScaler() 
scaler.fit(X)
xsa = scaler.transform(X)
xs = pd.DataFrame(xsa, columns = X.columns)
from sklearn.mixture import GaussianMixture 
gmm = GaussianMixture(n_components=3) 
gmm.fit(xs)
gmm_y = gmm.predict(xs)
plt.subplot(2, 2, 3)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[gmm_y], s=40)
plt.title('GMM Clustering')
plt.xlabel('Petal Length') 
plt.ylabel('Petal Width')
print('Observation: The GMM using EM algorithm based clustering matched the true labels 
more closely than the Kmeans.')

8) K-NEAREST NEIGHBOUR
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import datasets
# Load dataset 
iris=datasets.load_iris() 
print("Iris Data set loaded...")
# Split the data into train and test samples
x_train, x_test, y_train, y_test = train_test_split(iris.data,iris.target,test_size=0.1) 
print("Dataset is split into training and testing...")
print("Size of trainng data and its label",x_train.shape,y_train.shape) 
print("Size of trainng data and its label",x_test.shape, y_test.shape)
# Prints Label no. and their names 
for i in range(len(iris.target_names)):
 print("Label", i , "-",str(iris.target_names[i]))
 # Create object of KNN classifier
classifier = KNeighborsClassifier(n_neighbors=1)
# Perform Training 
classifier.fit(x_train, y_train) # Perform testing
y_pred=classifier.predict(x_test)
# Display the results
print("Results of Classification using K-nn with K=1 ") 
for r in range(0,len(x_test)):
 print(" Sample:", str(x_test[r]), " Actual-label:", str(y_test[r]), " Predicted-label:", str(y_pred[r]))
print("Classification Accuracy :" , classifier.score(x_test,y_test));
from sklearn.metrics import classification_report, confusion_matrix
print('Confusion Matrix')
print(confusion_matrix(y_test,y_pred)) 
print('Accuracy Metrics') 
print(classification_report(y_test,y_pred))


9) LOCALLY WEIGHTED REGRESSION ALGORITHM
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
def kernel(point,xmat, k): 
 m,n = np.shape(xmat)
 weights = np.mat(np.eye((m))) # eye - identity matrix 
 for j in range(m):
 diff = point - X[j]
 weights[j,j] = np.exp(diff*diff.T/(-2.0*k**2)) 
 return weights
def localWeight(point,xmat,ymat,k): 
 wei = kernel(point,xmat,k)
 W = (X.T*(wei*X)).I*(X.T*(wei*ymat.T)) 
 return W
def localWeightRegression(xmat,ymat,k): 
 m,n = np.shape(xmat)
 ypred = np.zeros(m) 
 for i in range(m):
 ypred[i] = xmat[i]*localWeight(xmat[i],xmat,ymat,k) 
 return ypred
def graphPlot(X,ypred):
 sortindex = X[:,1].argsort(0) #argsort - index of the smallest 
 xsort = X[sortindex][:,0]
fig = plt.figure()
 ax = fig.add_subplot(1,1,1) 
 ax.scatter(bill,tip, color='green')
 ax.plot(xsort[:,1],ypred[sortindex], color = 'red', linewidth=5) 
 plt.xlabel('Total bill')
 plt.ylabel('Tip') 
 plt.show();
# load data points
data = pd.read_csv('/Users/Chachu/Documents/Python Scripts/data10_tips.csv')
bill = np.array(data.total_bill) # We use only Bill amount and Tips data 
tip = np.array(data.tip)
mbill = np.mat(bill) # .mat will convert nd array is converted in 2D array 
mtip = np.mat(tip)
m= np.shape(mbill)[1] 
one = np.mat(np.ones(m))
X = np.hstack((one.T,mbill.T)) # 244 rows, 2 cols
ypred = localWeightRegression(X,mtip,2) # increase k to get smooth curves 
graphPlot(X,ypred