# AUTHOR: ALEXIS CARBILLET

# import librairies
import pickle 
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import pandas

db = {} 
# Its important to use binary mode 
dbfile = open('C:/Users/alexis/Desktop/image processing/lab 2/vehicleimg', 'rb') 
new_dict = pickle.load(dbfile)          
print(new_dict.shape)        # (13785, 186, 250, 3)
dbfile.close() 

db = {} 
# Its important to use binary mode 
dbfile = open('C:/Users/alexis/Desktop/image processing/lab 2/vehicletrgt', 'rb') 
labels = pickle.load(dbfile)          
print(labels.shape)          # (13785)
dbfile.close() 

## extract image descriptors

# we will use Oriented FAST and Rotated BRIEF (ORB)
orb = cv2.ORB_create() 
key_points, description = orb.detectAndCompute(new_dict[0], None)
img_building_keypoints = cv2.drawKeypoints(new_dict[0],key_points, new_dict[0],  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # Draw circles.

plt.figure(figsize=(16, 16))
plt.title('ORB Interest Points')
plt.imshow(img_building_keypoints)
plt.show()

## dimensionnality reduction : pca & svd
# first method: pca

pca_dims = PCA()
x=pca_dims.fit(description)
cumsum = np.cumsum(pca_dims.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1 # we get d=27 

pca = PCA(n_components=d)
X_reduced = pca.fit_transform(description)
X_recovered = pca.inverse_transform(X_reduced)
plt.figure(figsize=(16, 16))
plt.title('PCA')
plt.imshow(X_recovered)
plt.show()


# second method: svd
svd = TruncatedSVD(n_components=1)
X_reduced = svd.fit_transform(description)
X_restored = svd.inverse_transform(X_reduced)
plt.figure(figsize=(16, 16))
plt.title('SVD')
plt.imshow(X_restored)
plt.show()


# m=len(X_reduced)
# for i in range(1, len(new_dict)):
#     key_points, description = orb.detectAndCompute(new_dict[i], None)
#     X_reduced = svd.fit_transform(description)
#     if(len(X_reduced)>m): # we need to know the maximum number of columns. Then we will fill it with NaN and try to replace it with some technics
#         m=len(X_reduced)
# print(m) # m=418

# m=418
# key_points, description = orb.detectAndCompute(new_dict[0], None)
# svd = TruncatedSVD(n_components=1)
# X_reduced = svd.fit_transform(description)
# data = pandas.DataFrame(index=np.arange(13785) ,columns=np.arange(418))
# 
# for i in range(len(new_dict)):
#     key_points, description = orb.detectAndCompute(new_dict[i], None)
#     X_reduced = svd.fit_transform(description)
#     X_reduced = np.concatenate([X_reduced.transpose(), np.array([[0]*(m-len(X_reduced))])],axis=1) # fill with 0 the empty columns => we can't put None because the type is incompatible with int
#     data.loc[i] = X_reduced
#     print(i)

## save results in a dataset
# data['Labels'] = pandas.Series(labels, index=data.index)
# data.to_csv('project.csv')
new_dict = pandas.read_csv('project.csv', sep=',')
new_dict.drop(new_dict.columns[len(new_dict.columns)-1], axis=1, inplace=True)
print(new_dict.shape)
## machine learning
def fit(nb,train,test,y,yt,height_f1,type):
    nb.fit(train, y)
    z=f1_score(yt, nb.predict(test),average='weighted')
    print('the f1 score obtained with ',type,' is:',z)
    height_f1.append(z)

def ml(train,test,y,yt):
    height=[]
    height_f1=[]
    bars=['bayes','perceptron','MLP','tree','logistic regression','kNN 3 neighbors','kNN 7 neighbors','kNN 15 neighbors','SVC','Random Forest']
    # bayes
    nb = MultinomialNB()
    fit(nb,train,test,y,yt,height_f1,'bayes')
    # perceptron
    nb = Perceptron(tol=1e-3, random_state=0)
    fit(nb,train,test,y,yt,height_f1,'perceptron')
    # multi-layer perceptron
    nb = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    fit(nb,train,test,y,yt,height_f1,'multi-layer perceptron')
    # tree classifier
    nb = DecisionTreeClassifier(random_state=0)
    fit(nb,train,test,y,yt,height_f1,'tree')
    # logistic regression
    nb = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
    fit(nb,train,test,y,yt,height_f1,'logistic regression')
    # kNN 3
    nb = KNeighborsClassifier(n_neighbors=3)
    fit(nb,train,test,y,yt,height_f1,'kNN 3 neighbors')
    # kNN 7
    nb = KNeighborsClassifier(n_neighbors=7)
    fit(nb,train,test,y,yt,height_f1,'kNN 7 neighbors')
    # kNN 15
    nb = KNeighborsClassifier(n_neighbors=15)
    fit(nb,train,test,y,yt,height_f1,'kNN 15 neighbors')
    # SVC
    nb = SVC(gamma='auto')
    fit(nb,train,test,y,yt,height_f1,'SVC')
    # random forest
    nb = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    fit(nb,train,test,y,yt,height_f1,'random forest')
    y_pos = np.arange(len(bars))
    plt.figure()

    title='F1 score'
    plt.title(title)
    plt.bar(y_pos, height_f1)  # Create bars
    plt.xticks(y_pos, bars, rotation=90) # Create names on the x-axis
    plt.subplots_adjust(bottom=0.3, top=0.95) # Custom the subplot layout
    plt.show()    # Show graphic
    print('the best one  is ',bars[height_f1.index(max(height_f1))],' with a F1 score of ',height_f1[height_f1.index(max(height_f1))])

X_train, X_test, y_train, y_test = train_test_split(new_dict, labels, test_size=0.20, random_state=42)
ml(X_train,X_test,y_train,y_test)