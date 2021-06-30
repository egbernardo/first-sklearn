from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier #classificador ruim

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

# LOAD DATA
dSet = datasets.load_iris() #datasets.load tem outros exemplos
print(dSet.keys())
print(dSet.DESCR)

data = dSet.data
target = dSet.target
tgNames = dSet.target_names
# print(tgNames)

# VIEW DATA
print('DATA')
print(data.shape)
print()

print('TARGET')
print(target.shape)
print()

# SPLIT - TEST & TRAIN
XTrain, XTest, yTrain, yTest = train_test_split(data, target, test_size=0.7, random_state=42)

#CLASSIFIERS
cls_decTree = DecisionTreeClassifier()
cls_naive = GaussianNB()
cls_logreg = LogisticRegression(C=1, solver='lbfgs')
cls_knn = KNeighborsClassifier(n_neighbors=5)
cls_mlp = MLPClassifier(hidden_layer_sizes=2)
cls_dm = DummyClassifier()

# PREPROCESSING - escolher um para rodar o código corretamente
#pipe_cls = make_pipeline(StandardScaler(), cls_decTree)
#pipe_cls = make_pipeline(StandardScaler(), cls_naive)
pipe_cls = make_pipeline(StandardScaler(), cls_logreg) #--------funcionando-------
#pipe_cls = make_pipeline(StandardScaler(), cls_knn)
#pipe_cls = make_pipeline(StandardScaler(), cls_mlp)
#pipe_cls = cls_dm

# CLASSIFICATION
pipe_cls.fit(XTrain, yTrain)

# PREDICT
predict = pipe_cls.predict(XTest)
print('PREDICT')
print(predict)
print()

# RELEVANCE
print('Report')
# print(classification_report(yTest, predict, zero_division=0))
print(classification_report(yTest, predict, target_names=tgNames, zero_division=0))

print('Confusion Matrix')
print(confusion_matrix(yTest, predict))
print()

# TUNNING
scores = {'accuracy', 'precision_micro', 'recall_micro', 'f1_micro'}
cv = cross_validate(pipe_cls, data, target, cv=4, scoring = scores)

print('Accuracy: ', cv['test_accuracy'])
print('Precision: ', cv['test_precision_micro'])
print('Recall: ', cv['test_recall_micro'])
print('F1-score: ', cv['test_f1_micro'])

# AFTER TRAINING
print(cls_logreg.get_params().keys())
print(pipe_cls.get_params().keys())

print(pipe_cls)
params = {'logisticregression__C': [0.5, 1, 0.2], # é possível explorar outros parâmetros e decidir o melhor conforme a análise da saida
          'logisticregression__fit_intercept': [True, False],
          'logisticregression__solver': ['newton-cg', 'saga', 'sag','lbfgs']}

gs = GridSearchCV(estimator = pipe_cls, param_grid = params, cv = 3, scoring ='accuracy')
gs.fit(data, target)

print(gs.best_params_)
print(gs.best_score_)
