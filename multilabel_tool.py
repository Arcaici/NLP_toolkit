#---------------------------------------------- GRIDSEARCH ----------------------------------------------#

from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset, ClassifierChain
from skmultilearn.ensemble import LabelSpacePartitioningClassifier, MajorityVotingClassifier, RakelD, RakelO
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC

#################### gridsearch of ensemble classifiers ####################
# this function run a grid search over classifier respect to parameters that are passed
# classifier: LabelSpacePartitioningClassifier, MajorityVotingClassifier, RakelD, RakelO
#
# parameters = {
#     classifier: a classifier from skmultilearn.problem_transform: BinaryRelevance, LabelPowerset, ClassifierChain
#     classifier__classifier: Base classifier (SVM(), Logistic. Random Forest, ecc...)
#     classifier__classifier__*: parameter search of * param of classifier__classifier model
#     clusterer: a label space partitioning class.
#}

################################# example #################################
parameters = {
    'classifier': [LabelPowerset(), ClassifierChain()],
    'classifier__classifier': [RandomForestClassifier()],
    'classifier__classifier__n_estimators': [10, 20, 50],
    'clusterer' : [
        NetworkXLabelGraphClusterer(LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False), 'louvain'),
        NetworkXLabelGraphClusterer(LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False), 'lpa')
    ]
}

clf = GridSearchCV(LabelSpacePartitioningClassifier(), parameters, scoring = 'f1_macro')
clf.fit(X_train, y_train)

print (clf.best_params_, clf.best_score_)

#################### gridsearch classifiers ####################
# this function run a grid search over a model or type adapt or perform_trasform,
# respect to parameters that are passed
#
# parameters = {
#     classifier: Base classifier (SVM(), Logistic. Random Forest, ecc...)
#     classifier__*: parameter search of * param of classifier__classifier model
#}

############################ example ############################

parameters = [
    {
        'classifier': [BernoulliNB()],
        'classifier__alpha': [0.7, 1.0],
    },
    {
        'classifier': [SVC()],
        'classifier__kernel': ['rbf', 'linear'],
    },
]

clf = GridSearchCV(BinaryRelevance(), parameters, scoring='accuracy')
clf.fit(X_train, y_train)

print (clf.best_params_, clf.best_score_)

#------------------------------------------------ MODELS ------------------------------------------------#

from skmultilearn.problem_transform import BinaryRelevance
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from skmultilearn.cluster import NetworkXLabelGraphClusterer
from skmultilearn.cluster import LabelCooccurrenceGraphBuilder
from skmultilearn.ensemble import LabelSpacePartitioningClassifier

# Binary Relevance with Random Forest of Decision Trees -> RFDTBR
classifier = BinaryRelevance(
    classifier = RandomForestClassifier(),
    require_dense = [False, True]
)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

# Ensemble of Binary Relevance with Decision Tree
# clustere -> see http://scikit.ml/labelrelations.html#NetworkX for all possible clustere algorithm
classifier = LabelSpacePartitioningClassifier(
    classifier = BinaryRelevance(classifier= DecisionTreeClassifier()),
    clusterer =  NetworkXLabelGraphClusterer(LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False), 'louvain')
)
classifier.fit(X_train,y_train)
predictions = classifier.predict(X_test)

