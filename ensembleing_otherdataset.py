
import numpy as np
import string

import pickle
import time
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
from optimizers import GeneticOptimizer
from optimizers import get_individual_score
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import VotingClassifier
import scikitplot as skplt
#import decisionTree

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
#import BankingDataAnalysis
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import sys
from sklearn.metrics import r2_score
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingClassifier  # For Classification
import seaborn as sns
import numpy
import math
from scipy import stats

numpy.set_printoptions(threshold=sys.maxsize)

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 2000)
from sklearn.datasets.samples_generator import make_blobs
if __name__ == "__main__":


    X, y = make_blobs(n_samples=50, centers=2,
                      random_state=0, cluster_std=0.60)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn');

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state = 50)

    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test)
    print("y_test",y_test.shape)
    print("y_test", y_test.shape)
    X_test= X_test
    X_val =X_val
    y_test=y_test
    y_val=y_val
    n_estimators = len(y_test)
    pop_size = n_estimators // 2
    iterations = 2
    mutation_rate = 0.5
    crossover_rate = 0.75
    n_jobs = 8
    elitism = True
    n_point_crossover = False
    max_samples_ratio = 0.5

    print("\nGenerating estimators from Bagging method...")
    # bagging = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=n_estimators, max_samples=max_samples_ratio)
    bagging = BaggingClassifier(base_estimator=RandomForestClassifier(), n_estimators=n_estimators,
                                max_samples=max_samples_ratio)
    bagging.fit(X_train, y_train)
    predictions = bagging.predict(X_test)
    print("Bagging Score:", bagging.score)
    # print("predicition:",predictions)
    skplt.metrics.plot_confusion_matrix(
     y_test,
        predictions,
        figsize=(10, 6), title="Confusion matrix\n Deposite Category of Bagging Classifier")
    plt.xlim(-0.5, len(np.unique(y_test)) - 0.5)
    plt.ylim(len(np.unique(y_test)) - 0.5, -0.5)
    plt.show()

    val_initial_score = bagging.score(X_test, y_test)
    print(" Bagging Score : %f%%" % (val_initial_score))
    val_initial_score = np.asarray(val_initial_score)
    print("val_init", val_initial_score.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)
    #print("y_test:", y_test.shape)

    print("\n\n=======================================================================")
    print("Bagging Results")
    print("=======================================================================")

    rms = np.sqrt(np.mean(np.power((np.array(y_test) - np.array(predictions)), 2)))
    score = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    pearson_coef, p_value = stats.pearsonr(y_test, predictions)

    print("root mean square:", rms)
    print("score:", score)
    print("mean absolute error:", mae)
    print("mean squared error:", mse)
    print("pearson_coef:", pearson_coef)
    print("p_value:", p_value)
    ns_probs = [0 for _ in range(len(y_test))]
    lr_probs = predictions
    # calculate scores
    #ns_auc = roc_auc_score(y_test, ns_probs)
    #lr_auc = roc_auc_score(y_test, predictions)
    # summarize scores
    #print('No Skill: ROC AUC=%.3f' % (ns_auc))
    #print('Logistic: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()

    gen_opt = GeneticOptimizer(estimators=bagging.estimators_,
                               classes=bagging.classes_,
                               data=X_test,
                               target=y_test,
                               val=(val_initial_score, X_val, y_val),
                               pop_size=pop_size,
                               mutation_rate=mutation_rate,
                               crossover_rate=crossover_rate,
                               iterations=iterations,
                               elitism=elitism,
                               n_point_crossover=n_point_crossover,
                               n_jobs=n_jobs,
                               temp=y_val,temp2=y_test)

    best_found, test_initial_score = gen_opt.run_genetic_evolution()

    print()

    print("Best individual score found: %f%% (Gain: %f%%)" % (best_found[0] * 100, (best_found[0] - test_initial_score) * 100))
    print("Estimators combination for the best score:")
    print(best_found)
    print("Number of estimators: %d" % (len([estimator for estimator in best_found[1] if estimator])))

    print("\nTesting best combination on validation set...")
    final_score = get_individual_score(best_found[1], bagging.estimators_, X_test, y_test, bagging.classes_)
    print("Final score: %f%% (Gain: %f%%)" % (final_score * 100, (final_score - val_initial_score) * 100))

    filename = 'optimized_model_%d.ens' % int(time.time())
    pickle.dump((bagging, best_found[1]), open(filename, 'wb'))
    print("\nSaved optimized model as [%s]" % filename)

    print("\n================================================================================")
    print("\nGenerating estimators from Bagging method...")
    print("Best Bagging", len(best_found[1]))
    bagging.fit(X_train, y_train)
    gpredictions = bagging.predict(X_test)
    print("Bagging Score:", bagging.score)
    # print("predicition:",predictions)
    skplt.metrics.plot_confusion_matrix(
        y_test,
        gpredictions,
        figsize=(10, 6), title="Confusion matrix\n Deposite Category of Bagging Classifier")
    plt.xlim(-0.5, len(np.unique(y_test)) - 0.5)
    plt.ylim(len(np.unique(y_test)) - 0.5, -0.5)
    plt.show()

    val_initial_score = bagging.score(X_test, y_test)
    print(" Bagging Score : %f%%" % (val_initial_score))
    val_initial_score = np.asarray(val_initial_score)
    print("val_init", val_initial_score.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)
    # print("y_test:", y_test.shape)

    print("\n\n=======================================================================")
    print("Bagging Results")
    print("=======================================================================")

    rms = np.sqrt(np.mean(np.power((np.array(y_test) - np.array(gpredictions)), 2)))
    score = r2_score(y_test, gpredictions)
    mae = mean_absolute_error(y_test, gpredictions)
    mse = mean_squared_error(y_test, gpredictions)
    pearson_coef, p_value = stats.pearsonr(y_test, gpredictions)

    print("root mean square:", rms)
    print("score:", score)
    print("mean absolute error:", mae)
    print("mean squared error:", mse)
    print("pearson_coef:", pearson_coef)
    print("p_value:", p_value)
    ns_probs = [0 for _ in range(len(y_test))]
    lr_probs = gpredictions
    # calculate scores
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc = roc_auc_score(y_test, gpredictions)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()

    model1 = DecisionTreeClassifier()
    model2 = KNeighborsClassifier()
    model3 = LogisticRegression()

    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)
    model3.fit(X_train, y_train)
    pred1 = model1.predict(X_test)
    pred2 = model2.predict(X_test)
    pred3 = model3.predict(X_test)

    final_pred = np.array([])
    for i in range(0, len(X_test)):
        final_pred = np.append(final_pred, stats.mode([pred1[i], pred2[i], pred3[i]]))
    model1 = RandomForestClassifier(random_state=1)
    model2 = DecisionTreeClassifier(random_state=1)
    Voting = VotingClassifier(estimators=[('dt', model1), ('kn', model2),('lr', model3)], voting='hard')
    Voting.fit(X_train, y_train)
    vpredictions = Voting.predict(X_test)

    vscore = Voting.score(X_test, y_test)
    print("Voting Score", vscore)
    # print("predicition:", vpredictions)
    print("\n\n=======================================================================")
    print("Voting Results")
    print("=======================================================================")
    rms = np.sqrt(np.mean(np.power((np.array(y_test) - np.array(vpredictions)), 2)))
    score = r2_score(y_test, vpredictions)
    mae = mean_absolute_error(y_test, vpredictions)
    mse = mean_squared_error(y_test, vpredictions)
    pearson_coef, p_value = stats.pearsonr(y_test, vpredictions)

    print("root mean square:", rms)
    print("score:", score)
    print("mean absolute error:", mae)
    print("mean squared error:", mse)
    print("pearson_coef:", pearson_coef)
    print("p_value:", p_value)
    ns_probs = [0 for _ in range(len(y_test))]
    lr_probs = vpredictions
    # calculate scores
    #ns_auc = roc_auc_score(y_test, ns_probs)
    #lr_auc = roc_auc_score(y_test, vpredictions)
    # summarize scores
    #print('No Skill: ROC AUC=%.3f' % (ns_auc))
    #print('Logistic: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
    print("=======================================================================\n\n")
    skplt.metrics.plot_confusion_matrix(
     y_test,
        vpredictions,
        figsize=(10, 6), title="Confusion matrix\n Deposite Category Voting Classifier")
    plt.xlim(-0.5, len(np.unique(y_test)) - 0.5)
    plt.ylim(len(np.unique(y_test)) - 0.5, -0.5)
    plt.show()
    print("=======================================================================\n\n")
    #    params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9]}
    #    knn = neighbors.KNeighborsRegressor()
    #    model = GridSearchCV(knn, params, cv=5)
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)
    clf.fit(X_train, y_train)
    cpredictions = clf.predict(X_test)
    vscore = clf.score(X_test, y_test)
    print(" Gradient Boosting Score", vscore)
    # print("predicition:", vpredictions)
    print("\n\n=======================================================================")
    print(" Gradient Boosting Results")
    print("=======================================================================")
    rms = np.sqrt(np.mean(np.power((np.array(y_test) - np.array(cpredictions)), 2)))
    score = r2_score(y_test, cpredictions)
    mae = mean_absolute_error(y_test, cpredictions)
    mse = mean_squared_error(y_test, cpredictions)
    pearson_coef, p_value = stats.pearsonr(y_test, cpredictions)

    print("root mean square:", rms)
    print("score:", score)
    print("mean absolute error:", mae)
    print("mean squared error:", mse)
    print("pearson_coef:", pearson_coef)
    print("p_value:", p_value)
    ns_probs = [0 for _ in range(len(y_test))]
    lr_probs = cpredictions
    # calculate scores
    #ns_auc = roc_auc_score(y_test, ns_probs)
    #lr_auc = roc_auc_score(y_test, cpredictions)
    # summarize scores
    #print('No Skill: ROC AUC=%.3f' % (ns_auc))
    #print('Logistic: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
    print("=======================================================================\n\n")
    skplt.metrics.plot_confusion_matrix(
     y_test,
        cpredictions,
        figsize=(10, 6), title="Confusion matrix\n Deposite Category Voting Classifier")
    plt.xlim(-0.5, len(np.unique(y_test)) - 0.5)
    plt.ylim(len(np.unique(y_test)) - 0.5, -0.5)
    plt.show()
    # fit the model and make predictions
    # a=model.score(X_testid,y_testid)
    # print(" K nearest Neighbor Score",a)
    #    preds = model.predict(X_test)
    HVoting = VotingClassifier(estimators=[('vc', Voting), ('bc', bagging), ('gb', clf)], voting='hard')
    HVoting.fit(X_train, y_train)
    hvpredictions = HVoting.predict(X_test)
    vscore = HVoting.score(X_test, y_test)
    print("Voting Score", vscore)
    # print("predicition:", vpredictions)
    print("\n\n=======================================================================")
    print(" Level 2 Voting Results")
    print("=======================================================================")
    rms = np.sqrt(np.mean(np.power((np.array(y_test) - np.array(hvpredictions)), 2)))
    score = r2_score(y_test, hvpredictions)
    mae = mean_absolute_error(y_test, hvpredictions)
    mse = mean_squared_error(y_test, hvpredictions)
    pearson_coef, p_value = stats.pearsonr(y_test, hvpredictions)

    print("root mean square:", rms)
    print("score:", score)
    print("mean absolute error:", mae)
    print("mean squared error:", mse)
    print("pearson_coef:", pearson_coef)
    print("p_value:", p_value)
    ns_probs = [0 for _ in range(len(y_test))]
    lr_probs = hvpredictions
    # calculate scores
    #ns_auc = roc_auc_score(y_test, ns_probs)
    #lr_auc = roc_auc_score(y_test, hvpredictions)
    # summarize scores
    #print('No Skill: ROC AUC=%.3f' % (ns_auc))
    #print('Logistic: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
    print("=======================================================================\n\n")
    skplt.metrics.plot_confusion_matrix(
     y_test,
        vpredictions,
        figsize=(10, 6), title="Confusion matrix\n Deposite Category Voting Classifier")
    plt.xlim(-0.5, len(np.unique(y_test)) - 0.5)
    plt.ylim(len(np.unique(y_test)) - 0.5, -0.5)
    plt.show()
    retable = list()
    resulttable = list()
    for i in range(len(y_test)):
        a = [pred1[i], pred2[i], pred3[i], vpredictions[i], predictions[i], cpredictions[i], hvpredictions[i],
             gpredictions[i]]
        # print("Result Vector",a)
        retable.append(np.sum(a) / 9)
        #print("retable", retable[i])
        if retable[i] < 0.5:
            resulttable.append(0)
        else:
            resulttable.append(1)

    print("result table")
    print(resulttable)
    print("\n\n=======================================================================")
    print(" Level 3 Voting Results")
    print("=======================================================================")
    rms = np.sqrt(np.mean(np.power((np.array(y_test) - np.array(resulttable)), 2)))
    score = r2_score(y_test, resulttable)
    mae = mean_absolute_error(y_test, resulttable)
    mse = mean_squared_error(y_test, resulttable)
    pearson_coef, p_value = stats.pearsonr(y_test, resulttable)

    print("root mean square:", rms)
    print("score:", score)
    print("mean absolute error:", mae)
    print("mean squared error:", mse)
    print("pearson_coef:", pearson_coef)
    print("p_value:", p_value)
    ns_probs = [0 for _ in range(len(y_test))]
    lr_probs = resulttable
    # calculate scores
    #ns_auc = roc_auc_score(y_test, ns_probs)
    #lr_auc = roc_auc_score(y_test, resulttable)
    # summarize scores
    #print('No Skill: ROC AUC=%.3f' % (ns_auc))
    #print('Logistic: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
    print("=======================================================================\n\n")
    skplt.metrics.plot_confusion_matrix(
        y_test,
        resulttable,
        figsize=(10, 6), title="Confusion matrix\n Deposite Category Voting Classifier")
    plt.xlim(-0.5, len(np.unique(y_test)) - 0.5)
    plt.ylim(len(np.unique(y_test)) - 0.5, -0.5)
    plt.show()

    # Bagging
    ns_probs = [0 for _ in range(len(y_test))]
    #lr_probs = predictions
    #best_found_fina=individual_list
    ns_aucb = roc_auc_score(y_test, ns_probs)
    lr_aucb = roc_auc_score(y_test, predictions)
    lr_aucv = roc_auc_score(y_test, vpredictions)
    lr_aucc = roc_auc_score(y_test, cpredictions)
    lr_auchv = roc_auc_score(y_test, hvpredictions)
    lr_aucg = roc_auc_score(y_test, gpredictions)
    lr_auchb = roc_auc_score(y_test, resulttable)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_aucb))
    print('Bagging Logistic: ROC AUC=%.3f' % (lr_aucb))
    print('Voting Logistic: ROC AUC=%.3f' % (lr_aucv))
    print('Gradient Boosting Logistic: ROC AUC=%.3f' % (lr_aucc))
    print('Bagging with Evalutionary Genetic algorithm Logistic: ROC AUC=%.3f' % (lr_aucg))
    print('Cross Emsemble Voting Logistic: ROC AUC=%.3f' % (lr_auchv))
    print('Hybrid Evalutionary Ensemble Logistic: ROC AUC=%.3f' % (lr_auchb))

    # calculate roc curves
    ns_fprb, ns_tprb, _ = roc_curve(y_test, ns_probs)
    lr_fprb, lr_tprb, _ = roc_curve(y_test, predictions)
    lr_fprv, lr_tprv, _ = roc_curve(y_test, vpredictions)
    lr_fprc, lr_tprc, _ = roc_curve(y_test, cpredictions)
    lr_fprhv, lr_tprhv, _ = roc_curve(y_test, hvpredictions)
    lr_fprg, lr_tprg, _ = roc_curve(y_test, gpredictions)
    lr_fprhb, lr_tprhb, _ = roc_curve(y_test, resulttable)
    # plot the roc curve for the model
    plt.plot(ns_fprb, ns_tprb, linestyle='--', label='No Skill')
    plt.plot(lr_fprb, lr_tprb, marker='.', label='Bagging')
    plt.plot(lr_fprv, lr_tprv, marker='.', label='Voting')
    plt.plot(lr_fprc, lr_tprc, marker='.', label='Gradient Boosting')
    plt.plot(lr_fprhv, lr_tprhv, marker='.', label='Cross Ensemble')
    plt.plot(lr_fprg, lr_tprg, marker='.', label='Bagging Evalutionary Genetic')
    plt.plot(lr_fprhb, lr_tprhb, marker='.', label='Hybrid Evalutionary Ensemble')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    plt.title("Result Comparison Graph")
    # show the plot
    plt.show()

