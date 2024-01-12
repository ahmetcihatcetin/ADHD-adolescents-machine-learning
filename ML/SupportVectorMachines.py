### Import Libraries/Modules
# Data Analysis & Manipulation:
import pandas as pd

# Machine Learning Modules:
from sklearn import svm
from sklearn.model_selection import train_test_split

# Modules for Hyper-parameter Tuning:
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform
from numpy import logspace

# Modules for Performance Metrics:
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Modules for Permutation Feature Importance:
from sklearn.inspection import permutation_importance
import matplotlib.pyplot
import numpy

field_names_for_parent_data = [     "1. Eli boş durmaz, sürekli bir şeylerle (tırnak, parmak, giysi gibi…) oynar.", 
                                    "2. Büyüklere karşı arsız ve küstah davranır.",
                                    "3. Arkadaşlık kurmada ve sürdürmede zorlanır.",
                                    "4. Çabuk heyecanlanır, ataktır.",
                                    "5. Her şeye karışır ve yönetmek ister.",
                                    "6. Bir şeyler çiğner veya emer (parmak, giysi gibi)",
                                    "7. Sık sık ve kolayca ağlar.",
                                    "8. Her an sataşmaya hazırdır.",
                                    "9. Hayallere dalar.",
                                    "10. Zor öğrenir.",
                                    "11. Kıpır kıpırdır, tez canlıdır.",
                                    "12. Ürkektir (yeni durum, insan ve yerlerden).",
                                    "13. Yerinde duramaz, her an harekete hazırdır.",
                                    "14. Zarar verir.",
                                    "15. Yalan söyler, masallar uydurur.",
                                    "16. Utangaçtır.",
                                    "17. Yaşıtlarından daha sık başını derde sokar.",
                                    "18. Yaşıtlarından farklı (çocuksu, zor anlaşılır, kekeleyerek gibi…) konuşur.",
                                    "19. Hatalarını kabullenmez, başkalarını suçlar.",
                                    "20. Kavgacıdır.",
                                    "21. Somurtkan. ve asık suratlıdır.",
                                    "22. Çalma huyu vardır.",
                                    "23. Söz dinlemez ya da isteksiz ve zoraki dinler.",
                                    "24. Başkalarına göre endişelidir (yalnız kalma, hastalanma, ölüm gibi konularda)",
                                    "25. Başladığı bir işin sonunu getiremez.",
                                    "26. Hassastır, kolay incinir.",
                                    "27. Kabadayılık taslar, başkalarını rahatsız eder.",
                                    "28. Tekrarlayıcı durduramadığı hareketleri vardır.",
                                    "29. Kaba ve acımasızdır.",
                                    "30. Yaşına göre daha çocuksudur.",
                                    "31. Dikkati kolay dağılır ya da uzun süre dikkatini toplayamaz.",
                                    "32. Baş ağrıları olur.",
                                    "33. Ruh halinde ani ve göz batan değişiklikler olur.",
                                    "34. Kurallar ve kısıtlamalardan hoşlanmaz ve uymaz.",
                                    "35. Sürekli kavga eder.",
                                    "36. Kardeşleriyle iyi geçinemez.",
                                    "37. Zora gelemez.",
                                    "38. Diğer çocukları rahatsız eder.",
                                    "39. Genelde hoşnutsuz bir çocuktur.",
                                    "40. Yeme sorunları vardır (sofradan sık sık kalkar, iştahsızdır gibi…).",
                                    "41. Karın ağrıları olur.",
                                    "42. Uyku sorunları vardır (uykuya dalamam, erken uyanma, gece kalkma gibi…).",
                                    "43. Çeşitli ağrı ve sancıları olur.",
                                    "44. Bulantı kusmaları olur.",
                                    "45. Aile içinde daha az kayrıldığını düşünür.",
                                    "46. Övünür böbürlenir.",
                                    "47. İtilip kakılmaya müsaittir.",
                                    "48. Dışkılama sorunları vardır (sık ishal, kabızlık, düzensiz tuvalet alışkanlığı gibi…).",
                                    "Label"]

field_names_for_teacher_data = [    "1. Kıpır kıpırdır, yerinde duramaz.",
                                    "2. Zamansız ve uyumsuz sesler çıkarır.",
                                    "3. İstekleri hemen yerine getirilmelidir.",
                                    "4. Bilmiş tavırları vardır, bilgiçlik taslar.",
                                    "5. Aniden parlar, ne yapacağı belli olmaz.",
                                    "6. Eleştiri kaldıramaz.",
                                    "7. Dikkati dağınıktır.",
                                    "8. Diğer çocukları rahatsız eder.",
                                    "9. Hayallere dalar.",
                                    "10. Somurtur, surat asar.",
                                    "11. Bir anı bir anını tutmaz.",
                                    "12. Kavgacıdır.",
                                    "13. Büyüklerin sözünden çıkmaz.",
                                    "14. Hareketlidir, dur otur bilmez.",
                                    "15. Düşünmeden hareket eder.",
                                    "16. Öğretmenin ilgisi hep üzerinde olsun ister",
                                    "17. Arkadaş grubuna alınmaz.",
                                    "18. Başka çocuklar tarafından kolayca yönlendirilir.",
                                    "19. Oyun kurallarına uymaz, mızıkçıdır.",
                                    "20. Liderlik özelliği yoktur.",
                                    "21. Başladığı işin sonunu getiremez.",
                                    "22. Yaşından küçükmüş gibi davranır.",
                                    "23. Suçu başkasına atar.",
                                    "24. Geçimsizdir.",
                                    "25. Arkadaşlarıyla yardımlaşmaz.",
                                    "26. Zorluklardan hemen yılar.",
                                    "27. Öğretmenlerle işbirliği yapmaz.",
                                    "28. Zor öğrenir.",
                                    "Label"]

# Encode the data in utf-8:
for field in field_names_for_parent_data:
    field.encode()

# Encode the data in utf-8:
for field in field_names_for_teacher_data:
    field.encode()

def supportVectorMachinesUtilizingScikit(data_type,data_path, hyper_parameter_tuning=False, search_type=None):
    # Initialize the data_type specific variables:
    if data_type == 'parent':
        # Determine the correct field names for Parent:
        field_names = field_names_for_parent_data
        # Assign the data_type specific string for the file paths:
        data_type_string = 'Parent' 
    else:
        # Determine the correct field names for Teacher:
        field_names = field_names = field_names_for_teacher_data
        # Assign the data_type specific string for the file paths:
        data_type_string = 'Teacher'
    
    # load dataset:
    dataFrame = pd.read_csv(data_path, header=None, names=field_names)

    # Feature selection:
    X = dataFrame[field_names[0:-1]]            # X is another pandas.DataFrame which resembles a 2-D array in which rows could be accessed by their indexes. (And ofcourse columns could be accessed by their column_names)
    # Set the target variable (the classifier variable):
    y = dataFrame["Label"]                      # y is in the type of pandas.Series is a one-dimensional ndarray with axis labels

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

    ####### Hyper-parameter Tuning ######################################################
    if hyper_parameter_tuning:
        if search_type == 'randomized': ### Tuning with Randomized Search ##############
            # Create dictionary for the hyper parameters which we want to optimize:
            hyperParameters = {'C':         uniform(0.1, 10), 
                               'gamma':     ['scale', 'auto'] + list(logspace(-3, 3, 50)),
                               'kernel':    ['linear', 'rbf', 'poly', 'sigmoid']}
            # Create a SVM classifier which will be used for optimization:
            supportVectorMachineModel = svm.SVC()
            # Utilize the random search function provided by Scikit-learn in order to find the best hyperparameters:
            randomized_search = RandomizedSearchCV(estimator=supportVectorMachineModel, param_distributions=hyperParameters, n_iter=20, cv=5)

            # Fit the random search object to the data:
            randomized_search.fit(X_train, y_train)
            # Create a variable for the best model:
            bestModelHypertuned = randomized_search.best_estimator_
            # Print the best values for the hyperparameters:
            print('Best hyperparameters:',  randomized_search.best_params_)
            
        else:       #################################### Tuning with Grid Search #######
            # Create dictionary for the hyper parameters which we want to optimize:
            hyperParameters = {'C':         [0.1,1, 10, 100], 
                               'gamma':     [1,0.1,0.01,0.001],
                               'kernel':    ['rbf', 'poly', 'sigmoid']}
            # Create a SVM classifier which will be used for optimization:
            supportVectorMachineModel = svm.SVC()
            # Utilize the random search function provided by Scikit-learn in order to find the best hyperparameters:
            grid_search = GridSearchCV(supportVectorMachineModel,hyperParameters,refit=True,verbose=2)                                 # The number of cross-validation folds to be used.
            # Fit the random search object to the data:
            grid_search.fit(X_train, y_train)
            # Create a variable for the best model:
            bestModelHypertuned = grid_search.best_estimator_
            # Print the best values for the hyperparameters:
            print('Best hyperparameters:',  grid_search.best_params_)
    ####### END OF Hyper-parameter Tuning ###############################################

    # Create SVM classifier object:
    if hyper_parameter_tuning:
        classifierObject = bestModelHypertuned
    else:
        classifierObject = svm.SVC(kernel='rbf', gamma='auto')

    # Train SVM classifier:
    if not hyper_parameter_tuning:                  # No need to train again a hyper-parameter-tuned model since it has alreay been trained during tuning:
        classifierObject.fit(X_train, y_train)

    # Predict the labels for test dataset:
    y_pred = classifierObject.predict(X_test)

    ################################### Calculate Performance Metrics #######################################################
    ### Calculate numeric performance metrics and write them into the file with the path of performanceMetricsFilePath:
    performanceMetricsFilePath = r'C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\Output\SupportVectorMachines\PerformanceMetrics' + data_type_string + '.txt'
    # Wipe the content of the preformance metrics file which is the result of the previous execution:
    with open(performanceMetricsFilePath,'w',newline='',encoding='UTF-8') as FileWritten:
        FileWritten.write("")
    # Open the file in which the performance metrics will be written in append mode:
    with open(performanceMetricsFilePath,'a',newline='',encoding='UTF-8') as FileWritten:
        # Model Accuracy, how often is the classifier correct?
        FileWritten.write("Accuracy: "  + str(metrics.accuracy_score(y_test, y_pred)) + '\n')
        FileWritten.write("Precision: " + str(metrics.precision_score(y_test, y_pred,average='macro')) + '\n')
        FileWritten.write("Recall: "    + str(metrics.recall_score(y_test, y_pred,average='macro')) + '\n')
        FileWritten.write("F-1 Score: " + str(metrics.f1_score(y_test, y_pred,average='macro')))
    ###### Create the confusion matrix:
    confusionMatrix = confusion_matrix(y_test,y_pred)
    ConfusionMatrixDisplay(confusion_matrix=confusionMatrix).plot().figure_.savefig(r'C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\Output\SupportVectorMachines\SupportVectorMachines' + data_type_string + 'ConfusionMatrix' + '.png')
    ###### Permutation Feature Importance:
    # Permutation feature importance is defined as "the difference between the baseline metric and metric from permutating the feature column".
    # It is used for non-linear kernels!!!
    perm_importance = permutation_importance(classifierObject, X, y)        # Returns a Dictionary-like object. 
    # Normalize the feature importances such that the sum of all feature importances is 1.0, therefore the feature importances can be understood as percentages:
    perm_importance_normalized = perm_importance.importances_mean/perm_importance.importances_mean.sum()
    # Organize features for the plot:
    feature_names = X.columns
    features = numpy.array(feature_names)
    # Sort to plot in the order of importance:
    sorted_idx = perm_importance_normalized.argsort()
    # Plot:
    fig = matplotlib.pyplot.figure(figsize=(16, 16), layout='compressed')       # Create & initialize a figure with a size of 12x12 inches and a compressed layout
    matplotlib.pyplot.title('Permutation Feature Importance',fontsize=20)
    matplotlib.pyplot.barh(features[sorted_idx], perm_importance_normalized[sorted_idx], color='b', align='center')
    matplotlib.pyplot.xlabel('Relative Importance to the Model', fontsize=15)
    matplotlib.pyplot.xticks(fontsize=15)
    matplotlib.pyplot.yticks(fontsize=15)
    matplotlib.pyplot.savefig(r'C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\Output\SupportVectorMachines\SupportVectorMachines' + data_type_string + 'PermutationFeatureImportance' + '.png')

def main():
    supportVectorMachinesUtilizingScikit('parent', r"C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\ConnersParentData.csv", hyper_parameter_tuning=True, search_type='randomized')
    #decisionTreeUtilizingScikit('teacher', r"C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\ConnersTeacherData.csv")

if __name__ == "__main__":
    main()