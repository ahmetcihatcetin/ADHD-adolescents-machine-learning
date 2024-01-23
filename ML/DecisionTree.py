# -*- coding: utf-8 -*-
# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function

# Import necessary libraries for decision tree visualization:
from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus

# Modules for Performance Metrics:
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Import necessary libraries for plotting the ROC Curve:
import seaborn.objects

# Import necessary libraries for plotting the feature importance plot:
import matplotlib.pyplot as plt
import numpy as np

field_names_for_parent_data = [         "1. Eli boş durmaz, sürekli bir şeylerle (tırnak, parmak, giysi gibi…) oynar.", 
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

field_names_for_teacher_data = [        "1. Kıpır kıpırdır, yerinde duramaz.",
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

field_names_for_doctor_notes_data = [   "1. Hareketli",
                                        "2. Kıpır kıpır",
                                        "3. Çok konuşur/konuşkan",
                                        "4. Sabırsız"
                                        "5. Aceleci",
                                        "6. Sırasını beklemez",
                                        "7. Söz keser",
                                        "8. Araya girer",
                                        "9. Koşturur",
                                        "10. Eli ayağı durmaz",
                                        "11. Motor takılmış gibi",
                                        "12. Aklına eseni yapar",
                                        "13. Sıkılgan",
                                        "14. Yaramaz",
                                        "15. Atılgan",
                                        "16. Öfkeli",
                                        "17. İnatçı",
                                        "18. Çok tartışır",
                                        "19. Düşünmeden aniden hareket eder",
                                        "20. Dikkatsiz",
                                        "21. Dinlemez",
                                        "22. Dağınık",
                                        "23. Düzensiz",
                                        "24. Sakar",
                                        "25. Dalgın",
                                        "26. Savruk",
                                        "27. Hayalci",
                                        "28. Ödevlerini bitirmez/bitiremez",
                                        "29. İsteksiz",
                                        "30. Görevlerini organize edemez",
                                        "31. Eşya kaybeder",
                                        "32. Unutkan",
                                        "33. Söz dinlemez",
                                        "34. Kavgacı",
                                        "35. Sinirli",
                                        "36. İsyankar",
                                        "37. Geçimsiz",
                                        "Label"]
                                        
# Encode the data in utf-8:
for field in field_names_for_parent_data:
    field.encode()

# Encode the data in utf-8:
for field in field_names_for_teacher_data:
    field.encode()

# Encode the data in utf-8:
for field in field_names_for_doctor_notes_data:
    field.encode()

def decisionTreeUtilizingScikit(data_type,data_path):
    # Initialize the data_type specific variables:
    if data_type == 'parent':
        # Determine the correct field names for Parent:
        field_names = field_names_for_parent_data
        # Assign the data_type specific string for the file paths:
        data_type_string = 'Parent' 
    elif data_type == 'teacher':
        # Determine the correct field names for Teacher:
        field_names = field_names = field_names_for_teacher_data
        # Assign the data_type specific string for the file paths:
        data_type_string = 'Teacher'
    elif data_type == 'doctors':
        # Determine the correct field names for Doctor's Notes:
        field_names = field_names = field_names_for_doctor_notes_data
        # Assign the data_type specific string for the file paths:
        data_type_string = 'DoctorsNotes'
    
    # load dataset:
    dataFrame = pd.read_csv(data_path, header=None, names=field_names)
    #print(dataFrame.head())
    
    # Feature selection:
    X = dataFrame[field_names[0:-1]]            # X is another pandas.DataFrame which resembles a 2-D array in which rows could be accessed by their indexes. (And ofcourse columns could be accessed by their column_names)
    # Set the target variable (the classifier variable):
    y = dataFrame["Label"]                      # y is in the type of pandas.Series is a one-dimensional ndarray with axis labels
    
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
    
    # Create Decision Tree classifer object
    classifierObject = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10)
    # Train Decision Tree Classifer
    classifierObject = classifierObject.fit(X_train,y_train)
    #Predict the response for test dataset
    y_pred = classifierObject.predict(X_test)

    # Print out the expected_label VS predicted_label:
    #y_test_list = y_test.to_list()
    #y_pred_list = y_pred.tolist()
    #print(len(y_test_list))
    #for i in range (0,len(y_test_list)):
    #    print(y_test_list[i]+" vs "+y_pred_list[i])

    ################################### Calculate Performance Metrics #######################################################
    performanceMetricsFilePath = r'C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\Output\DecisionTree\PerformanceMetrics' + data_type_string + '.txt'
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
    ConfusionMatrixDisplay(confusion_matrix=confusionMatrix).plot().figure_.savefig(r'C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\Output\DecisionTree\ConfusionMatrix' + data_type_string + '.png')

    # Plot ROC Curve:
    # Predict probabilities for the test set:
    probabilitiesOfClasses = classifierObject.predict_proba(X_test)
    # Keep probablities for only the positive outcome: ADHD_positive
    probabilitiesOfClasses_pos_class = probabilitiesOfClasses[:,1]    
    # Generate probabilities for 45-degrees line (45-degrees line will be used as a reference!)
    noskill_probabilities = [0 for number in range(len(y_test))]
    # Calculate the related data which are false positive rate and true positive rate for the test set:
    falsePosRate_decisionTree, truePosRate__decisionTree,_ = metrics.roc_curve(y_test, probabilitiesOfClasses_pos_class, pos_label='ADHD_positive')
    # Calculate the related data for 45-degrees line:
    falsePosRate_noSkill, truePosRate_noSkill,_ = metrics.roc_curve(y_test, noskill_probabilities, pos_label='ADHD_positive')
    # Plot the ROC Curve with a 45-degrees line as a reference by utilizing seaborn objects library:
    myPlot = seaborn.objects.Plot().add(seaborn.objects.Line(color='red'),x=falsePosRate_decisionTree, y=truePosRate__decisionTree).add(seaborn.objects.Line(color='blue',linestyle='dashed'),x=falsePosRate_noSkill, y=truePosRate_noSkill).layout(size=(8,5))
    # Save the plot on PNG file:
    myPlot.save(r'C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\Output\DecisionTree\ROC_Curve' + data_type_string + '.png')
    ################################### Calculate Performance Metrics END ###################################################

    # Visualize the used decision tree:
    dot_data = StringIO()
    export_graphviz(classifierObject, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = field_names[0:-1],class_names=['ADHD_negative','ADHD_positive'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png(r'C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\Output\DecisionTree\decisionTreeUsed' + data_type_string + '.png')
    Image(graph.create_png())

    ###### Plot the Importance of each Feature:
    # Gini-based:
    feature_importance = classifierObject.feature_importances_                  # The impurity-based feature importances. Type: ndarray
    sorted_idx = np.argsort(feature_importance)                                 # Perform an indirect quicksort on the feature importances ndarray
    plt.figure(2)
    plt.figure(figsize=(12, 12), layout='compressed')       # Create & initialize a figure with a size of 12x12 inches and a compressed layout
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx])
    plt.title('Feature Importance', fontsize=20)
    plt.xlabel('Relative Importance to the Model', fontsize=15)
    plt.savefig(r'C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\Output\DecisionTree\FeatureImportance' + data_type_string + '.png')

def main():
    decisionTreeUtilizingScikit('doctors', r"C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\DoctorsNotesData.csv")
    #decisionTreeUtilizingScikit('parent', r"C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\ConnersParentData.csv")
    #decisionTreeUtilizingScikit('teacher', r"C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\ConnersTeacherData.csv")

if __name__ == "__main__":
    main()