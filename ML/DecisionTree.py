# -*- coding: utf-8 -*-
# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

# Import necessary libraries for decision tree visualization:
from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus

field_names = [     "1. Eli boş durmaz, sürekli bir şeylerle (tırnak, parmak, giysi gibi…) oynar.", 
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

for field in field_names:
    field.encode()

def decisionTreeUtilizingScikit(data_path):
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
    classifierObject = DecisionTreeClassifier()
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

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    # Visualize the used decision tree:
    dot_data = StringIO()
    export_graphviz(classifierObject, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = field_names[0:-1],class_names=['ADHD_negative','ADHD_positive'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png('decisionTree.png')
    Image(graph.create_png())

def main():
    decisionTreeUtilizingScikit("ConnersParentData.csv")

if __name__ == "__main__":
    main()