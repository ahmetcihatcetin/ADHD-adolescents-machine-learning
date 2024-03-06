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
import matplotlib.pyplot as plt
import numpy as np

# Import necessary libraries for K-fold Cross Validation
from sklearn.model_selection import KFold

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

field_names_for_risk_factors_data = [   "1. Cinsiyet",
                                        "2. Gebelikte sigara kullanımı",
                                        "3. DDA",
                                        "4. Erken doğum",
                                        "5. Anne yaşı",
                                        "6. Doğumda anne yaşı",
                                        "7. İndüklenmiş eylem",
                                        "8. Fetaldistres-hipoksi",
                                        "9. Sezaryendoğum",
                                        "10. Sarılık",
                                        "11. Gebeliktekomp",
                                        "12. Kardeşlerdedehb",
                                        "Label"]

field_names_for_combined_conners_parent_and_teacher_data = [
                                        "ConnersParent-1. Eli boş durmaz, sürekli bir şeylerle (tırnak, parmak, giysi gibi…) oynar.", 
                                        "ConnersParent-2. Büyüklere karşı arsız ve küstah davranır.",
                                        "ConnersParent-3. Arkadaşlık kurmada ve sürdürmede zorlanır.",
                                        "ConnersParent-4. Çabuk heyecanlanır, ataktır.",
                                        "ConnersParent-5. Her şeye karışır ve yönetmek ister.",
                                        "ConnersParent-6. Bir şeyler çiğner veya emer (parmak, giysi gibi)",
                                        "ConnersParent-7. Sık sık ve kolayca ağlar.",
                                        "ConnersParent-8. Her an sataşmaya hazırdır.",
                                        "ConnersParent-9. Hayallere dalar.",
                                        "ConnersParent-10. Zor öğrenir.",
                                        "ConnersParent-11. Kıpır kıpırdır, tez canlıdır.",
                                        "ConnersParent-12. Ürkektir (yeni durum, insan ve yerlerden).",
                                        "ConnersParent-13. Yerinde duramaz, her an harekete hazırdır.",
                                        "ConnersParent-14. Zarar verir.",
                                        "ConnersParent-15. Yalan söyler, masallar uydurur.",
                                        "ConnersParent-16. Utangaçtır.",
                                        "ConnersParent-17. Yaşıtlarından daha sık başını derde sokar.",
                                        "ConnersParent-18. Yaşıtlarından farklı (çocuksu, zor anlaşılır, kekeleyerek gibi…) konuşur.",
                                        "ConnersParent-19. Hatalarını kabullenmez, başkalarını suçlar.",
                                        "ConnersParent-20. Kavgacıdır.",
                                        "ConnersParent-21. Somurtkan. ve asık suratlıdır.",
                                        "ConnersParent-22. Çalma huyu vardır.",
                                        "ConnersParent-23. Söz dinlemez ya da isteksiz ve zoraki dinler.",
                                        "ConnersParent-24. Başkalarına göre endişelidir (yalnız kalma, hastalanma, ölüm gibi konularda)",
                                        "ConnersParent-25. Başladığı bir işin sonunu getiremez.",
                                        "ConnersParent-26. Hassastır, kolay incinir.",
                                        "ConnersParent-27. Kabadayılık taslar, başkalarını rahatsız eder.",
                                        "ConnersParent-28. Tekrarlayıcı durduramadığı hareketleri vardır.",
                                        "ConnersParent-29. Kaba ve acımasızdır.",
                                        "ConnersParent-30. Yaşına göre daha çocuksudur.",
                                        "ConnersParent-31. Dikkati kolay dağılır ya da uzun süre dikkatini toplayamaz.",
                                        "ConnersParent-32. Baş ağrıları olur.",
                                        "ConnersParent-33. Ruh halinde ani ve göz batan değişiklikler olur.",
                                        "ConnersParent-34. Kurallar ve kısıtlamalardan hoşlanmaz ve uymaz.",
                                        "ConnersParent-35. Sürekli kavga eder.",
                                        "ConnersParent-36. Kardeşleriyle iyi geçinemez.",
                                        "ConnersParent-37. Zora gelemez.",
                                        "ConnersParent-38. Diğer çocukları rahatsız eder.",
                                        "ConnersParent-39. Genelde hoşnutsuz bir çocuktur.",
                                        "ConnersParent-40. Yeme sorunları vardır (sofradan sık sık kalkar, iştahsızdır gibi…).",
                                        "ConnersParent-41. Karın ağrıları olur.",
                                        "ConnersParent-42. Uyku sorunları vardır (uykuya dalamam, erken uyanma, gece kalkma gibi…).",
                                        "ConnersParent-43. Çeşitli ağrı ve sancıları olur.",
                                        "ConnersParent-44. Bulantı kusmaları olur.",
                                        "ConnersParent-45. Aile içinde daha az kayrıldığını düşünür.",
                                        "ConnersParent-46. Övünür böbürlenir.",
                                        "ConnersParent-47. İtilip kakılmaya müsaittir.",
                                        "ConnersParent-48. Dışkılama sorunları vardır (sık ishal, kabızlık, düzensiz tuvalet alışkanlığı gibi…).",
                                        "ConnersTeacher-1. Kıpır kıpırdır, yerinde duramaz.",
                                        "ConnersTeacher-2. Zamansız ve uyumsuz sesler çıkarır.",
                                        "ConnersTeacher-3. İstekleri hemen yerine getirilmelidir.",
                                        "ConnersTeacher-4. Bilmiş tavırları vardır, bilgiçlik taslar.",
                                        "ConnersTeacher-5. Aniden parlar, ne yapacağı belli olmaz.",
                                        "ConnersTeacher-6. Eleştiri kaldıramaz.",
                                        "ConnersTeacher-7. Dikkati dağınıktır.",
                                        "ConnersTeacher-8. Diğer çocukları rahatsız eder.",
                                        "ConnersTeacher-9. Hayallere dalar.",
                                        "ConnersTeacher-10. Somurtur, surat asar.",
                                        "ConnersTeacher-11. Bir anı bir anını tutmaz.",
                                        "ConnersTeacher-12. Kavgacıdır.",
                                        "ConnersTeacher-13. Büyüklerin sözünden çıkmaz.",
                                        "ConnersTeacher-14. Hareketlidir, dur otur bilmez.",
                                        "ConnersTeacher-15. Düşünmeden hareket eder.",
                                        "ConnersTeacher-16. Öğretmenin ilgisi hep üzerinde olsun ister",
                                        "ConnersTeacher-17. Arkadaş grubuna alınmaz.",
                                        "ConnersTeacher-18. Başka çocuklar tarafından kolayca yönlendirilir.",
                                        "ConnersTeacher-19. Oyun kurallarına uymaz, mızıkçıdır.",
                                        "ConnersTeacher-20. Liderlik özelliği yoktur.",
                                        "ConnersTeacher-21. Başladığı işin sonunu getiremez.",
                                        "ConnersTeacher-22. Yaşından küçükmüş gibi davranır.",
                                        "ConnersTeacher-23. Suçu başkasına atar.",
                                        "ConnersTeacher-24. Geçimsizdir.",
                                        "ConnersTeacher-25. Arkadaşlarıyla yardımlaşmaz.",
                                        "ConnersTeacher-26. Zorluklardan hemen yılar.",
                                        "ConnersTeacher-27. Öğretmenlerle işbirliği yapmaz.",
                                        "ConnersTeacher-28. Zor öğrenir.",
                                        "Label"]

field_names_for_combined_conners_and_doctors_notes_data = [
                                        "ConnersParent-1. Eli boş durmaz, sürekli bir şeylerle (tırnak, parmak, giysi gibi…) oynar.", 
                                        "ConnersParent-2. Büyüklere karşı arsız ve küstah davranır.",
                                        "ConnersParent-3. Arkadaşlık kurmada ve sürdürmede zorlanır.",
                                        "ConnersParent-4. Çabuk heyecanlanır, ataktır.",
                                        "ConnersParent-5. Her şeye karışır ve yönetmek ister.",
                                        "ConnersParent-6. Bir şeyler çiğner veya emer (parmak, giysi gibi)",
                                        "ConnersParent-7. Sık sık ve kolayca ağlar.",
                                        "ConnersParent-8. Her an sataşmaya hazırdır.",
                                        "ConnersParent-9. Hayallere dalar.",
                                        "ConnersParent-10. Zor öğrenir.",
                                        "ConnersParent-11. Kıpır kıpırdır, tez canlıdır.",
                                        "ConnersParent-12. Ürkektir (yeni durum, insan ve yerlerden).",
                                        "ConnersParent-13. Yerinde duramaz, her an harekete hazırdır.",
                                        "ConnersParent-14. Zarar verir.",
                                        "ConnersParent-15. Yalan söyler, masallar uydurur.",
                                        "ConnersParent-16. Utangaçtır.",
                                        "ConnersParent-17. Yaşıtlarından daha sık başını derde sokar.",
                                        "ConnersParent-18. Yaşıtlarından farklı (çocuksu, zor anlaşılır, kekeleyerek gibi…) konuşur.",
                                        "ConnersParent-19. Hatalarını kabullenmez, başkalarını suçlar.",
                                        "ConnersParent-20. Kavgacıdır.",
                                        "ConnersParent-21. Somurtkan. ve asık suratlıdır.",
                                        "ConnersParent-22. Çalma huyu vardır.",
                                        "ConnersParent-23. Söz dinlemez ya da isteksiz ve zoraki dinler.",
                                        "ConnersParent-24. Başkalarına göre endişelidir (yalnız kalma, hastalanma, ölüm gibi konularda)",
                                        "ConnersParent-25. Başladığı bir işin sonunu getiremez.",
                                        "ConnersParent-26. Hassastır, kolay incinir.",
                                        "ConnersParent-27. Kabadayılık taslar, başkalarını rahatsız eder.",
                                        "ConnersParent-28. Tekrarlayıcı durduramadığı hareketleri vardır.",
                                        "ConnersParent-29. Kaba ve acımasızdır.",
                                        "ConnersParent-30. Yaşına göre daha çocuksudur.",
                                        "ConnersParent-31. Dikkati kolay dağılır ya da uzun süre dikkatini toplayamaz.",
                                        "ConnersParent-32. Baş ağrıları olur.",
                                        "ConnersParent-33. Ruh halinde ani ve göz batan değişiklikler olur.",
                                        "ConnersParent-34. Kurallar ve kısıtlamalardan hoşlanmaz ve uymaz.",
                                        "ConnersParent-35. Sürekli kavga eder.",
                                        "ConnersParent-36. Kardeşleriyle iyi geçinemez.",
                                        "ConnersParent-37. Zora gelemez.",
                                        "ConnersParent-38. Diğer çocukları rahatsız eder.",
                                        "ConnersParent-39. Genelde hoşnutsuz bir çocuktur.",
                                        "ConnersParent-40. Yeme sorunları vardır (sofradan sık sık kalkar, iştahsızdır gibi…).",
                                        "ConnersParent-41. Karın ağrıları olur.",
                                        "ConnersParent-42. Uyku sorunları vardır (uykuya dalamam, erken uyanma, gece kalkma gibi…).",
                                        "ConnersParent-43. Çeşitli ağrı ve sancıları olur.",
                                        "ConnersParent-44. Bulantı kusmaları olur.",
                                        "ConnersParent-45. Aile içinde daha az kayrıldığını düşünür.",
                                        "ConnersParent-46. Övünür böbürlenir.",
                                        "ConnersParent-47. İtilip kakılmaya müsaittir.",
                                        "ConnersParent-48. Dışkılama sorunları vardır (sık ishal, kabızlık, düzensiz tuvalet alışkanlığı gibi…).",
                                        "ConnersTeacher-1. Kıpır kıpırdır, yerinde duramaz.",
                                        "ConnersTeacher-2. Zamansız ve uyumsuz sesler çıkarır.",
                                        "ConnersTeacher-3. İstekleri hemen yerine getirilmelidir.",
                                        "ConnersTeacher-4. Bilmiş tavırları vardır, bilgiçlik taslar.",
                                        "ConnersTeacher-5. Aniden parlar, ne yapacağı belli olmaz.",
                                        "ConnersTeacher-6. Eleştiri kaldıramaz.",
                                        "ConnersTeacher-7. Dikkati dağınıktır.",
                                        "ConnersTeacher-8. Diğer çocukları rahatsız eder.",
                                        "ConnersTeacher-9. Hayallere dalar.",
                                        "ConnersTeacher-10. Somurtur, surat asar.",
                                        "ConnersTeacher-11. Bir anı bir anını tutmaz.",
                                        "ConnersTeacher-12. Kavgacıdır.",
                                        "ConnersTeacher-13. Büyüklerin sözünden çıkmaz.",
                                        "ConnersTeacher-14. Hareketlidir, dur otur bilmez.",
                                        "ConnersTeacher-15. Düşünmeden hareket eder.",
                                        "ConnersTeacher-16. Öğretmenin ilgisi hep üzerinde olsun ister",
                                        "ConnersTeacher-17. Arkadaş grubuna alınmaz.",
                                        "ConnersTeacher-18. Başka çocuklar tarafından kolayca yönlendirilir.",
                                        "ConnersTeacher-19. Oyun kurallarına uymaz, mızıkçıdır.",
                                        "ConnersTeacher-20. Liderlik özelliği yoktur.",
                                        "ConnersTeacher-21. Başladığı işin sonunu getiremez.",
                                        "ConnersTeacher-22. Yaşından küçükmüş gibi davranır.",
                                        "ConnersTeacher-23. Suçu başkasına atar.",
                                        "ConnersTeacher-24. Geçimsizdir.",
                                        "ConnersTeacher-25. Arkadaşlarıyla yardımlaşmaz.",
                                        "ConnersTeacher-26. Zorluklardan hemen yılar.",
                                        "ConnersTeacher-27. Öğretmenlerle işbirliği yapmaz.",
                                        "ConnersTeacher-28. Zor öğrenir.",
                                        "Doctor-1. Hareketli",
                                        "Doctor-2. Kıpır kıpır",
                                        "Doctor-3. Çok konuşur/konuşkan",
                                        "Doctor-4. Sabırsız"
                                        "Doctor-5. Aceleci",
                                        "Doctor-6. Sırasını beklemez",
                                        "Doctor-7. Söz keser",
                                        "Doctor-8. Araya girer",
                                        "Doctor-9. Koşturur",
                                        "Doctor-10. Eli ayağı durmaz",
                                        "Doctor-11. Motor takılmış gibi",
                                        "Doctor-12. Aklına eseni yapar",
                                        "Doctor-13. Sıkılgan",
                                        "Doctor-14. Yaramaz",
                                        "Doctor-15. Atılgan",
                                        "Doctor-16. Öfkeli",
                                        "Doctor-17. İnatçı",
                                        "Doctor-18. Çok tartışır",
                                        "Doctor-19. Düşünmeden aniden hareket eder",
                                        "Doctor-20. Dikkatsiz",
                                        "Doctor-21. Dinlemez",
                                        "Doctor-22. Dağınık",
                                        "Doctor-23. Düzensiz",
                                        "Doctor-24. Sakar",
                                        "Doctor-25. Dalgın",
                                        "Doctor-26. Savruk",
                                        "Doctor-27. Hayalci",
                                        "Doctor-28. Ödevlerini bitirmez/bitiremez",
                                        "Doctor-29. İsteksiz",
                                        "Doctor-30. Görevlerini organize edemez",
                                        "Doctor-31. Eşya kaybeder",
                                        "Doctor-32. Unutkan",
                                        "Doctor-33. Söz dinlemez",
                                        "Doctor-34. Kavgacı",
                                        "Doctor-35. Sinirli",
                                        "Doctor-36. İsyankar",
                                        "Doctor-37. Geçimsiz",
                                        "Label"]

field_names_for_combined_conners_and_risk_factors_data = [
                                        "ConnersParent-1. Eli boş durmaz, sürekli bir şeylerle (tırnak, parmak, giysi gibi…) oynar.", 
                                        "ConnersParent-2. Büyüklere karşı arsız ve küstah davranır.",
                                        "ConnersParent-3. Arkadaşlık kurmada ve sürdürmede zorlanır.",
                                        "ConnersParent-4. Çabuk heyecanlanır, ataktır.",
                                        "ConnersParent-5. Her şeye karışır ve yönetmek ister.",
                                        "ConnersParent-6. Bir şeyler çiğner veya emer (parmak, giysi gibi)",
                                        "ConnersParent-7. Sık sık ve kolayca ağlar.",
                                        "ConnersParent-8. Her an sataşmaya hazırdır.",
                                        "ConnersParent-9. Hayallere dalar.",
                                        "ConnersParent-10. Zor öğrenir.",
                                        "ConnersParent-11. Kıpır kıpırdır, tez canlıdır.",
                                        "ConnersParent-12. Ürkektir (yeni durum, insan ve yerlerden).",
                                        "ConnersParent-13. Yerinde duramaz, her an harekete hazırdır.",
                                        "ConnersParent-14. Zarar verir.",
                                        "ConnersParent-15. Yalan söyler, masallar uydurur.",
                                        "ConnersParent-16. Utangaçtır.",
                                        "ConnersParent-17. Yaşıtlarından daha sık başını derde sokar.",
                                        "ConnersParent-18. Yaşıtlarından farklı (çocuksu, zor anlaşılır, kekeleyerek gibi…) konuşur.",
                                        "ConnersParent-19. Hatalarını kabullenmez, başkalarını suçlar.",
                                        "ConnersParent-20. Kavgacıdır.",
                                        "ConnersParent-21. Somurtkan. ve asık suratlıdır.",
                                        "ConnersParent-22. Çalma huyu vardır.",
                                        "ConnersParent-23. Söz dinlemez ya da isteksiz ve zoraki dinler.",
                                        "ConnersParent-24. Başkalarına göre endişelidir (yalnız kalma, hastalanma, ölüm gibi konularda)",
                                        "ConnersParent-25. Başladığı bir işin sonunu getiremez.",
                                        "ConnersParent-26. Hassastır, kolay incinir.",
                                        "ConnersParent-27. Kabadayılık taslar, başkalarını rahatsız eder.",
                                        "ConnersParent-28. Tekrarlayıcı durduramadığı hareketleri vardır.",
                                        "ConnersParent-29. Kaba ve acımasızdır.",
                                        "ConnersParent-30. Yaşına göre daha çocuksudur.",
                                        "ConnersParent-31. Dikkati kolay dağılır ya da uzun süre dikkatini toplayamaz.",
                                        "ConnersParent-32. Baş ağrıları olur.",
                                        "ConnersParent-33. Ruh halinde ani ve göz batan değişiklikler olur.",
                                        "ConnersParent-34. Kurallar ve kısıtlamalardan hoşlanmaz ve uymaz.",
                                        "ConnersParent-35. Sürekli kavga eder.",
                                        "ConnersParent-36. Kardeşleriyle iyi geçinemez.",
                                        "ConnersParent-37. Zora gelemez.",
                                        "ConnersParent-38. Diğer çocukları rahatsız eder.",
                                        "ConnersParent-39. Genelde hoşnutsuz bir çocuktur.",
                                        "ConnersParent-40. Yeme sorunları vardır (sofradan sık sık kalkar, iştahsızdır gibi…).",
                                        "ConnersParent-41. Karın ağrıları olur.",
                                        "ConnersParent-42. Uyku sorunları vardır (uykuya dalamam, erken uyanma, gece kalkma gibi…).",
                                        "ConnersParent-43. Çeşitli ağrı ve sancıları olur.",
                                        "ConnersParent-44. Bulantı kusmaları olur.",
                                        "ConnersParent-45. Aile içinde daha az kayrıldığını düşünür.",
                                        "ConnersParent-46. Övünür böbürlenir.",
                                        "ConnersParent-47. İtilip kakılmaya müsaittir.",
                                        "ConnersParent-48. Dışkılama sorunları vardır (sık ishal, kabızlık, düzensiz tuvalet alışkanlığı gibi…).",
                                        "ConnersTeacher-1. Kıpır kıpırdır, yerinde duramaz.",
                                        "ConnersTeacher-2. Zamansız ve uyumsuz sesler çıkarır.",
                                        "ConnersTeacher-3. İstekleri hemen yerine getirilmelidir.",
                                        "ConnersTeacher-4. Bilmiş tavırları vardır, bilgiçlik taslar.",
                                        "ConnersTeacher-5. Aniden parlar, ne yapacağı belli olmaz.",
                                        "ConnersTeacher-6. Eleştiri kaldıramaz.",
                                        "ConnersTeacher-7. Dikkati dağınıktır.",
                                        "ConnersTeacher-8. Diğer çocukları rahatsız eder.",
                                        "ConnersTeacher-9. Hayallere dalar.",
                                        "ConnersTeacher-10. Somurtur, surat asar.",
                                        "ConnersTeacher-11. Bir anı bir anını tutmaz.",
                                        "ConnersTeacher-12. Kavgacıdır.",
                                        "ConnersTeacher-13. Büyüklerin sözünden çıkmaz.",
                                        "ConnersTeacher-14. Hareketlidir, dur otur bilmez.",
                                        "ConnersTeacher-15. Düşünmeden hareket eder.",
                                        "ConnersTeacher-16. Öğretmenin ilgisi hep üzerinde olsun ister",
                                        "ConnersTeacher-17. Arkadaş grubuna alınmaz.",
                                        "ConnersTeacher-18. Başka çocuklar tarafından kolayca yönlendirilir.",
                                        "ConnersTeacher-19. Oyun kurallarına uymaz, mızıkçıdır.",
                                        "ConnersTeacher-20. Liderlik özelliği yoktur.",
                                        "ConnersTeacher-21. Başladığı işin sonunu getiremez.",
                                        "ConnersTeacher-22. Yaşından küçükmüş gibi davranır.",
                                        "ConnersTeacher-23. Suçu başkasına atar.",
                                        "ConnersTeacher-24. Geçimsizdir.",
                                        "ConnersTeacher-25. Arkadaşlarıyla yardımlaşmaz.",
                                        "ConnersTeacher-26. Zorluklardan hemen yılar.",
                                        "ConnersTeacher-27. Öğretmenlerle işbirliği yapmaz.",
                                        "ConnersTeacher-28. Zor öğrenir.",
                                        "RiskFactors-1. Cinsiyet",
                                        "RiskFactors-2. Gebelikte sigara kullanımı",
                                        "RiskFactors-3. DDA",
                                        "RiskFactors-4. Erken doğum",
                                        "RiskFactors-5. Anne yaşı",
                                        "RiskFactors-6. Doğumda anne yaşı",
                                        "RiskFactors-7. İndüklenmiş eylem",
                                        "RiskFactors-8. Fetaldistres-hipoksi",
                                        "RiskFactors-9. Sezaryendoğum",
                                        "RiskFactors-10. Sarılık",
                                        "RiskFactors-11. Gebeliktekomp",
                                        "RiskFactors-12. Kardeşlerdedehb",
                                        "Label"]

field_names_for_combined_doctors_notes_and_risk_factors_data = [
                                        "Doctor-1. Hareketli",
                                        "Doctor-2. Kıpır kıpır",
                                        "Doctor-3. Çok konuşur/konuşkan",
                                        "Doctor-4. Sabırsız"
                                        "Doctor-5. Aceleci",
                                        "Doctor-6. Sırasını beklemez",
                                        "Doctor-7. Söz keser",
                                        "Doctor-8. Araya girer",
                                        "Doctor-9. Koşturur",
                                        "Doctor-10. Eli ayağı durmaz",
                                        "Doctor-11. Motor takılmış gibi",
                                        "Doctor-12. Aklına eseni yapar",
                                        "Doctor-13. Sıkılgan",
                                        "Doctor-14. Yaramaz",
                                        "Doctor-15. Atılgan",
                                        "Doctor-16. Öfkeli",
                                        "Doctor-17. İnatçı",
                                        "Doctor-18. Çok tartışır",
                                        "Doctor-19. Düşünmeden aniden hareket eder",
                                        "Doctor-20. Dikkatsiz",
                                        "Doctor-21. Dinlemez",
                                        "Doctor-22. Dağınık",
                                        "Doctor-23. Düzensiz",
                                        "Doctor-24. Sakar",
                                        "Doctor-25. Dalgın",
                                        "Doctor-26. Savruk",
                                        "Doctor-27. Hayalci",
                                        "Doctor-28. Ödevlerini bitirmez/bitiremez",
                                        "Doctor-29. İsteksiz",
                                        "Doctor-30. Görevlerini organize edemez",
                                        "Doctor-31. Eşya kaybeder",
                                        "Doctor-32. Unutkan",
                                        "Doctor-33. Söz dinlemez",
                                        "Doctor-34. Kavgacı",
                                        "Doctor-35. Sinirli",
                                        "Doctor-36. İsyankar",
                                        "Doctor-37. Geçimsiz",
                                        "RiskFactors-1. Cinsiyet",
                                        "RiskFactors-2. Gebelikte sigara kullanımı",
                                        "RiskFactors-3. DDA",
                                        "RiskFactors-4. Erken doğum",
                                        "RiskFactors-5. Anne yaşı",
                                        "RiskFactors-6. Doğumda anne yaşı",
                                        "RiskFactors-7. İndüklenmiş eylem",
                                        "RiskFactors-8. Fetaldistres-hipoksi",
                                        "RiskFactors-9. Sezaryendoğum",
                                        "RiskFactors-10. Sarılık",
                                        "RiskFactors-11. Gebeliktekomp",
                                        "RiskFactors-12. Kardeşlerdedehb",
                                        "Label"]

field_names_for_combined_conners_and_doctors_notes_and_risk_factors_data = [
                                        "ConnersParent-1. Eli boş durmaz, sürekli bir şeylerle (tırnak, parmak, giysi gibi…) oynar.", 
                                        "ConnersParent-2. Büyüklere karşı arsız ve küstah davranır.",
                                        "ConnersParent-3. Arkadaşlık kurmada ve sürdürmede zorlanır.",
                                        "ConnersParent-4. Çabuk heyecanlanır, ataktır.",
                                        "ConnersParent-5. Her şeye karışır ve yönetmek ister.",
                                        "ConnersParent-6. Bir şeyler çiğner veya emer (parmak, giysi gibi)",
                                        "ConnersParent-7. Sık sık ve kolayca ağlar.",
                                        "ConnersParent-8. Her an sataşmaya hazırdır.",
                                        "ConnersParent-9. Hayallere dalar.",
                                        "ConnersParent-10. Zor öğrenir.",
                                        "ConnersParent-11. Kıpır kıpırdır, tez canlıdır.",
                                        "ConnersParent-12. Ürkektir (yeni durum, insan ve yerlerden).",
                                        "ConnersParent-13. Yerinde duramaz, her an harekete hazırdır.",
                                        "ConnersParent-14. Zarar verir.",
                                        "ConnersParent-15. Yalan söyler, masallar uydurur.",
                                        "ConnersParent-16. Utangaçtır.",
                                        "ConnersParent-17. Yaşıtlarından daha sık başını derde sokar.",
                                        "ConnersParent-18. Yaşıtlarından farklı (çocuksu, zor anlaşılır, kekeleyerek gibi…) konuşur.",
                                        "ConnersParent-19. Hatalarını kabullenmez, başkalarını suçlar.",
                                        "ConnersParent-20. Kavgacıdır.",
                                        "ConnersParent-21. Somurtkan. ve asık suratlıdır.",
                                        "ConnersParent-22. Çalma huyu vardır.",
                                        "ConnersParent-23. Söz dinlemez ya da isteksiz ve zoraki dinler.",
                                        "ConnersParent-24. Başkalarına göre endişelidir (yalnız kalma, hastalanma, ölüm gibi konularda)",
                                        "ConnersParent-25. Başladığı bir işin sonunu getiremez.",
                                        "ConnersParent-26. Hassastır, kolay incinir.",
                                        "ConnersParent-27. Kabadayılık taslar, başkalarını rahatsız eder.",
                                        "ConnersParent-28. Tekrarlayıcı durduramadığı hareketleri vardır.",
                                        "ConnersParent-29. Kaba ve acımasızdır.",
                                        "ConnersParent-30. Yaşına göre daha çocuksudur.",
                                        "ConnersParent-31. Dikkati kolay dağılır ya da uzun süre dikkatini toplayamaz.",
                                        "ConnersParent-32. Baş ağrıları olur.",
                                        "ConnersParent-33. Ruh halinde ani ve göz batan değişiklikler olur.",
                                        "ConnersParent-34. Kurallar ve kısıtlamalardan hoşlanmaz ve uymaz.",
                                        "ConnersParent-35. Sürekli kavga eder.",
                                        "ConnersParent-36. Kardeşleriyle iyi geçinemez.",
                                        "ConnersParent-37. Zora gelemez.",
                                        "ConnersParent-38. Diğer çocukları rahatsız eder.",
                                        "ConnersParent-39. Genelde hoşnutsuz bir çocuktur.",
                                        "ConnersParent-40. Yeme sorunları vardır (sofradan sık sık kalkar, iştahsızdır gibi…).",
                                        "ConnersParent-41. Karın ağrıları olur.",
                                        "ConnersParent-42. Uyku sorunları vardır (uykuya dalamam, erken uyanma, gece kalkma gibi…).",
                                        "ConnersParent-43. Çeşitli ağrı ve sancıları olur.",
                                        "ConnersParent-44. Bulantı kusmaları olur.",
                                        "ConnersParent-45. Aile içinde daha az kayrıldığını düşünür.",
                                        "ConnersParent-46. Övünür böbürlenir.",
                                        "ConnersParent-47. İtilip kakılmaya müsaittir.",
                                        "ConnersParent-48. Dışkılama sorunları vardır (sık ishal, kabızlık, düzensiz tuvalet alışkanlığı gibi…).",
                                        "ConnersTeacher-1. Kıpır kıpırdır, yerinde duramaz.",
                                        "ConnersTeacher-2. Zamansız ve uyumsuz sesler çıkarır.",
                                        "ConnersTeacher-3. İstekleri hemen yerine getirilmelidir.",
                                        "ConnersTeacher-4. Bilmiş tavırları vardır, bilgiçlik taslar.",
                                        "ConnersTeacher-5. Aniden parlar, ne yapacağı belli olmaz.",
                                        "ConnersTeacher-6. Eleştiri kaldıramaz.",
                                        "ConnersTeacher-7. Dikkati dağınıktır.",
                                        "ConnersTeacher-8. Diğer çocukları rahatsız eder.",
                                        "ConnersTeacher-9. Hayallere dalar.",
                                        "ConnersTeacher-10. Somurtur, surat asar.",
                                        "ConnersTeacher-11. Bir anı bir anını tutmaz.",
                                        "ConnersTeacher-12. Kavgacıdır.",
                                        "ConnersTeacher-13. Büyüklerin sözünden çıkmaz.",
                                        "ConnersTeacher-14. Hareketlidir, dur otur bilmez.",
                                        "ConnersTeacher-15. Düşünmeden hareket eder.",
                                        "ConnersTeacher-16. Öğretmenin ilgisi hep üzerinde olsun ister",
                                        "ConnersTeacher-17. Arkadaş grubuna alınmaz.",
                                        "ConnersTeacher-18. Başka çocuklar tarafından kolayca yönlendirilir.",
                                        "ConnersTeacher-19. Oyun kurallarına uymaz, mızıkçıdır.",
                                        "ConnersTeacher-20. Liderlik özelliği yoktur.",
                                        "ConnersTeacher-21. Başladığı işin sonunu getiremez.",
                                        "ConnersTeacher-22. Yaşından küçükmüş gibi davranır.",
                                        "ConnersTeacher-23. Suçu başkasına atar.",
                                        "ConnersTeacher-24. Geçimsizdir.",
                                        "ConnersTeacher-25. Arkadaşlarıyla yardımlaşmaz.",
                                        "ConnersTeacher-26. Zorluklardan hemen yılar.",
                                        "ConnersTeacher-27. Öğretmenlerle işbirliği yapmaz.",
                                        "ConnersTeacher-28. Zor öğrenir.",
                                        "Doctor-1. Hareketli",
                                        "Doctor-2. Kıpır kıpır",
                                        "Doctor-3. Çok konuşur/konuşkan",
                                        "Doctor-4. Sabırsız"
                                        "Doctor-5. Aceleci",
                                        "Doctor-6. Sırasını beklemez",
                                        "Doctor-7. Söz keser",
                                        "Doctor-8. Araya girer",
                                        "Doctor-9. Koşturur",
                                        "Doctor-10. Eli ayağı durmaz",
                                        "Doctor-11. Motor takılmış gibi",
                                        "Doctor-12. Aklına eseni yapar",
                                        "Doctor-13. Sıkılgan",
                                        "Doctor-14. Yaramaz",
                                        "Doctor-15. Atılgan",
                                        "Doctor-16. Öfkeli",
                                        "Doctor-17. İnatçı",
                                        "Doctor-18. Çok tartışır",
                                        "Doctor-19. Düşünmeden aniden hareket eder",
                                        "Doctor-20. Dikkatsiz",
                                        "Doctor-21. Dinlemez",
                                        "Doctor-22. Dağınık",
                                        "Doctor-23. Düzensiz",
                                        "Doctor-24. Sakar",
                                        "Doctor-25. Dalgın",
                                        "Doctor-26. Savruk",
                                        "Doctor-27. Hayalci",
                                        "Doctor-28. Ödevlerini bitirmez/bitiremez",
                                        "Doctor-29. İsteksiz",
                                        "Doctor-30. Görevlerini organize edemez",
                                        "Doctor-31. Eşya kaybeder",
                                        "Doctor-32. Unutkan",
                                        "Doctor-33. Söz dinlemez",
                                        "Doctor-34. Kavgacı",
                                        "Doctor-35. Sinirli",
                                        "Doctor-36. İsyankar",
                                        "Doctor-37. Geçimsiz",
                                        "RiskFactors-1. Cinsiyet",
                                        "RiskFactors-2. Gebelikte sigara kullanımı",
                                        "RiskFactors-3. DDA",
                                        "RiskFactors-4. Erken doğum",
                                        "RiskFactors-5. Anne yaşı",
                                        "RiskFactors-6. Doğumda anne yaşı",
                                        "RiskFactors-7. İndüklenmiş eylem",
                                        "RiskFactors-8. Fetaldistres-hipoksi",
                                        "RiskFactors-9. Sezaryendoğum",
                                        "RiskFactors-10. Sarılık",
                                        "RiskFactors-11. Gebeliktekomp",
                                        "RiskFactors-12. Kardeşlerdedehb",
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

# Encode the data in utf-8:
for field in field_names_for_risk_factors_data:
    field.encode()

# Encode the data in utf-8:
for field in field_names_for_combined_conners_parent_and_teacher_data:
    field.encode()

# Encode the data in utf-8:
for field in field_names_for_combined_conners_and_doctors_notes_data:
    field.encode()

# Encode the data in utf-8:
for field in field_names_for_combined_conners_and_risk_factors_data:
    field.encode()

# Encode the data in utf-8:
for field in field_names_for_combined_doctors_notes_and_risk_factors_data:
    field.encode()

# Encode the data in utf-8:
for field in field_names_for_combined_conners_and_doctors_notes_and_risk_factors_data:
    field.encode()

def supportVectorMachinesUtilizingScikit(data_type,data_path, hyper_parameter_tuning=False, search_type=None, cross_validation=False):
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
    elif data_type == 'risk':
        # Determine the correct field names for Risk Factors:
        field_names = field_names = field_names_for_risk_factors_data
        # Assign the data_type specific string for the file paths:
        data_type_string = 'RiskFactors'
    elif data_type == 'combinedConnersParentAndTeacher':
        # Determine the correct field names for Doctor's Notes:
        field_names = field_names = field_names_for_combined_conners_parent_and_teacher_data
        # Assign the data_type specific string for the file paths:
        data_type_string = 'CombinedConnersParentAndTeacher'
    elif data_type == 'combinedConnersAndDoctorsNotes':
        # Determine the correct field names for Doctor's Notes:
        field_names = field_names = field_names_for_combined_conners_and_doctors_notes_data
        # Assign the data_type specific string for the file paths:
        data_type_string = 'CombinedConnersAndDoctorsNotes'
    elif data_type == 'combinedConnersAndRiskFactors':
        # Determine the correct field names for Doctor's Notes:
        field_names = field_names = field_names_for_combined_conners_and_risk_factors_data
        # Assign the data_type specific string for the file paths:
        data_type_string = 'CombinedConnersAndRiskFactors'
    elif data_type == 'combinedDoctorsNotesAndRiskFactors':
        # Determine the correct field names for Doctor's Notes:
        field_names = field_names = field_names_for_combined_doctors_notes_and_risk_factors_data
        # Assign the data_type specific string for the file paths:
        data_type_string = 'CombinedDoctorsNotesAndRiskFactors'
    elif data_type == 'combinedConnersAndDoctorsNotesAndRiskFactors':
        # Determine the correct field names for Doctor's Notes:
        field_names = field_names = field_names_for_combined_conners_and_doctors_notes_and_risk_factors_data
        # Assign the data_type specific string for the file paths:
        data_type_string = 'combinedConnersAndDoctorsNotesAndRiskFactors'
    
    # load dataset:
    dataFrame = pd.read_csv(data_path, header=None, names=field_names)

    # Feature selection:
    X = dataFrame[field_names[0:-1]]            # X is another pandas.DataFrame which resembles a 2-D array in which rows could be accessed by their indexes. (And ofcourse columns could be accessed by their column_names)
    # Set the target variable (the classifier variable):
    y = dataFrame["Label"]                      # y is in the type of pandas.Series is a one-dimensional ndarray with axis labels

    if cross_validation == False:
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
                randomized_search = RandomizedSearchCV(estimator=supportVectorMachineModel, param_distributions=hyperParameters, n_iter=20, cv=5, n_jobs=-1)

                # Fit the random search object to the data:
                randomized_search.fit(X_train, y_train)
                # Create a variable for the best model:
                bestModelHypertuned = randomized_search.best_estimator_
                # Print the best values for the hyperparameters:
                hyperParameterFilePath = r'C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\Output\SupportVectorMachines\HyperParameters' + data_type_string + '.txt'
                # Wipe the content of the hyper parameters file which is the result of the previous execution:
                with open(hyperParameterFilePath,'w',newline='',encoding='UTF-8') as FileWritten:
                    FileWritten.write("")
                # Open the file in which the hyper parameters will be written in append mode:
                with open(hyperParameterFilePath,'a',newline='',encoding='UTF-8') as FileWritten:
                    FileWritten.write('Best hyperparameters: ' + str(randomized_search.best_params_))

            else:       #################################### Tuning with Grid Search #######
                # Create dictionary for the hyper parameters which we want to optimize:
                hyperParameters = {'C':         [0.1,1, 10, 100], 
                                   'gamma':     [1,0.1,0.01,0.001],
                                   'kernel':    ['rbf', 'poly', 'sigmoid']}
                # Create a SVM classifier which will be used for optimization:
                supportVectorMachineModel = svm.SVC()
                # Utilize the random search function provided by Scikit-learn in order to find the best hyperparameters:
                grid_search = GridSearchCV(supportVectorMachineModel,hyperParameters,refit=True,verbose=2, n_jobs=-1)                                 # The number of cross-validation folds to be used.
                # Fit the random search object to the data:
                grid_search.fit(X_train, y_train)
                # Create a variable for the best model:
                bestModelHypertuned = grid_search.best_estimator_
                # Print the best values for the hyperparameters:
                hyperParameterFilePath = r'C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\Output\SupportVectorMachines\HyperParameters' + data_type_string + '.txt'
                # Wipe the content of the hyper parameters file which is the result of the previous execution:
                with open(hyperParameterFilePath,'w',newline='',encoding='UTF-8') as FileWritten:
                    FileWritten.write("")
                # Open the file in which the hyper parameters will be written in append mode:
                with open(hyperParameterFilePath,'a',newline='',encoding='UTF-8') as FileWritten:
                    FileWritten.write('Best hyperparameters: ' + str(grid_search.best_params_))

            # Set the classifier object as the model with the best performance:
            classifierObject = bestModelHypertuned
        ####### END OF Hyper-parameter Tuning ###############################################

        elif not hyper_parameter_tuning:
            # Create SVM classifier object:
            classifierObject = svm.SVC(kernel='rbf', gamma='auto', probability=True)
        
        # Train the classifier:
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
            FileWritten.write("F-1 Score: " + str(metrics.f1_score(y_test, y_pred,average='macro')) + '\n')
        ###### Create the confusion matrix:
        confusionMatrix = confusion_matrix(y_test,y_pred)
        confusionDisplayObject = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix)
        confusionDisplayObject.plot()
        confusionDisplayObject.ax_.set(xlabel='Tahmin Edilen Sınıf', ylabel='Gerçek Sınıf')
        confusionDisplayObject.figure_.savefig(r'C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\Output\SupportVectorMachines\ConfusionMatrix' + data_type_string + '.png')
        ###### Permutation Feature Importance:
        # Permutation feature importance is defined as "the difference between the baseline metric and metric from permutating the feature column".
        # It is used for non-linear kernels!!!
        perm_importance = permutation_importance(classifierObject, X, y)        # Returns a Dictionary-like object. 
        # Normalize the feature importances such that the sum of all feature importances is 1.0, therefore the feature importances can be understood as percentages:
        perm_importance_normalized = perm_importance.importances_mean/perm_importance.importances_mean.sum()
        # Organize features for the plot:
        feature_names = X.columns
        features = np.array(feature_names)
        # Sort to plot in the order of importance:
        sorted_idx = perm_importance_normalized.argsort()
        # Plot:
        fig = plt.figure(figsize=(16, 16), layout='compressed')       # Create & initialize a figure with a size of 12x12 inches and a compressed layout
        plt.title('Maddelerim Permutasyonsal Önem Sıralaması',fontsize=20)
        plt.barh(features[sorted_idx], perm_importance_normalized[sorted_idx], color='b', align='center')
        plt.xlabel('Maddenin Modele Kıyasla Önemi', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.savefig(r'C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\Output\SupportVectorMachines\SupportVectorMachines' + data_type_string + 'PermutationFeatureImportance' + '.png')
        ########################################### Plot ROC Curve #################################################
        # Predict probabilities for the test set:
        probabilitiesOfClasses = classifierObject.predict_proba(X_test)
        # Keep probablities for only the positive outcome: ADHD_positive
        probabilitiesOfClasses_pos_class = probabilitiesOfClasses[:,1]    
        # Generate probabilities for 45-degrees line (45-degrees line will be used as a reference!)
        noskill_probabilities = [0 for number in range(len(y_test))]
        # Calculate AUC Scores:
        supportVectorMachinesAUC = metrics.roc_auc_score(y_test, probabilitiesOfClasses_pos_class)
        noSkillAUC = metrics.roc_auc_score(y_test,noskill_probabilities)
        ## Write AUC score into the performance metrics file:
        # Open the file in which the performance metrics will be written in append mode:
        with open(performanceMetricsFilePath,'a',newline='',encoding='UTF-8') as FileWritten:
            FileWritten.write("AUC Score: " + str(supportVectorMachinesAUC))
        # Calculate the related data which are false positive rate and true positive rate for the test set:
        falsePosRate_supportVectorMachines, truePosRate_supportVectorMachines,_ = metrics.roc_curve(y_test, probabilitiesOfClasses_pos_class, pos_label='ADHD_positive')
        # Calculate the related data for 45-degrees line:
        falsePosRate_noSkill, truePosRate_noSkill,_ = metrics.roc_curve(y_test, noskill_probabilities, pos_label='ADHD_positive')
        # Plot the ROC Curve of the decision tree predictions and no skill predictions which is a 45-degrees line (since it either predicts positive or negative for all data points) as a reference :
        plt.figure(3)
        plt.title(label='Receiver Operating Characteristic (ROC) Eğrisi')
        plt.xlabel('Yanlış Pozitif Oranı')
        plt.ylabel('Doğru Pozitif Oranı')
        plt.plot(falsePosRate_supportVectorMachines, truePosRate_supportVectorMachines, label='Destek Vektör Makineleri, AUC:' + str(supportVectorMachinesAUC), linestyle='solid', linewidth=3)
        plt.scatter(falsePosRate_supportVectorMachines,truePosRate_supportVectorMachines, linewidth=0.5)
        plt.plot(falsePosRate_noSkill, truePosRate_noSkill, label='Algoritma Kullanılmadan, AUC:' + str(noSkillAUC), linestyle='dashed', linewidth=2)
        plt.legend(loc=4)
        plt.gca().set_facecolor('#cbced0')
        plt.grid(visible=True, color='#ffffff') 
        # Save the plot on PNG file:
        plt.savefig(r'C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\Output\SupportVectorMachines\ROC_Curve' + data_type_string + '.png')
    elif cross_validation == True:
        ############################# K-fold Cross Validation ###################################################################
        accuracyScores = []
        precisionScores = []
        recallScores = []
        fOneScores = []
        aucScores = []
        # Initialize the file which will hold hyper parameters:
        if hyper_parameter_tuning:
            hyperParameterFilePath = r'C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\Output\SupportVectorMachines\KFoldHyperParameters' + data_type_string + '.txt'
            # Wipe the content of the hyper parameters file which is the result of the previous execution:
            with open(hyperParameterFilePath,'w',newline='',encoding='UTF-8') as FileWritten:
                FileWritten.write("")

        kf = KFold(n_splits=10, shuffle=True)
        for train_indeces, test_indeces in kf.split(X):
            X_train, X_test = X.iloc[train_indeces,:], X.iloc[test_indeces,:]
            y_train, y_test = y.iloc[train_indeces], y.iloc[test_indeces]

            if hyper_parameter_tuning:
                if search_type == 'randomized': ### Tuning with Randomized Search ##############
                    # Create dictionary for the hyper parameters which we want to optimize:
                    hyperParameters = {'C':         uniform(0.1, 10), 
                                       'gamma':     ['scale', 'auto'] + list(logspace(-3, 3, 50)),
                                       'kernel':    ['linear', 'rbf', 'poly', 'sigmoid']}
                    # Create a SVM classifier which will be used for optimization:
                    supportVectorMachineModel = svm.SVC()
                    # Utilize the random search function provided by Scikit-learn in order to find the best hyperparameters:
                    randomized_search = RandomizedSearchCV(estimator=supportVectorMachineModel, param_distributions=hyperParameters, n_iter=20, cv=5, n_jobs=-1)

                    # Fit the random search object to the data:
                    randomized_search.fit(X_train, y_train)
                    # Create a variable for the best model:
                    bestModelHypertuned = randomized_search.best_estimator_
                    # Print the best values for the hyperparameters:
                    # Open the file in which the hyper parameters will be written in append mode:
                    with open(hyperParameterFilePath,'a',newline='',encoding='UTF-8') as FileWritten:
                        FileWritten.write('Best hyperparameters: ' + str(randomized_search.best_params_) + '\n')

                else:       #################################### Tuning with Grid Search #######
                    # Create dictionary for the hyper parameters which we want to optimize:
                    hyperParameters = {'C':         [0.1,1, 10, 100], 
                                       'gamma':     [1,0.1,0.01,0.001],
                                       'kernel':    ['rbf', 'poly', 'sigmoid']}
                    # Create a SVM classifier which will be used for optimization:
                    supportVectorMachineModel = svm.SVC()
                    # Utilize the random search function provided by Scikit-learn in order to find the best hyperparameters:
                    grid_search = GridSearchCV(supportVectorMachineModel,hyperParameters,refit=True,verbose=2, n_jobs=-1)           # The number of cross-validation folds to be used.
                    # Fit the random search object to the data:
                    grid_search.fit(X_train, y_train)
                    # Create a variable for the best model:
                    bestModelHypertuned = grid_search.best_estimator_
                    # Print the best values for the hyperparameters:
                    # Open the file in which the hyper parameters will be written in append mode:
                    with open(hyperParameterFilePath,'a',newline='',encoding='UTF-8') as FileWritten:
                        FileWritten.write('Best hyperparameters: ' + str(grid_search.best_params_) + '\n')

                # Set the classifier object as the model with the best performance:
                classifierObject = bestModelHypertuned
            ####### END OF Hyper-parameter Tuning ###############################################  
            elif not hyper_parameter_tuning:
                # Create SVM classifier object:
                classifierObject = svm.SVC(kernel='rbf', gamma='auto', probability=True)

            # Train the Classifer:
            classifierObject = classifierObject.fit(X_train,y_train)
            #Predict the response for test dataset
            y_pred = classifierObject.predict(X_test)
            # Calculate and Append the performance metrics for current fold:
            accuracyScores.append(metrics.accuracy_score(y_test, y_pred))
            precisionScores.append(metrics.precision_score(y_test, y_pred,average='macro'))
            recallScores.append(metrics.recall_score(y_test, y_pred,average='macro'))
            fOneScores.append(metrics.f1_score(y_test, y_pred,average='macro'))
            ## Calculate the Area Under Curve for the current fold:
            # Predict probabilities for the test set:
            probabilitiesOfClasses = classifierObject.predict_proba(X_test)
            # Keep probablities for only the positive outcome: ADHD_positive
            probabilitiesOfClasses_pos_class = probabilitiesOfClasses[:,1]    
            # Generate probabilities for 45-degrees line (45-degrees line will be used as a reference!)
            noskill_probabilities = [0 for number in range(len(y_test))]
            # Calculate AUC Scores:
            supportVectorMachinesAUC = metrics.roc_auc_score(y_test, probabilitiesOfClasses_pos_class)
            aucScores.append(supportVectorMachinesAUC)
        ### Cross-validate: calculate the MEANS
        performanceMetricsFilePath = r'C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\Output\SupportVectorMachines\KFoldPerformanceMetrics' + data_type_string + '.txt'
        # Wipe the content of the preformance metrics file which is the result of the previous execution:
        with open(performanceMetricsFilePath,'w',newline='',encoding='UTF-8') as FileWritten:
            FileWritten.write("")
        # Open the file in which the performance metrics will be written in append mode:
        with open(performanceMetricsFilePath,'a',newline='',encoding='UTF-8') as FileWritten:
            FileWritten.write("Accuracy: "  + str(np.mean(accuracyScores)) + '\n')
            FileWritten.write("Precision: " + str(np.mean(precisionScores)) + '\n')
            FileWritten.write("Recall: "    + str(np.mean(recallScores)) + '\n')
            FileWritten.write("F-1 Score: " + str(np.mean(fOneScores)) + '\n')
            FileWritten.write("AUC Score: " + str(np.mean(aucScores)))              

def main():
    supportVectorMachinesUtilizingScikit(data_type='parent'                                      , data_path=r"C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\ConnersParentData.csv",                                 hyper_parameter_tuning=True,   search_type='randomized', cross_validation=False)
    #supportVectorMachinesUtilizingScikit(data_type='teacher'                                     , data_path=r"C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\ConnersTeacherData.csv",                                hyper_parameter_tuning=True,   search_type='randomized', cross_validation=False)
    #supportVectorMachinesUtilizingScikit(data_type='doctors'                                     , data_path=r"C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\DoctorsNotesData.csv",                                  hyper_parameter_tuning=True,   search_type='randomized', cross_validation=False)
    #supportVectorMachinesUtilizingScikit(data_type='risk'                                        , data_path=r"C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\RiskFactorsData.csv",                                   hyper_parameter_tuning=False,  search_type='randomized', cross_validation=False)
    #supportVectorMachinesUtilizingScikit(data_type='combinedConnersParentAndTeacher'             , data_path=r"C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\ConnersData.csv",                                       hyper_parameter_tuning=True,   search_type='randomized', cross_validation=False)
    #supportVectorMachinesUtilizingScikit(data_type='combinedConnersAndDoctorsNotes'              , data_path=r"C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\CombinedConnersAndDoctorsNotes.csv",                    hyper_parameter_tuning=True,   search_type='randomized', cross_validation=False)
    #supportVectorMachinesUtilizingScikit(data_type='combinedConnersAndRiskFactors'               , data_path=r"C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\CombinedConnersAndRiskFactors.csv",                     hyper_parameter_tuning=True,   search_type='randomized', cross_validation=False)
    #supportVectorMachinesUtilizingScikit(data_type='combinedDoctorsNotesAndRiskFactors'          , data_path=r"C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\CombinedDoctorsNotesAndRiskFactors.csv",                hyper_parameter_tuning=False,   search_type='randomized', cross_validation=False)
    #supportVectorMachinesUtilizingScikit(data_type='combinedConnersAndDoctorsNotesAndRiskFactors', data_path=r"C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\CombinedConnersAndDoctorsNotesAndRiskFactors.csv",      hyper_parameter_tuning=True,   search_type='randomized', cross_validation=False)

if __name__ == "__main__":
    main()