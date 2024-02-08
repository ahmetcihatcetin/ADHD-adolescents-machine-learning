### Import Libraries/Modules
# Data Analysis & Manipulation:
import pandas as pd

# Machine Learning Modules:
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# Modules for Hyper-parameter Tuning:
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from numpy import logspace

# Modules for Performance Metrics:
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Import necessary libraries for plotting the ROC Curve:
import matplotlib.pyplot as plt

# Import necessary libraries for K-fold Cross Validation
from sklearn.model_selection import KFold
from numpy import mean

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
for field in field_names_for_combined_conners_and_doctors_notes_data:
    field.encode()

# Encode the data in utf-8:
for field in field_names_for_combined_conners_and_risk_factors_data:
    field.encode()

# Encode the data in utf-8:
for field in field_names_for_combined_conners_and_doctors_notes_and_risk_factors_data:
    field.encode()

def naiveBayesGaussianUtilizingScikit(data_type, data_path, hyper_parameter_tuning=False, search_type=None, cross_validation=False):
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30) # 70% training and 30% test

        ####### Hyper-parameter Tuning ######################################################
        if hyper_parameter_tuning:
            if search_type == 'randomized': ### Tuning with Randomized Search ##############
                pass
            else:       #################################### Tuning with Grid Search ###########
                cv_method = RepeatedStratifiedKFold(n_splits=5, 
                                                    n_repeats=3)
                params_NB = {'var_smoothing': logspace(0,-9, num=100)}
                grid_search = GridSearchCV( GaussianNB(), 
                                            param_grid=params_NB, 
                                            cv=cv_method,
                                            verbose=1,
                                            n_jobs=-1, 
                                            scoring='accuracy')
                # Fit the grid search object to the data:
                grid_search.fit(X_train, y_train)
                # Create a variable for the best model:
                bestModelHypertuned = grid_search.best_estimator_
                # Print the best values for the hyperparameters:
                print('Best hyperparameters:',  grid_search.best_params_)
            
            # Set the classifier object as the model with the best performance:
            classifierObject = bestModelHypertuned
        ####### END OF Hyper-parameter Tuning ###############################################
        elif not hyper_parameter_tuning:
            # Create a Gaussian classifier:
            classifierObject = GaussianNB()

        # Train the Gaussian classifier:
        classifierObject.fit(X_train, y_train)

        # Predict the labels for test dataset:
        y_pred = classifierObject.predict(X_test)

        ################################### Calculate Performance Metrics #######################################################
        ### Calculate numeric performance metrics and write them into the file with the path of performanceMetricsFilePath:
        performanceMetricsFilePath = r'C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\Output\NaiveBayes\PerformanceMetrics' + data_type_string + '.txt'
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
        ConfusionMatrixDisplay(confusion_matrix=confusionMatrix).plot().figure_.savefig(r'C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\Output\NaiveBayes\ConfusionMatrix' + data_type_string + '.png')
        ###### Plot the ROC Curve:
        y_pred_proba = classifierObject.predict_proba(X_test)[::,1]
        fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba, pos_label='ADHD_positive')
        auc = metrics.roc_auc_score(y_test, y_pred_proba)
        plt.figure(2)
        plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
        plt.legend(loc=4)
        plt.savefig(r'C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\Output\NaiveBayes\ROC_Curve' + data_type_string + '.png')
        ################################### END OF Performance Metrics #########################################################
    elif cross_validation == True:
        ############################# K-fold Cross Validation ###################################################################
        accuracyScores = []
        precisionScores = []
        recallScores = []
        fOneScores = []

        kf = KFold(n_splits=10, shuffle=True)
        for train_indeces, test_indeces in kf.split(X):
            X_train, X_test = X.iloc[train_indeces,:], X.iloc[test_indeces,:]
            y_train, y_test = y.iloc[train_indeces], y.iloc[test_indeces]

            if hyper_parameter_tuning:
                if search_type == 'randomized': ### Tuning with Randomized Search ##############
                    pass
                else:       #################################### Tuning with Grid Search ###########
                    cv_method = RepeatedStratifiedKFold(n_splits=5, 
                                                        n_repeats=3)
                    params_NB = {'var_smoothing': logspace(0,-9, num=100)}
                    grid_search = GridSearchCV( GaussianNB(), 
                                                param_grid=params_NB, 
                                                cv=cv_method,
                                                verbose=1,
                                                n_jobs=-1, 
                                                scoring='accuracy')
                    # Fit the grid search object to the data:
                    grid_search.fit(X_train, y_train)
                    # Create a variable for the best model:
                    bestModelHypertuned = grid_search.best_estimator_
                    # Print the best values for the hyperparameters:
                    print('Best hyperparameters:',  grid_search.best_params_)

                # Set the classifier object as the model with the best performance:
                classifierObject = bestModelHypertuned
            ####### END OF Hyper-parameter Tuning ###############################################
            elif not hyper_parameter_tuning:
                # Create a Gaussian classifier:
                classifierObject = GaussianNB()

            # Train the Gaussian classifier:
            classifierObject.fit(X_train, y_train)

            # Predict the labels for test dataset:
            y_pred = classifierObject.predict(X_test)

            # Calculate and Append the performance metrics for current fold:
            accuracyScores.append(metrics.accuracy_score(y_test, y_pred))
            precisionScores.append(metrics.precision_score(y_test, y_pred,average='macro'))
            recallScores.append(metrics.recall_score(y_test, y_pred,average='macro'))
            fOneScores.append(metrics.f1_score(y_test, y_pred,average='macro'))

        ### Cross-validate: calculate the MEANS
        performanceMetricsFilePath = r'C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\Output\NaiveBayes\KFoldPerformanceMetrics' + data_type_string + '.txt'
        # Wipe the content of the preformance metrics file which is the result of the previous execution:
        with open(performanceMetricsFilePath,'w',newline='',encoding='UTF-8') as FileWritten:
            FileWritten.write("")
        # Open the file in which the performance metrics will be written in append mode:
        with open(performanceMetricsFilePath,'a',newline='',encoding='UTF-8') as FileWritten:
            FileWritten.write("Accuracy: "  + str(mean(accuracyScores)) + '\n')
            FileWritten.write("Precision: " + str(mean(precisionScores)) + '\n')
            FileWritten.write("Recall: "    + str(mean(recallScores)) + '\n')
            FileWritten.write("F-1 Score: " + str(mean(fOneScores)))

def main():
    naiveBayesGaussianUtilizingScikit('parent', r"C:\Users\ahmet\Documents\ADHD Machine Learning\ADHD-adolescents-machine-learning\Data\ConnersParentData.csv", hyper_parameter_tuning=True, search_type='grid', cross_validation=True)

if __name__ == "__main__":
    main()