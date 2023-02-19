# ASVspoof

Stajımda denemiş olduğum sahte ses tespiti uygulamasıdır. ASVspoof veri setleri üzerinde uygulanmıştır. Öznitelik çıkarım yöntemi olarak MFCC(Mel Frequency Kepstrum Coefficients) kullanılmıştır.  

Ses dosyalarının uzunluklarının birbirinden farklı olması sebebiyle MFCC matrislerinin boyutları birbiriyle uyumlu çıkmamaktadır. Bu yüzden uzun süreli ses sinyalini kesme işlemi yapılmıştır. Bunun için Librosa paketinden yararlanılarak her ses dosyasının süresi bir histogram üzerinde gösterilmiştir. Histogram üzerinden ses sürelerinin eşitliğinin en yoğun olduğu ve kesme sonucunda veri kaybının en az olacağı bölge seçildi. Bu bölge maksimum ses süresidir. Bunun altında kalanlara sıfır ile doldurma işlemleri yapılırken üstünde kalanlara da kesme işlemi yapılarak eşitlenmiştir. MFCC’nin tek boyutlu olarak kalması için de her saniyenin MFCC öznitelikleri diğer saniyenin yanına yazılarak süre x 13 tane öznitelik elde edilmiş oldu.  

İşlemler sonucunda elde edilen veri setini bir JSON dosyasında tutarak sürekli olarak ses dosyalarını okuma ve okunan sinyallerin öznitelik çıkarımı işlemine sokulması engellenmek istenmiştir.  

Oluşturulan verisetleri farklı modeller üzerinde denenmiştir.
