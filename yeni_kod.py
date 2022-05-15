"""IMPORT EDİLECEK KÜTÜPHANELER
json
numpy
keras
tensorflow
pickle
nltk
snowballstemmer
"""
import json  # datasetlerimiz json metin dosyası formatında olacak.
import numpy as np
import random
import pickle  # modeller pickle dosyası şeklinde kaydedilecek.
from tensorflow.python.keras.models import Sequential  # modellerimizdeki katmanların lineer bir dizisini tutacağız.
from tensorflow.python.keras.layers import Dense, Embedding, Dropout, Activation, \
    GlobalAveragePooling1D  # katmanlarımız için gerekli olan yapılar.
from tensorflow.python.keras.optimizers import \
    gradient_descent_v2  # gradient descent optimizasyonları için kullanacağız.
from tensorflow.python.keras.models import load_model
import nltk  # dil işleme kütüphanemiz.
from snowballstemmer import TurkishStemmer  # türkçe destekle kelime köklerini ayıracağız.

nltk.download("punkt")  # cümleleri kelimelere aıyrmak için öncelikle nltk modülümüzü indiriyoruz.

with open("stop_words2.json", encoding="utf8") as file:  # dataset dosyamızı utf-8 olarak açıyoruz.
    movies = json.load(file)  # data değişkenine json dosyası açıldı.

stemmer = TurkishStemmer()  # kök ayırma işlemini türkçe destekle yapıyoruz.

synopsis = []  # ayıklanmış özetlerin tutulacağı liste değişkeni.
names = []  # json dosyamızdaki film isimlerinin tutulacağı liste.
documents = []  # json dosyamızdaki isim ve özetlerin beraber tutulacağı liste.
#ignore_letters = ["!", "'", "?", ",", "."]  # cümle içindeki bu noktalama işaretlerini atlıyoruz.

"""
for movie in movies["movies"]:
    word = nltk.word_tokenize(movie["synopsis"])  # json dosyamızdaki özetlerdeki cümleleri kelimelere ayırıyoruz.
    synopsis.extend(word)  # ayırdığımız kelimeleri listeye ekliyoruz.
    documents.append((word, movie["name"]))  # ayıklanmış kelime listemizi ve ait olduğu film adını beraber ekliyoruz.
    if movie["name"] not in names:
        names.append(movie["name"])  # film adını listeye ekliyoruz.
"""
for movie in movies["movies"]:
    word= nltk.word_tokenize(movie["Ozet"])
    synopsis.extend(word)
    documents.append((word,movie["Film Adı"]))
    if movie["Film Adı"] not in names:
        names.append(movie["Film Adı"])


#synopsis = [stemmer.stemWord(w.lower()) for w in synopsis if w not in ignore_letters]  # kelimelerin köklerini alma,
# harfleri küçültme ve atlayacağımız işaretleri kontrol etme işlemlerini yapıp kelimelerimizi düzenliyoruz.
synopsis = sorted(
    list(synopsis))  # kelimelerimizi kümeye çevirip sıralıyoruz. küme olduğu için aynı elemanlar tekrar sayılmadı.
names = sorted(list(names))  # aynı işlemi film isimlerimiz için yapıyoruz.
print(len(names), "adet film var.")
print(len(synopsis), "adet ayıklanmış kelime var.")

pickle.dump(synopsis, open("synopsis.pkl",
                           "wb"))  # kelimelerimizi arayüzde kullanabilmek için binary formda pickle dosyalarına
# kaydediyoruz.
pickle.dump(names, open("names.pkl", "wb"))  # aynı şekilde etiketlerimizi de kaydediyoruz.



##################################################################################################################################3




training_data = []
output_empty = [0] * len(names)  # boş bir output listemiz. kaç tane film varsa o kadar bir uzunluğu olacak.

for doc in documents:  # eğitim setlerimizi oluşturacağız.
    bag = []  # kelimelerimizi 0 ve 1 değerlerine çevirip öğrenmeyi sağlayacağız.
    pattern_words = doc[0]  # ayıklanmış kelimelerimizi alıyoruz.
    pattern_words = [stemmer.stemWord(word.lower()) for word in #FEHİME
                     pattern_words]  # kelimelerimizi küçük harflere çeviriyoruz.
    for word in synopsis:
        if word in pattern_words:  # eğer en başta ayıkladığımız kelime eğitim seti için ayıklanmış kelimeler
            # içinde varsa 1, eğer yoksa 0 ekliyoruz.
            bag.append(1)
        else:
            bag.append(0)

    output_row = list(output_empty)  # eğitim sonucu çıkış değerimiz.
    output_row[names.index(
        doc[1])] = 1  # lineer hale getirdiğimiz kelimenin film ismini de 1 yapıyoruz, diğer film isimleri 0.

    training_data.append([bag, output_row])  # eğitim datamıza hem ayıklanmış kelimelerimizin hem de film isimlerimizin
    # lineer hale çevrilmiş 0-1 halini ekliyoruz. modelimizi bu datayı kullanarak eğiteceğiz.

random.shuffle(training_data)  # eğitim datamızı daha iyi eğitebilmek için karıştırıyoruz.
training_data = np.array(training_data)  # eğitim datamızı array'e çeviriyoruz.
train_x = list(training_data[:,
               0])  # training datamızın bütün 0. indislerini train_x listemize atıyoruz. özetler burada yer alıyor
train_y = list(training_data[:,
               1])  # train datamızın bütün 1. indislerini train_y listemize atıyoruz. film isimleri burada yer alıyor
print("train data created.")

model = Sequential()  # katmanlı modelimiz.
model.add(Dense(len(train_x[0]), input_shape=(len(train_x[0]),),
                activation="relu"))  # giriş katmanı 128 nöron içeriyor ve fonksiyonu relu.
model.add(Dropout(0.5))  # ezberi önlemek için dropout değerimiz 0.5.

# arada bir katman daha vardı nöron sayısı belirsiz

model.add(Dense(len(train_y[0]),
                activation="softmax"))  # çıkış katmanımız etiket sayısı kadar nörona sahip ve fonksiyonu softmax.

sgd = gradient_descent_v2.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9,
                              nesterov=True)  # modelin optimizasyonu için SGD kullanıyoruz.
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])  # modeli compile ediyoruz.

fittedmodel = model.fit(np.array(train_x), np.array(train_y), epochs=20, batch_size=500, verbose=1)  # modelimizin formu.
model.save("bitirme.h5", fittedmodel)  # modelimizi kaydediyoruz.

print("model created.")

########################################################################################################################
#
# model = load_model("bitirme.h5")  # modelimizi yüklüyoruz.
# dataset = json.loads(open("bitirme_dataset.json").read())  # datasetimizi yüklüyoruz.
# synopsis = pickle.load(open("synopsis.pkl", "rb"))  # özet kelimelerimizi yüklüyoruz.
# names = pickle.load(open("names.pkl", "rb"))  # film isimlerimizi yüklüyoruz.
#
# ###############################################################################################################
#
# def raw_sentence(sentence):  # cümle düzenleyeceğiz.
#     sentence_words = nltk.word_tokenize(sentence)  # kullanıcının girdiği özeti kelimelere ayırıyoruz.
#     sentence_words = [stemmer.stemWord(word.lower()) for word in sentence_words]  # kelimeleri köklere ayırıyoruz.
#     return sentence_words
#
# def words_bag(sentence, words, show_details=True):  # cümle içindeki kelimelerin 0-1 karşılığını döndüreceğiz.
#     sentence_words = raw_sentence(sentence)  # önce cümleleri kelimelere ayırmak için fonksiyonu çağırıyoruz.
#     bag = [0] * len(words)
#     for s in sentence_words:
#         for i, word in enumerate(words):
#             if word == s:  # kullanıcıdan aldığımız kelimeler ile fonksiyona gönderilen kelimeler aynı ise 1 atıyoruz.
#                 bag[i] = 1
#     return np.array(bag)
#
# def predict(sentence):  # tahminleme yapacağımız fonksiyon.
#     ERROR_TRESHOLD = 0.04
#     p = words_bag(sentence, synopsis, show_details=False)
#     res = model.predict(np.array([p]))[0]
#     results = [[i, r] for i, r in enumerate(res) if r > ERROR_TRESHOLD]
#     results.sort(key=lambda x: x[1], reverse=True)
#     r = results[0]
#     r2 = results[1]
#     r3 = results[2]
#     return_list = {"movie": names[r[0]], "probability": str(r[1])}
#     return_list2 = {"movie": names[r2[0]], "probability": str(r2[1])}
#     return_list3 = {"movie": names[r3[0]], "probability": str(r3[1])}
#     print(return_list)
#     print(return_list2)
#     print(return_list3)
#     return return_list
#
# def response(ints, intents_json):  # cevap verme fonksiyonumuz.
#     tag = ints["movie"]
#     intents_list = intents_json["movies"]  # film isimlerimizi alıyoruz.
#     for i in intents_list:
#         if i["name"] == tag:  # eğer şu anki film ismimiz listemizde varsa,
#             result = [i["name"], i["synopsis"], i["genres"], i["actors"], i["directors"]]
#             break
#     return result
#
# #########################################################################################################################
#
# msg = input('Lütfen film özetini girin. \n')  # kullanıcı özeti giriyor.
#
# if msg != " ":  # eğer boş bir özet girilmemişse
#
#     entry_word = nltk.word_tokenize(msg)  # özeti ayıklayıp küçük harflere dönüştürüyoruz.
#     entry_word = [stemmer.stemWord(word.lower()) for word in entry_word]
#
#     ints = predict(msg)  # özetimizi tahmin edilmesi için fonksiyona gönderiyoruz.
#     res = response(ints, dataset)  # cevap için fonksiyona gidiyoruz.
#     print("Film: ", res[0])
#     print("Türler: ", res[2])
#     print("Oyuncular: ", res[3])
#     print("Yönetmen: ", res[4])
#     print("\n", res[1])
#
# else:
#     print("Boş bir özet girdiniz.")
