*SpamDetection*

<div  dir='ltr'>

ابتدا لازم است که پکیج ها import شوند.  
</div>

```
import nltk
import string as st
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
```

<div  dir='ltr'>

همانظورکه میدانیم لازم است قیل از انجام هرگونه عملیات پردازش طبیعی متن، یم سری پیش پردازش بر روی ان انجام دهیم که به صورت مقابل تابع آن نشان داده می‌شود.

در این بخش پیش پردازش دو مرحله انجام می شود. مرحله اول اختصاص پیدا کرده است به حدف Punctuation ها و مرحله دوم اختصاص پیدا کرده است به حدف StopWord ها که توسط nltk.corpus لود شده و با یک حلقه for تمامی این ها از هر پیام حذف می شود.

</div>

```
def preprocessing(message):
    messageNoPunctuation = ''.join([char  for char in message  if char not in string.punctuation])
    messageNoStopWords = [word for word in messageNoPunctuation.split() if word.lower() not in stopwords.words('english')]
    return messageNoStopWords
```

<div dir='rtl'>

در مرحله بعد دیتاست که شامل یک فایل message ها است به صورت زیر به صورت DataFrame بارگزاری می شود تا بر روی ان پیش پردازش ها صورت بگیرد.
طور که میدانیم لازم است داده های متنی خود را به یک الگوریتم یادگیری نظر Naive bayse وارد کنیم تا مدلی یادگیری شو ئ از این پس spam یا ham بودن پیام را پیش بینی کتد. اما داده ها برای ورود به الگوریتم لازم است بردار عددی باشند. چون سیستم عدد می فهمد. از این رو با استفاده از کلاس CountVectorizer هر message به کد تبدیل شده تا از این طریق مدل با یک بردار اموزش بببیند. لذا به صورت زیر خواهیم داشت. در واقع با استفاده از CountVectorizer یک دیکشنری تخت عنوان Bag Of Words ساخته می شود و به کلمه یک عدد یا کد الصاق می شود. 
</div>

```
df = pd.read_csv('Dataset/SMSSpamCollection', sep='\t', names=['label', 'text'])
bow = CountVectorizer(analyzer=preprocessing).fit(df['text'])
```
<div dir='rtl'>
در ادامه در خطوط زیر برخی از متد های CountVetorizer مورد بررسی و ارزیابی قرار میگیرد.
ابتدا پیام شماره 4 توسط bag of word آموزش دیده به یک دیکشنری کلمات یا توکن های مربوطه اش بتدیل می‌شود. همچنین در خط اخر میتوان با متد get_feature_name_out() متوجه شد هر کد برای کدام واژه است. همچنین با متد bow.vocabulary_ میتوان به کل دیکشنری دست یافت.
</div>

```
print(bow.vocabulary_)
bow_4 = bow.transform([df['text'][3]])
print(bow_4)
print(bow_4.shape)
print(bow.get_feature_names_out()[4068])
```

<div dir='rtl'>

در مرحله بعد لازم است با bow برازش شده، پیام ها transform می شود:

</div>

```
messages_bow = bow.transform(df['text'])
```
<div  dir="ltr">

با استفاده از کلاس TFIDFTransformer میزان تکرار یا فروانی هر واژه را مشخص می کند. کلمه ای که با فراوانی بیشتری در پیان وجود داشته باشد مقدار یا ارزش کمتری به خود میگیرد و کلمه ای که مفدار یا ارزش بشتری به خود میگیرد اگر کمتر در پیام ها تکرار شده باشد. 
</div>

```
messages_bow = bow.transform(df['text'])
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer =TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow_4)
print(tfidf4)
```
<div  dir="ltr">


 
</div>




















