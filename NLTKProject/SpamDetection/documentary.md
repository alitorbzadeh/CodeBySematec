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

در مرحله بعد دیتاست که شامل یک فایل message ها است به صورت زیر به صورت DataFrame بارگزاری می شود تا بر روی ان پیش پردازش ها صورت بگیرد. همچنین در مرحله بعد لازم است توسط کلاس CountVectorizer هر message به کد تبدیل شده تا از این طریق مدل با یک بردار اموزش بببیند. لذا به صورت زیر خواهیم داشت. در واقع با استفاده از CountVectorizer یک دیکشنری تخت عنوان Bag Of Words ساخته می شود و به کلمه یک عدد یا کد الصاق می شود. 
</div>

```
df = pd.read_csv('Dataset/SMSSpamCollection', sep='\t', names=['label', 'text'])
bow = CountVectorizer(analyzer=preprocessing).fit(df['text'])
```
