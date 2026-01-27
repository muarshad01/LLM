## LLM finetuning approaches

* __Finetuning__: Adapting a pretrained model to a specific task by training the model on additional data.

#### What is LLM Fine-tuning?

* Fine-tuning LLM involves the additional training of a pre-existing model,
which has previously acquired patterns and features from an extensive data set, using a smaller, domain-specific dataset. In the context of "LLM Fine-tuning," LLM denotes a "Large Language Model," such as the GPT series by OpenAI.
 
***

* 5:00

#### Finetuning Types
1. Instruction finetuning
* Training a language on a set of tokens using specific instructions
* Can handle broader set of tasks
* Larger datasets, greater computational power needed
* LoRA / QLoRA
* LoRA: Two smaller martices that approximate a larger martirx are fine-tuned.
2. classification finetuning
* Model is trained to recognise a specific set of class labels, such as spam or no spam

***

* 10:00

***

* 15:00

* Step-1: Download and process dataset
* Step-2: Creating dataloaders


```python
import urllib.request
import zipfile
import os
from pathlib import Path

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    # Downloading the file
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    # Unzipping the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # Add .tsv file extension
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")

try:
    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
    print(f"Primary URL failed: {e}. Trying backup URL...")
    url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path) 
```


```python
import pandas as pd

df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
df
```

```python
print(df["Label"].value_counts())
```

***


20:25
ham subset with the spam so the total
20:28
now the new data frame is balance DF and
20:30
if you print out the balance DF value
20:33
counts you'll see that the number of ham
20:36
which is no spam emails are 747 and the
20:38
number of spam emails are
20:41
747 so after executing the previous code
20:44
to balance the data set we can see that
20:46
we now have an equal amount of spam and




***

* 20:00


no spam messages great this is exactly
20:50
what we wanted now we can go a step
20:52
further and we can to take a look at the
20:54
labels instead of having ham and spam uh
20:58
we can assign ham to be equal to zero
21:00
and spam to be equal to
21:02
one so these are the label encodings of
21:06
each of our emails so one note which
21:09
I've written here is that this process
21:11
is similar to converting text into token
21:13
IDs remember in uh when we pre-trained
21:17
the large language model we had a big
21:18
vocabulary the GPT vocabulary which had
21:22
uh more than 50,000 words in fact it had
21:24
50257 tokens and every token had a token
21:27
idid
21:29
this is a much simpler mapping we have
21:30
only two tokens kind of and they're
21:33
mapped to zero and
21:35
one now as we usually do in machine
Training, validation and testing dataset splits
21:37
learning tasks we'll take the data set
21:39
and split it into three parts we will
21:42
take this 747 data and we'll split it
21:44
70% will be used for training 10% will
21:47
will use for validation and 20% will use
21:50
for
21:52
testing um as I've mentioned here these
21:55
ratios are generally common in machine
21:56
learning to train adjust and eval at
21:58
models so here you can see I've written
22:01
a random split function what this
22:03
function is doing is that it just takes
22:05
the train end which is the fraction of
22:07
the training data which is train Frack
22:09
it's going to be 7 validation Frack is
22:11
going to be 0.2 so we first construct
22:14
the training data frame which is 70% of
22:17
the main data frame the validation data
22:20
frame is the remaining is 20% and the
22:22
test data frame is the remaining 10% is
22:25
the remaining 20% sorry the validation
22:27
data frame is the 10% the test data
22:30
frame is the remaining 20% of the full
22:32
data
22:33
frame and then when this function is
22:35
called out it will actually return the
22:37
training data frame the validation data
22:40
frame and the testing data frame so it
22:42
will return three data frames to us so
22:45
we now we can actually test this
22:47
function so we have this balance data
22:48
frame and we pass it into this function
22:50
called random split and once it is
22:52
passed into this function we also
22:54
specify the train fraction which is 7
22:57
and we specify the validation fraction
22:59
which is 0.1 and then we construct the
23:01
train data frame the validation data
23:03
frame and the test data frame so let us
23:06
check
23:08
uh whether the length makes sense so I'm
23:12
just going to type in new code here
23:14
which is length of train DF let's see
23:18
what's the
23:19
length it's
23:22
1045 yeah so length of train DF is
23:25
1045 uh because the total number of spam
23:28
and not spam is 747 + 747 which is
23:32
1494 then let me also print out length
23:37
of validation
23:40
DF and let me also print out length of
23:44
test DF and let me print out all of
23:48
these actually so that uh we can see
23:51
whether all of them indeed add up to
23:54
1494 Okay so
23:59
right so now I'm printing
24:01
this
24:18
uh okay so here you see that the length
24:21
of train DF is 1045 validation DF is 149
24:25
and test DF is 300 so let's add them
24:27
here 1045 1045 + 149 +
24:33
300 and let's see so it's 1494 and this
24:36
is 747 + 747 so that makes sense this is
24:40
kind of a check that the training
24:42
validation and testing data frames have
24:43
been created correctly I like to do
24:46
these checks once in a while to just
24:48
make sure that we are on the right track
24:49
in the code now what you can do is that
24:52
we'll also convert these data frames
24:54
into CSV files because we'll need to
24:56
reuse them later so we are just going to
24:58
use the 2 CSV function so you can search
25:02
this 2
25:04
CSV pandas what this does is that it can
25:07
take your data frame and it can convert
25:09
it into a CSV file I'll also add the
25:12
link to this in the information
25:14
description uh so you can apply this
25:17
function to the training data frame
25:18
validation data frame and also to the
25:20
testing data frame and then you can get
25:22
the train. CSV validation. CSV and the
25:25
test. CSV files so until now we have we
25:28
have reached a stage where we have
25:30
finished the first step which I had
25:32
mentioned over here and that first step
25:34
was to download and pre-process the data
25:36
set we downloaded the data set we
25:38
balanced the data set which was a part
25:40
of pre-processing so that the number of
25:42
spam and not spam are the same which is
25:45
747 and then we cons divided the data
25:48
set into training 70% validation 20% and
25:52
testing
Summary and next steps
25:54
10% and now in the next lecture what we
25:56
are going to see is that we are going to
25:58
first create data loaders so that we can
26:00
also do batch processing and it's
26:02
generally much better when you work with
26:04
large language models to use data
26:06
loaders and then we are going to see how
26:08
can we load the pre-trend Ws how can we
26:10
modify the model Etc you might be
26:13
thinking right how can a classification


***


26:15
task be done using large language models
26:18
so what happens towards the end is that
26:19
we usually fit another neural network at
26:22
the end of this and that will have a
26:24
softmax output so that the uh class so
26:28
that will be the
26:30
classification output so there is some
26:32
augmentation which we'll need to do here
26:34
so that the output is either spam or no
26:36
spam which is zero or one so we'll take
26:39
the architecture the same architecture
26:41
we worked on earlier but we'll augment
26:42
it we'll augment the end part of it the
26:45
architecture so that it's suitable for
26:48
classification so this brings us to the
26:50
end of the lecture thanks a lot everyone
26:52
I hope you are liking this approach of
26:54
whiteboard notes plus coding we have
26:57
covered a huge number of lectures in
26:58
this series before but if you are
27:00
landing onto this video series for the
27:02
first time it's fine I usually try to
27:04
make every lecture self-content but if
27:06
you want to revise any of your Concepts
27:08
or want to learn these Concepts from
27:10
scratch please go through the previous
27:12
lectures in this series also thanks a
27:15
lot everyone and I look forward to
27:16
seeing you in the next lecture

***








