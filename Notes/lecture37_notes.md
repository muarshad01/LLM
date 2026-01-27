

* Pretrained LLMs are good at text completion
* They struggle with following instructions:
  * Fix the grammer in this text
  * convert this text into passive voice

***

* 5:00

* Training on a dataset where the (input, output) pairs are explicitly provided. It is also called __"supervidsed instruction finetuning".__

***

* 10:00

***

* 15:00


```python
import json
import os
import urllib


def download_and_load_file(file_path, url):

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    # The book originally contained this unnecessary "else" clause:
    #else:
    #    with open(file_path, "r", encoding="utf-8") as file:
    #        text_data = file.read()

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


file_path = "instruction-data.json"
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)

data = download_and_load_file(file_path, url)
print("Number of entries:", len(data))
```

***

* 20:00
  
20:32
999 you see here the instruction is what
20:35
is the opposite of complicated there is
20:36
no input over here so let's see how our
20:39
code deals with that so when you input
20:41
the data index by 999 into the format
20:43
input function it gives the model input
20:46
and this will be below is an instruction
20:48
and then we just have the instruction
20:49
there is no input and then you have to
20:52
give the desired response which is
20:53
response and then the output here so the
20:56
output in this case was an antonym of
20:59
complicated is simple right so then that
21:02
will be the desired response and then
21:05
the model input will be combined with
21:07
the desired response and then we'll get
21:09
this entire answer so this entire output
21:12
is a mix of the prompt and the response
21:15
and then this whole thing is fed to the
21:16
large language model when we do the fine
21:18
tuning
21:19
later for now I just want to show you
21:22
that the data set uh was first
21:26
formatted uh through the Alpac style
21:28
format and converted into a specific
21:30
prompt and response output like this you
21:33
can of course change this when I share
21:35
this code file with you there is no need
21:36
to stick with this particular prompt but
21:39
for the sake of Simplicity and to follow
21:41
the convention we are doing this in this
21:43
video because uh if you see the Stanford
21:46
alpaka repository there are about
21:49
29,000 um stars
21:51
and around 4,000 Forks which means that
21:55
it's a pretty popular repository and
21:57
many people use this this kind of a
21:59
configuration this this kind of a
22:01
configuration when they do find
Splitting dataset into train-test-validation
22:04
tuning great so up till now what we have
22:06
done is that we have converted our
22:08
instructions into alpaka format now the
22:10
next thing is we will split our data set
22:12
into training testing and validation
22:14
right so we have the data now uh which I
22:17
have showed over here this has 1100
22:19
pairs we'll split them into training
22:22
testing and validation so we are going
22:24
to use 85% for training 10% for testing
22:27
and the remaining five 5% for validation
22:30
so we'll just uh index or we'll just get
22:33
the train data the test data and the
22:35
validation data from our main data based
22:38
on this these
22:39
fractions so the initial 85% is the
22:42
train data then the 10% is the test data
22:45
and the remaining 5% is the validation
22:47
data you can even print out the training
22:50
data set length the validation data set
22:52
length and the testing data set length
22:54
so you'll see that the training data is
22:56
9 935 pairs the validation data is 55
23:00
Pairs and the testing data is 110
23:04
pairs even on the Whiteboard I have seen
23:06
that or I've written rather that the
23:08
next step is partitioning the data set
23:10
into training testing and validation
23:13
training is 85% testing is 10%
23:16
validation is 5% of course you can feel
23:19
free to play around with these
23:20
parameters when I share the code with
23:22
you there are many things in this code
23:24
which are not set in stone which means
23:26
they are not fixed and we can continue
23:28
changing so many things we can change
23:30
these these fractions we can change this
23:34
format we can use the
23:36
Microsoft 53 format which I showed you
23:39
and all of this is open for exploration
23:41
right now this field is so new that uh
23:44
right now is the time to really start
23:46
exploring get into research that way you
23:48
will also develop lot of confidence as a
23:50
machine learning and an llm
23:53
engineer so today we are going to end
23:55
this lecture until this part where we
23:57
saw the dat data set download and
23:59
formatting in The Next Step what we are
24:02
going to see is that we are going to
24:03
batch the data set now this is a um
24:07
topic which will need some amount of
24:09
detailing because it's not very
24:11
straightforward we have to make sure
24:12
that the input length is the same for uh
24:16
all of the instructions then we have to
24:19
convert the instructions into token IDs
24:21
we have to pad them with tokens uh there
24:23
are some specific things which we need
24:25
to do which we'll look at in The Next
24:27
Step which is organizing data into
24:29
training batches and I've also started
24:31
writing the code for the next
24:34
lecture uh in four to five lectures
24:36
we'll build our own personal assistant
24:38
chatbot so then you will have built your
24:40
own chat GPT completely from scratch um
24:43
and that will set you apart from all the
24:45
students who are just consumers of chat
24:47
GPT you'll now build your own
24:49
personalized assistant and then you will
24:51
have the confidence that whenever you
24:53
approach any company you can build a
24:55
custom chatbot for that company as well
24:57
by following the same procedure thank
25:00
you so much everyone I hope you are
25:01
enjoying these lectures where it's a mix
25:03
of whiteboard approach as well as coding
25:06
approach uh please try to follow along
25:09
write notes if you are coming to this
25:11
lecture for the first time I encourage
25:12
you to watch all the previous lectures
25:14
which have happened so far so that you
25:16
can strengthen your understanding
25:18
anyways I make the lecture so that it's
25:20
as selfcontained as possible thanks
25:23
everyone and I look forward to seeing
25:24
you in the next lecture


