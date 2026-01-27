

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


model uh and this is exactly what we are
15:21
going to do in the code right now before
15:24
that I just want to show you these two
15:26
types of
15:27
formatting uh so if so currently we have
15:31
this instruction input and output right
15:33
there are two ways to actually format
15:35
this data set and convert it into a
15:37
prompt the first is the alpaka prompt
15:39
style and the second is the 53 prompt
15:42
style so as I showed you the alpaka
15:44
prompt style is that uh you convert you
15:48
have the prompt which is as follows
15:49
below is an instruction that describes a
15:52
task Write a response that appropriately
15:55
completes the request then in the
15:57
instruction you
15:59
uh add this you add this
16:03
instruction uh then in the input you add
16:07
the input and in the response you add
16:08
the output so that's the alpaka prompt
16:11
so this instruction input and output
16:13
which was there uh that is converted
16:16
into this type of a prompt that below is
16:18
an instruction here is the instruction
16:20
here is the input occasion and the
16:22
response is the correct spelling which
16:23
is occasion with One S now 53 which was
16:27
developed by Microsoft it's another fine
16:29
tuning style where the prompt is user
16:32
and assistant in the user you directly
16:35
give the instruction which is identify
16:36
the correct spelling of the following
16:38
word occasion and then in the assistant
16:40
you directly give the output so here you
16:42
see the difference between F and the
16:44
alpaka is that in the F prompting the
16:47
user what the user has is instruction
16:50
plus the input
16:53
so so instruction
16:59
so instruction
17:01
plus the input is actually fused over
17:04
here whereas in the alpaka prompt the
17:07
instruction and the input is separated
17:10
we can use either of these in fact when
17:12
I share the code file with you I will
17:14
encourage you to try the 53 prompt style
17:16
as well uh but since the alpaka prompt
17:19
style is more common we are going to use
17:21
this and we are going to convert the
17:24
instruction input and output which we
17:25
have into prompts such as what is
17:29
mentioned in the alpaka prompt
17:31
style okay so let us convert our
17:33
instructions into alpaka format we are
17:36
going to define a function which is
17:37
called format input it's going to take
17:39
an entry uh so you can think of as one
17:42
entry as this thing which has key value
17:45
pairs the key has instruction input
17:47
output and corresponding values right so
17:51
when this
17:52
function uh returns an entry you first
17:55
construct the instruction text which is
17:57
below is an instruction that describes a
17:59
task Write a response that appropriately
18:01
completes the request and then in the
18:03
instruction you take the dictionary
18:07
which is entry and then you find the
18:09
value corresponding to the instruction
18:11
key so in this case the value will be
18:13
identify the correct spelling of the
18:15
following word so then the prompt will
18:17
be below is an
18:18
instruction uh and then this is identify
18:21
the correct uh identify the correct
18:24
spelling of the following word that's
18:26
the instruction text then you have to
18:28
specify the input text which is input
18:30
and then you specify that particular
18:32
input in this case the input is occasion
18:35
now see what we are doing here if the
18:37
input is not present then you just
18:39
return blank so in cases like these
18:41
where the input is not present the input
18:44
will be left blank and this is mentioned
18:46
in the alpaka repository also if the in
18:49
input is not present then we just have
18:51
below is an instruction instruction and
18:53
the
18:55
response right so this is my format in
18:57
input function which takes the entry
19:00
dictionary and then it gives me the
19:02
instruction text and it gives me the
19:04
input text and it combines them together
19:06
so when you run the format input it will
19:09
give you this output if the input is
19:12
present and it will give you this output
19:14
if the input is absent currently we have
19:17
not added the response but I'll show you
19:19
where we can add it okay so this is the
19:22
format input function now let us test it
19:25
uh on a data set so we'll take the data
19:27
index by 50 and we have already seen
19:29
what that is before identify the correct
19:31
spelling of the following word and we
19:33
will pass this input to the format input
19:36
so now the format input takes in this
19:38
data and gives the model input the model
19:41
input is basically until this point
19:43
below is an instruction and then input
19:45
is the occasion and then we have to add
19:47
the response to this right so then here
19:49
we say that the response will be the
19:51
output um so the dictionary
19:54
indexed uh dictionary and then we look
19:56
at the value corresponding to the output
19:59
key so then we have the instruction and
20:01
the input and then we append this
20:03
desired response to the model input and
20:06
so the desired response is the correct
20:07
spelling is occasion so when you print
20:10
the model input plus the desired
20:11
response you'll get the model input as
20:13
the prompt and then you'll get the
20:15
response itself now this full thing is
20:17
later fed as an input to the large
20:19
language model so that it trains on this
20:21
entire
20:23
prompt So currently we saw an example of
20:25
a data which has the input right what if
20:27
we have an example of a data which does
20:29
not have the input so data index by



***


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





