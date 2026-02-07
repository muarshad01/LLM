## Evaluating the LLM (Whole Field!!!)

#### How to measure LLM Performance?

#### Extracting and Saving Responses

***

* 10:00

In practice, instruction-finetuned LLMs such as chatbots are evaluated via multiple approaches:
1. Short-answer and multiple choice benchmarks such as MMLU [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300), which test the general knowldge of a model.
2. Human preference comparison to other LLMs, such as LMSYS chatbot arena -  [LMSYS Org](https://lmsys.org/)
3. Automated conversational benchmarks, where another LLM like GPT-4 is used to evaluate the responses, such as [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/) completes the request.

***


given by multiple llms and then they compare the performance between the large language models so this is
15:54
basically having a human in the loop and the human using their own intuition and
16:00
understanding they Benchmark or compare llms that's the second way the Third Way which is also fairly common is using a
Evaluation Method 3 - LLM measures another LLM
16:07
large language model itself to evaluate how close the llm responses to the
16:13
actual output so let's say here I have collected a file which has the instruction input output but it also has
16:19
the model response so in the third category what is done is basically you look at the true data which is the true
16:25
output which we expected and we look at the model response and then we ask a large language model
16:31
itself to compare between the output and the model response and to assign a score so this is very uh this is fairly
16:38
straightforward to do because it's fully automated we just look at the output we look at the model response and then we
16:44
ask a trained a very massive large language model to uh compare these two
16:49
and find the score if you think about it this is also like taking the easy way out because we don't know how the llm is
16:56
evaluating right what are the metrics through which the llm itself is comparing between the output the true
17:02
output and the model's response we are trusting the llm is extremely and supremely smart to do this to get this
17:10
score for us since this method is simple we are also going to implement this method in this particular video or in
17:17
this particular lecture Series where what we are going to do is that we have the output and the model response we are
17:24
going to ask an llm to look at the output to look at the model response to compare and to ass a
17:29
score but it's very important for you all to be aware of these three types of llm evaluation which is extremely
17:37
important uh so considering the scale of the task at hand we will Implement an approach similar to method three which
17:43
involves evaluating the responses automatically using another llm this will allow us to efficiently assess the
17:50
quality of the generated responses without the need for extensive human involvement thereby saving time and
17:57
resources while still obtaining meaningful performance indicators if you
18:02
have this code and if you can run this code using the mlu test uh that would be
18:08
awesome and I would really like to see if someone works on that part of the
18:14
code and take the instruction find T llm and run the MML test on it awesome now
Collecting LLM responses in JSON format
18:20
the next step is that we need to basically collect the responses for the entire test file right earlier we only
18:28
saw the responses for three for the first three examples of the test data now what we need to do is that we need
18:34
to collect model responses for all of the uh all of the instructions in the
18:40
test data set and we need to collect these responses in a separate file so what we are doing now is that we'll
18:47
prepare the responses for the evaluation process and we will essentially construct a new file or create a new
18:54
file which is titled instruction data with response. Json for record keeping
18:59
this file will essentially contain the instruction input the true output and also the model
19:05
response um so to give you a visual this is how that file will look like now if you look at this first file which is
19:11
just the instruction data. Json file over here instruction data. Json this only con consists of the instruction the
19:18
input and the true output it does not contain the model response but now we are developing one more file or creating
19:25
one more file which is called instruction data with response so this Con consists of the instruction the
19:31
input output and it also consists of the model response this will make it very easy for us to later evaluate the
19:38
performance of the model because then we simply have to compare the output and the model response for every instruction
19:44
and assign a score so in this piece of code what is done here is that uh in this piece of
19:51
code if you see we are looking at all of the in all of the input instructions in the test data and we are generating the
19:58
response for all of the inputs in the test data and then we are collecting the responses in a file called instruction
20:04
data with response that's all so when you run this code the generate function will be called on all of the instruction
20:10
input pairs in the test data the responses will be generated and only the
20:16
response will be collected and then it will be appended to the instruction data with response file so overall the
20:22
instruction data with response file will look exactly like what I'm showing on the screen right now where in the
20:28
instruction input and output dictionary the model response will also be appended now for every single uh instruction
20:36
input output pair awesome now if you run this process it will take some time for me it took
20:42
around 10 to 15 minutes to create this instruction data with response file but it will be created and then it will be
20:48
stored for you so remember now this test data is the dictionary which consists of
20:56
the instruction input output and the model response also we can test this by printing the first element of the test
21:02
data dictionary so if you print out the first element you'll see that we have the instruction we have the input we
21:08
have the output and then we have the model response as well the the output is here and then the model response is here
21:15
so the test data dictionary essentially now uh consists of everything which we
21:21
need it consists of the instruction the input output and the model response now what we can do is that once
21:28
we have uh we have this file so everything is ready for us to evaluate the output and the model response now
Saving the LLM parameters
21:35
what we'll do is that we'll just save our fine tune model this is extremely important because if you accidentally
21:41
close your working session then you don't want to fine tune again right remember fine tuning took four hours for
21:48
me on my PC and I just want to I just want to reuse the fine tune weights again when I start my session the next
21:54
time so please don't forget about this step it's very simple you just have to use the tor. save command and model.
22:01
state dictionary which will ensure that all the train parameters will be stored in this file name which is gpt2 medium
22:09
355 million sf. pth this is the file where we are storing the model and to
22:15
load the model in a future session you simply have to do load State dict so two commands are important tor. save model.
22:22
State dict and tor. save or load State dict rather the first command is model
22:28
do State dict so saving this and the second is load State dict and then you have to load the file in which you have
22:36
stored the parameters awesome I hope everyone of you is with me until this point so we
22:43
have reached this stage where we have successfully collected the responses in a file called
22:49
instruction instruction data with response. Json we have collected the responses in this file and we have also
22:56
discussed about a way in which we are going to compare the output and the model response we have not discussed too
23:02
many details about this but we have seen the three Frameworks for evaluation and we have shortlisted this last framework
23:08
where we'll use use an llm to compare between the output and the model
23:13
response now we are ready to move to the next part which is essentially evaluating the fine tuned large language
23:20
model so the evaluation process essentially will will come in this building block
23:28
what we we have seen Okay so until now we saw extracting the responses and we have also seen qualitative evaluation
23:34
where we looked at the response so let's say for example I can qualitatively look at the output and I can look at the
23:41
response and I can say whether it's correct or not qualitatively right but we have not yet mathematically or
23:47
Quantified scoring the responses this is the part which we'll Implement in evaluating the llm so after extracting
23:55
the responses by our fine tuned llm we will use another large language model to automatically evaluate these responses
24:03
and let's see how we are going to do that in practice which is the large language model which we are going to
Evaluating the fine-tuning LLM introduction
24:08
use so this brings us to the step number seven in the evaluating the fine tuned llm and as I mentioned in this section
24:15
we will Implement a method to automate the response evaluation of the fine tuned llm using a another larger llm so
24:22
we'll use a bigger larger llm which is pre-trained and it's extremely supremely knowledgeable to compare our model
24:29
response and the true response and to assign a score to implement this evaluation step
Ollama introduction and setup
24:35
we are going to use a software or it can be called as an application which is
24:40
called as AMA so let me take you to the AMA
24:46
application so here if you go to ama.com you can just type it ama.com you'll see that this interface
24:54
essentially comes up and the simplest way to think about AMA is that it's an efficient application to run large
25:00
language models on your laptop so you can learn you can run various large language models you can run Lama 3 which
25:07
is developed by meta you can run 53 developed by Microsoft you can run
25:13
Mistral you can leun gamma 2 similarly you can leun several models on your PC
25:18
and remember using o Lama you do not do pre-training you just do inference which means that the model is already
25:24
pre-trained you just look at the responses or uh you look at the output
25:30
which is given by the model you use the model in inference mode using AMA you do not use it in
25:36
pre-training mode so what we are going to do is that we are going to implement the evaluation step uh by utilizing an
25:44
instruction ftuned 88 billion parameter Lama 3 Model so we are going to be using
25:50
an 8 billion parameter Lama 3 Model uh and that's and we are going to access
25:56
that through olama so the reason we are going to utilize
26:02
this instruction fine tuned model is that because it's already fine tuned on a huge number of instructions so if you
26:08
search about this this is the Lama 38 billion instruction fine tuning model
26:14
and here the par 8 billion parameters are already optimized which means that this model is already trained to follow
26:21
instructions and it's supremely smart so what we are going to do is that we are going to utilize this model to compare
26:28
between the True Result and the our llm model output so using this Lama 38
26:34
billion we are going to compare the actual output and our model response and we are going to assign a score so we are
26:41
going to tell this instruction finetune Lama model that your next instruction is
26:47
that look at the output look at the model response and assign a score to how well my model is doing and since this
26:53
llama model is already trained for instruction for following instruction it will do a great job at this new
27:00
instruction which is essentially finding an evaluation score this is also a great time for all
27:06
of you to learn about AMA which is a very commonly used uh llm inference
27:11
application okay so uh one thing to remember is that ama is only a tool for
27:17
generating text using llm inference and it does not support training or fine-tuning llm so let me now show you
27:24
uh the download process for AMA so that you can follow the similar instruction functions on your laptop all right so
27:30
the next step for us is to download olama um so I'm using a Mac here so I'm
27:36
going to show you how to install it and run it on Mac and I'll also give you instructions if you're using Windows so
27:43
you have to go to ama.com so let me type ama.com here and you have to just click
27:48
on download over here once you click on download uh the entire so if you are on
27:54
Mac OS or Linux or Windows you can download the Appo rate version for you so I have clicked
28:01
on download for mac o and you can see that the download process starts here it's a file which is
28:07
177 uh 177 MB so you can download it and then you can open it follow instructions
28:13
click on next next and next and then AMA will be installed it's pretty simple the installation process does not take too
28:20
much time I'm going to cancel this download over here because I've already installed it then what you can do is
28:26
that then you have to open your terminal so so here you can see Ive opened my Mac terminal over here and then what you
28:32
have to essentially type is that you have to type O Lama run Lama 3 so this is the correct command o Lama run Lama 3
28:40
and let me type it over here also uh o Lama run Lama 3 this is the command
28:48
which you have to type on the terminal and if you are using a Mac you can directly type this command if you are
28:53
using Windows then the command which you might need to type might be o Lama
29:00
serve so if you're using Windows type this command first o Lama serve and then type O Lama run Lama 3 okay um so type
29:11
these commands in the sequence if you're using Mac you can directly type or o Lama run Lama 3 let me show you my
29:17
terminal again so here you can see I have typed o Lama run Lama let me expand
29:23
this let me expand my terminal so that you can see it in more detail so here
29:28
you can see that Ive run o Lama run Lama 3 and the when you try to run Lama 3 the
29:35
files which are downloaded are of size 4.7 GB so it's an 8 billion parameter
29:40
model right so it takes a lot of space and memory to download this so it took around 15 to 20 minutes for me to
29:47
download this on my desktop uh but as the downloading is happening you'll see
29:53
all of these instructions being printed out on your terminal and at the end you will see success U when you see the
29:59
success these arrows will appear these right hand side arrows which means that now the Lama 3 is loaded which means you
30:05
can interact with Lama 3 large language model you can ask any question to this so for example here I have asked what do
30:11
llamas eat and now the llm which will respond to you is not chat GPT or any
30:17
other llm the llm which will respond is this Lama 38 billion instruct this is that llm which will
30:23
respond to you um so that's what AMA helps you to do so currently have run
30:28
Lama 3 over here right similarly you can use o Lama to run uh any other llm also
30:34
you can use o Lama to run 53 you can use it to run mistal GMA 2 Etc on your
30:40
laptop so I'm running this on my laptop and all these results which I'm seeing are on my laptop as well awesome so once
30:47
you reach this stage it will mean that o am I successfully running for you so when when I'm going to implement the
30:54
next part of this code please keep o Lama running if you shut down this terminal AMA will not be running and
31:00
then your code won't execute so please keep AMA running the simplest way to test is that just make sure to ask some
31:08
question over here and if you get a response it means that ama has been running successfully awesome now I'm going to
31:15
move back or switch back to code to explain the rest to you okay I hope you
31:20
have installed AMA now and you have run the O Lama run Lama 3 command on the
31:25
terminal which I have demonstrated I have just provided a simple code block here which verifies that the AMA session
31:32
is running properly before we use AMA to evaluate the test set responses so our
31:37
final goal is to use o Lama and Lama 3 especially to compare the output and the
31:43
model response to assign a score but before doing that we need to check whether o Lama is running or not so you
31:49
can run this code and here it should come O Lama running is equal to true if this comes false which means that ama is
31:56
not successfully running uh now I showed you the O Lama run command on the terminal right there is
Query Llama 3 and generate response
32:03
an alternative for this command instead of going to the terminal each time we can interact with the Lama model uh
32:09
using the API through python so we can create a function which is called as the query model what this function does is
32:15
that uh you can pass in the model so here I'm using Lama 3 so this function when it's called it will pass a query to
32:22
Lama 3 and the query will also be provided by us so we'll be providing the query
32:29
uh we'll be providing the query as the prompt so when you call this function query model you'll have to provide a
32:34
prompt and you'll have to provide a model and what this function will do is that this this function will send a
32:40
request uh will send an API request to this Lama 3 Model based on the prompt and then the response will be generated
32:47
and then the response will be returned so instead of writing or instead of running
32:53
the uh o Lama run Lama 3 on the the terminal each time we can simply do an
32:59
API call where we can fetch the results and bring it into our jupyter notebook interface I'm not going through this
33:06
code in detail because this will distract us from the main purpose of this whole video lecture which is to uh
33:13
perform the model evaluation so simply note that what this query model does is
33:19
that it helps us to pass in a prompt to Lama 3 or whichever llm which we are
33:24
using through o Lama and to give us the responses that's it so now what we'll do is that the prompt
33:30
which we are going to give here is that these are this is the output this is the model response compare these and assign
33:37
a score that's it that's all we are going to do but this function query model will be very important to us
33:42
because we are going to pass in the prompt through this function okay now again I want to
33:48
mention that before running the subsequent code cells ensure that all Lama is still running the previous code
33:54
which I showed you over here should print o Lama running to be equal to True uh to confirm that the model is
34:01
active and ready to receive requests so now here is just a small demo of how the query model works so in the query model
34:08
here you can prescribe the uh input which is what do llamas eat and then you
34:13
can also mention the model so here I'm using Lama 3 in O Lama you can use various models so you can search models
34:20
here and view all so you can see the different types of models which are present um in in ama so you can do gamma
34:30
2 also here so I can do I can do gamma 2 also over here but
34:36
the reason we are sticking with Lama 3 is that we are using this 8 billion instruction fine tuned model since it's
34:42
already fine tuned on the instruction or use instruction data set okay so uh to the query model
34:49
function you pass in any prompt and the model and that llm will look at the prompt and generate a response so you
34:55
can print out the response over here now please note here that this process also takes a lot of compute memory so to
35:03
print out one result here it took a long time for me because I on CPU I highly
35:08
encourage if you have a GPU access for all of you to uh use the GPU wherever
35:13
possible if you have a CPU it's fine I'm showing you only those things which can run on a CPU but just make sure that
35:19
this will take time and at least free up memory space free up at least 15 to 20 GB on your desktop for this entire code
35:27
to run so this query and printing out this result is not as straightforward as I've shown it here it took a long time
35:34
to generate this text and to to query so let's say I've just passed in one query
35:40
here right if I passed in three or four queries my laptop is does not run my laptop hangs so that's why I can pass a
35:47
maximum of two queries in one session simultaneously otherwise my laptop just hangs because the processing speed or
35:53
the processing power is not there so now what we can do is that using using the query model function we
Using Llama 3 to evaluate our fine-tuned LLM
35:59
can evaluate the responses which are generated by our fine tune model with a prompt and in that prompt we have to ask
36:06
the Lama 3 Model to rate our fine tuned models responses on a scale of 1 12 100
36:11
based on the given test response so let me show you the prompt which you are going to give so the prompt which now we
36:17
are going to give to the query model is that uh given the input and the input is
36:22
going to be this given this input uh
36:28
actually given the input so it's the format input function so the input is going to be actually the instruction
36:35
along with the input this is going to be the input over here so given this input and the correct output so the correct
36:42
output is this output
36:47
um so given the instruction and the correct output score the model response
36:53
and the model response is the model response in this Json file which is this model response score the model response
37:01
on a scale of 0 to 100 where 100 is the best score so what the llm will do now
37:06
is that it will look at the output it will look at the model response based on this instruction and the input and then it will assign a score so we are using
37:14
the query model function which we defined earlier and then we are passing this prompt you see how we are using
37:19
another large language model to evaluate our large language model um so we are using the Lama 3B large Lang language
37:27
model Lama 38 billion large language model to evaluate our fine tuned llm and
37:33
here we have specified the model so we don't need to specify it again um when we call the query model because if it's
37:40
not specified by default the query model will use Lama 3 when I share this code file with you you can feel free to
37:46
explore with other llms also the sky is the limit here it's it's an exploratory
37:51
notebook and I've got good decent results here but you can of course feel free to explore with larger models
37:58
different models as well so here you can see that I'm testing for three data
38:03
points or three testing data here and I'm printing the model I'm printing the
38:08
model response and I'm printing the score also which is given by the this Lama 8 billion
38:14
model again when I ran this code on my laptop my laptop crashed initially and
38:19
then it took around 35 minutes to print out the responses for the
38:25
three um instruction input output pair so three samples in the test data so for
38:30
me it was impossible to run this code on the entire test data set because I could only do it for two to three samples so
38:37
here is the response so for the first query for the first instruction the
38:42
rewrite the sentence using a Sim the our model response was the car is as fast as
38:49
a bullet and the data set which is the actual response is the car is as fast as lighting so here is the qualitative
38:55
evaluation which is given by the llm and rate the model response the car is as fast as a bullet 85 out of 100 here's
39:02
why the response uses simil correctly comparing the speed of the car to something else in this case a bullet the
39:08
comparison is relevant and makes sense as bullets are known for their High Velocity the phrase as fast as is used
39:15
correctly to introduce the simile the only reason I wouldn't give it a perfect score is that some people
39:20
might find the comparison slightly less Vivid or evocative uh for example comparing
39:26
something to lightning as in the actual response can be more dramatic dramatic however as fast as a
39:32
bullet is strong and effective simile that effectively conveys so it says that although the um response can be more
39:41
dramatic it still captures the essence and it still captures the grammatical meaning of a simile so overall the model
39:48
did a great job and the score is 85 out of 100 that's for the first for the second uh for the second input the
39:55
actual response was the type of Cloud typically associated with thunderstorms is
40:01
cumulonimbus cumulonimbus but the model response is a thunderstorm is a type of
40:06
cloud Etc it's pretty long so the score for this model as given by the Llama llm
40:12
is 20 out of 100 and the reason is because the response doesn't directly answer the question about what type of
40:17
cloud is typically associated with thunderstorms instead it provides a general description of thunderstorms
40:23
which is not relevant to the original questions at all the response of also contain some inaccuracies such as
40:29
stating that thunderstorms for more high pressure regions whereas the llm pointed out that thunderstorms can actually
40:35
occur in various weather patterns that's why the score is 20 out of 100 in the third question the actual
40:42
response was Jane Austin but the response of the model was the author of Pride and Prejudice is George Bernard
40:48
Shaw this is wrong so the llm rates is zero out of 100 the correct answer is
40:53
Jane Austin not George Bernard Shaw George Bernard Shaw was an Irish playright and author but he did not
40:59
write Pride and Prejudice the response is completely Incorrect and does not provide any relevant information this is
41:05
awesome right it looks that the llm which we are using is pretty smart and it's doing an awesome job at qualitative
41:11
or even quantitative evaluation so as human I can think of some qualitative uh similarities or
41:18
qualitatively I can think how to evaluate the evaluate the model's response but even the qualitative
41:24
evaluation of the llm is actually amazing the kind of points which the llm introduced do not come naturally to me
41:31
for example as a human I did not think about this dramatic effect um as a human I did not know
41:39
that uh thunderstorms can also occur in various weather patterns and uh stating
41:45
that thunderstorms form over high pressure regions might be inaccurate so as humans I have some limitations even
41:51
in qualitative qualitative evaluation and the llm is filling those limitations it's doing an awesome job at qualitative
41:58
evaluation and although the quantitative evaluation I'm not still sure why it's 85 why it's 20 it kind of makes sense
42:05
here it's 20 because most of the answer was wrong here it's zero because it's a
42:10
factual question so the answer is factually incorrect and here it's 85 because most of the answer seems to be
42:15
generally correct so I would trust actually the responses which have been given by this large language model and I
42:22
would say that this evaluation is being done nicely we even have a quantitative metric now
42:31
great so based on the generated response we can observe that the Lama 3 Model provides reasonable evaluations and is
42:37
capable of also assigning partial points or even zero points when the answer is incorrect the previous prompts however
42:44
returned highly detailed evaluations we don't want these qualitative evaluations we just want a score so then we can
42:50
change the prompt by saying that respond with integer number only and when you do that the the result you can see is only
42:57
with integer numbers and this is now the metric which we are evaluating instead of reporting the model accuracy the
43:04
classification accuracy we are using another llm to evaluate our llm and we
43:09
are going to assign a quantitative score it's still not as robust as a
43:15
classification accuracy because many people might argue that why not assign a score of 88 or 90 and this is still
43:22
subjective but it generally makes sense and I would like to go ahead with with this evaluation although there might be
43:29
other evaluations which also make sense and that's why this field of llm evaluation is a field of such
43:36
a um such an open research because there is a huge scope for what are the metrics
43:42
which we should use for evaluations uh does it make sense to use mlu does it make sense to use another
43:48
llm to evaluate one llm it just leaves room open for a lot of subjectivity when
43:54
you are doing a brain tumor classification problem the accuracy just has one value based on how many correct
43:59
answers are there and based on how many wrong answers are there but in this case there is no one correct value right um
44:06
there can be many correct values for this there is a lot of subjectivity in the evaluation over here and this
44:12
subjectivity itself leaves room for a lot of research and lot of
44:18
Explorations U if you have compute power on your system or if you're using GPU
44:23
here is a function which I have written which actually goes through the entire test data set and gives and calculates
44:29
this kind of a score for the for all of the um instructions in the test data set
44:35
and then finds an average score uh so I'm not actually running the
44:40
above function because of Hardware Hardware limitations I'm using MacBook Air 2020 if you have a M3 MacBook Air it
44:48
takes about 1 minute on that if you have a GPU it might be of a similar speed it will be very
44:54
fast so when you run the above code you will see see that our model the fine tune model achieves an average score
45:01
above 50 which provides a useful Benchmark for comparison against other models or experimenting with different
45:07
training configurations it's worth noting that ama is not entirely deterministic which
45:13
means that the scores which you obtain versus the scores which I have shown over here they might be slightly
45:18
different but in any case we now have a framework for a quantitative evaluation
Steps to improve the fine-tuned model
45:24
as has been mentioned over here in a lot of detail there are number more of things which we can do to improve the fine tuning performance right so if you
45:31
run the fine tuning on the entire test data set you will see that the average score is just above 50 and which is not
45:37
that great so if you look at so here is the instruction input output and model response uh where the model has been run
45:45
for all of the test data sets using M3 MacBook Air and now if you see some answers are correct but some answers are
45:52
wrong so for example let's see some sentences
45:58
rewrite the sentence the lecture was delivered in a clear manner the actual output was the lecture was delivered
46:04
clearly but our model response is the lecture was delivered in a clear manner so the model response is the same as the
46:10
input so in this case the model is making a mistake right uh in many such
46:16
cases you'll find that the model makes a mistake but in many cases the model is also good so that's why the accuracy
46:21
which you have obtained the evaluation accuracy is around 55% so here what I have written is there
46:27
can be several ways to improve the model's performance first of all you can adjust the hyper parameters such as
46:32
learning rate batch size or number of epo secondly you can increase the size of the training data set we only used
46:39
1100 instruction input output pairs that's why maybe and we are using 85% as
46:45
training data so in a sense we are only training on 800 instruction input output pairs those are not
46:50
enough so you can increase the size of the training data set one thing which I can recommend you to do is this
46:57
alpaka um so let me type this over here Stanford alpaka so if you go to this
47:04
repository over here they have a repository which contains data set of 52,000 instruction output pairs so
47:11
that's 50 times higher than the length of the data set which we used if you use such a data set you might get better
47:17
responses and I would be very I would be encouraging all of you to try the same
47:23
code but on a much larger data set and I have even showed you this data set over here the alpaka fine tuning data set
47:31
I'll share the repository Link in the video description also okay then what you can do is you
47:38
can experiment with different forms different prompts or instruction formats to guide the model responses more
47:45
effectively um and then finally you can consider the use of a larger pre-train model pre-train model which may have
47:51
greater capacity to comp to capture complex patterns and generate more accurate responses in our case we used a
47:57
gpt2 355 million right so let me show you the model which we have used in our
48:03
case yeah we have used a gpt2 medium which is 355 million parameters again if
48:09
you have the compute power or if you have access to GPU you can use gpt2 large which has 774 million parameters
48:16
and you can use gpt2 Excel which has more than a billion parameters but again you should do this only if you have
48:23
compute power for example if you use gpt2 Excel the number of Transformers is
48:29
48 and the number of attention heads in each Transformer is 25 uh so that's one configuration which
48:35
you can definitely vary you can vary the you can vary the length or the size
48:42
of the pre-train model a larger pre-train model may have greater capacity to capture complex patterns and
48:48
generate more accurate responses lastly we can also use parameter efficient fine tuning so the
48:54
fine tuning technique which I showed you is pretty simp simple it just simply uh
49:00
fine tuning or updating all of the parameters in the architecture based on the specific data set but there are
49:06
parameter efficient fine tuning techniques which have been developed such as Laura uh and uh so let me just show you
49:13
that Laura parameter efficient fine tuning and you can see that this is the
49:19
kind of fine tuning where uh the fine tuning process is made much more efficient than normal fine tuning I did
49:28
not have time to cover this fine tuning process in this lecture or the previous lectures but you I'll share
49:35
the uh information related to this in subsequent lectures you can also search a bit about it and if you have time you
49:41
can go ahead and Implement parameter efficient fine tuning it's not very difficult but it just leads to
49:46
computational speed up a lot and sometimes it even leads to speed up in accuracy uh okay so with that we have
Recap and summary
49:53
come to the end of this lecture where we have successfully in fact uh we have started from scratch and we have
50:00
successfully fine-tuned the model on the instruction data set and now we can say that a model
50:06
can answer instruction successfully but more importantly we have also seen the evaluation metrics predominantly we saw
50:13
that there are three evaluation metrics for uh instruction data instruction
50:18
ftuned llms the first is mlu which is measuring massive multitask language
50:24
understanding the second is uh human preference comparison such as such
50:30
as asking humans to rate llm output and the third one which we have used in this
50:35
lecture is using another larger large language model to evaluate our llm
50:40
performance so we have used the meta Lama 8 billion instruct model to
50:46
evaluate the performance of our model and this was an automatic or this was an
50:52
automated way and this proceeded in a simpler manner that's why we use this third method
50:57
but if you have time I would highly encourage you to use our own code but use benchmarks such as measuring massive
51:03
multitask language understanding you will learn so much through this process this mlu paper is highly impactful and
51:10
highly relevant and if you want to contribute to llm evaluation research there is a lot of scope for this
51:16
contribution because this field is novel it's evolving and I think that there is there
51:22
are still a lot of breakthroughs remaining to happen in the llm evaluation space so thanks a lot everyone we come to the
51:29
end of this lecture where we covered the entire instruction F tuning
51:35
pipeline um let me just show you that pipeline once and what all we have covered yeah so we have essentially
51:41
covered these nine steps which have been showed over here uh we started out with stage one which is preparing the data
51:48
set which was very important then we fine tuned the large language model and we did not stop there we even evaluated
51:55
the large language model model we extracted responses did a qualitative evaluation and then even did a
52:01
quantitative scoring using AMA through this we also learned a new
52:06
tool called AMA and uh with this we have now come to the end of the instruction
52:12
finetuning llm lectures these were five to six very comprehensive and highly detailed lectures which taught you
52:18
everything about the fine tuning process thanks a lot everyone I hope you have liked these lectures which were a mix of
52:26
whiteboard approach plus coding I don't think there are any other videos out there which show about fine tuning
52:31
pre-training in so much detail the reason I'm showing you all these details is that I firmly believe that a student
52:38
who knows the nuts and bolts of how models work of how exactly things are
52:43
assembled those will be the students who will be strong ml Engineers strong llm engineers and those will be the students
52:49
who contribute to Noel research or breakthroughs thank you so much everyone I look forward to seeing you in the next
52:55
lecture






