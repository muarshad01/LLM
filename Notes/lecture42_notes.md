## Evaluating the LLM (Whole Field!!!)

* How to measure LLM Performance?

* Extracting and Saving Responses

***

* 10:00

In practice, instruction-finetuned LLMs such as chatbots are evaluated via multiple approaches:
1. Short-answer and multiple choice benchmarks such as MMLU [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300), which test the general knowldge of a model.
2. Human preference comparison to other LLMs, such as LMSYS chatbot arena -  [LMSYS Org](https://lmsys.org/)
3. Automated conversational benchmarks, where another LLM like GPT-4 is used to evaluate the responses, such as [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/) completes the request.

***

* 15:00

***

* 20:00

#### Evaluating the fine-tuned LLM

* [Ollama](https://ollama.com/)

* [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)

```
$ ollama server (only on Windows)
$ ollama run llam3
```

***

* 30:00

***

* 35:00

***

* 40:00




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












