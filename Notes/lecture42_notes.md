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

***

* 45:00

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













