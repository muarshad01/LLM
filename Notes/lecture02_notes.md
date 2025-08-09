## Building LLMs from Scratch

## Six major aspects 
1. What exactly is a LLM?
2. What does __Large__ mean in the LLM terminology?
3. What is difference between modern LLMs and earlier NLP models?
4. What's the secret-sauce behind LLMs? What's really makes them so good?
5. LLM, GenAI, DL, ML, AI
6. Applications of LLMS?

#### 1. What are Large Language Models (LLMs)
* [ChatGPT](https://chatgpt.com/)
* LLM is just a neural network (NN), which is designed to __Understand, Generate and Respond to human-like
text.__

* LLMs are essentially just neural networks (NN), which are designed for very generic type of text related applications such as understanding generating and responding to human like text. 

* ChatGPT the demonstration, which I just showed you is an LLM but what many people don't know about or they don't think about llms is that at the core of llm they are just neural networks, which are designed to do these
tasks. So, if anyone asks you what an llm is tell them that they are deep neural networks (DNN) trained on massive amount of data, which help to do specific tasks such as understanding, generating and responding to human like text, and in many cases they also respond like humans.

#### 2. What does __Large__ mean in the LLM terminology?
* By model size, I mean the number-of-parameters in the model.
* LLM typically have __Billions of Parameters__.
* That's the reason that's the first-part of the terminology __Large__.
* Why why are they called __language models__? That's pretty clear, if you remember the example, which I showed you over here, these models only deal with language they do not deal with other modalities like image or video. __Question answering, translation, sentiment analysis, and so many more tasks.__
 
#### 3. LLMs versus earlier NLP Models
* NLP models were designed for very specific tasks. For example, there is one particular NLP model, which might be designed for __language translation__. There might be one specific NLP model, which might be for __sentiment analysis__. 
* LLMs on the other hand can do a wide range of NLP tasks.

#### 4. What's the Secret-Sauce behind LLMs? What's really makes them so good?
* For LLMs, the Secret Sauce is __Transformer__ architecture.
* [Paper: Attention is all you need](https://arxiv.org/abs/1706.03762)

#### 5. LLM, GenAI, DL, ML, AI
1. AI 
2. ML 
3. DL
4. LLM 

* __AI__ is the broadest umbrella. Any machine, which is remotely behaving like humans or it has some sort of intelligence, comes under the bucket of AI.
* What's the difference between AI and ML?
  * Example, __rule-based ChatBot__, is an example of AI because it covers intelligence. Rule-based intelligence it's not learning-based on your responses.
* __ML__ involves neural networks (NN) plus it involves things, which are not neural networks, like __Decision Trees (DTs)__.
* __DL__ usually ONLY involves neural networks (NN).
  * Example, __predict handwritten-digit classification__. If you train a neural network (NN). I've given this neural network (NN) a bunch-of-digits to learn and the task of this neural network (NN) is whenever I give it a new digit it should identify what digit is it now?
*  __LLMs__ falls under DL llms why because deep learning and they only deal with text.
*  __GenAI__ you can think of as a mixture of LLM plus deep learning, why because GenAI also deals with other modalities like image, sound, video, etc.

#### 6. Applications of LLMS?
25:47
come to the last part of this lecture which is applications of llms and as I speak the applications go on increasing
25:54
but overall they can be divided into five main categories llm as number one they can be used to create
26:00
new content of course so if I write here
26:06
write uh poem about solar
26:13
system in the format of a detective story maybe this content does not exist
26:20
anywhere right now but you can create this content using uh llms so here you
26:27
can see there's a poem about solar system in a detective story format in the quiet sprawl of the Milky Way a
26:34
detective roamed the Stars by Night and Day his name was Orion in Cosmic Affairs
26:39
where Mysteries burst the case on his desk Dan quite bizarre yeah I won't read the full poem but here you can see you
26:46
have created new content you can even write books with llms generate media articles with llms and people are
26:53
already doing these things then you can use llms as chat
26:58
bots so you can actually interact with them as virtual assistant so here you see at the start the example which I
27:05
showed you it's like I'm chatting with this llm right it asks me what's your favorite form of relaxation I say
27:11
reading a book I can continue conversations with it and not just for me big companies Airlines uh then hotel
27:18
reservation desks they need chat Bots right let's say you call a customer care representative in 5 years it's highly
27:25
likely that all of those people are AI AI chat Bots and how are they built they
27:30
are built through the knowledge of llms in fact through this lecture or through this playlist rather we will be building
27:36
our own llm so you will be fully equipped to develop your own chatbot in any field which you uh which you like so
27:44
chatbots is one of the huge and major applications right now of large language models Banks uh movie restaurant movies
27:52
resturants all of them need chat Bots when you're booking shows when you are dealing with the bank you want to know
27:57
how to open an account all of these are going to be automated and are already being automated through llms that's one
28:04
of the biggest applications and through this playlist you will learn the skills to develop your own chatbot then the
28:10
third application is machine translation which means that we can of course translate the text to any language right
28:16
so then I I say basically that translate this poem to
28:26
French and uh here you can can see that chat GPT is working on this translation and immediately translates it to French
28:32
we don't even need Google translate now I'm sure this translation is accurate because it's it's amazing at this
28:39
translation tasks so you can very quickly translate the llms uh output into any language and it
28:47
does have some support for regional languages not a very broad support but it does support few Regional languages
28:54
in fact a lot of research is being done to improve the llm outputs for regional
29:00
languages for regional languages so that's Point number three point number four which we saw already is new text
29:07
generation such as writing poems writing books writing media articles news
29:12
articles it can basically create new text for you which did not exist before
29:18
in literature and finally llms can also be used for sentiment analysis you can
29:23
give a big bunch of paragraph and ask the llm to identify whether what's the
29:28
sentiment here this might be useful for hate speech detection let's say on social media like Twitter Instagram Etc
29:36
so all of these five applications are the pillars of llms and what they can do
29:41
there are several more applications which I'm probably missing right now but these five are the major ones to
29:47
illustrate these applications let me just take you through a portal which we have recently created for school teachers so here you can see we have
29:55
created this portal by learning about llm ourselves so here it's YouTube generator McQ generator text summarizer
30:02
text rewriter worksheet generator you we can do so many awesome things let me go to the lesson plan right now and say
30:10
that I want to create a lesson plan on the topic of gravity and I want to align it with the cbsc curriculum of India and
30:16
let me click click on generate uh so you'll see that the llm
30:21
is working on it a bit and it does take some time but now you can see the lesson plan is created for gravity where we
30:28
have the objective we have the assessment key points the opening the introduction to new material guided
30:33
practice independent practice Etc this would save so much time for the teacher
30:39
and it's amazing this was not possible 5 years back but now with the llms we can
30:44
do all of these applications we can do so many other cool things such as uh
30:50
text McQ generator let's say you are a teacher and you you want to generate the questions on let's say World War II
30:58
and uh you want one hard one medium and one easy question you click on generate and
31:05
you wait for the response of that llm you see immediately within 5 Seconds
31:12
we have got question and answers for World War II we have three questions we have their correct answer and the
31:17
explanation we build this llm froms or rather we build this llm application from scratch and once you go through
31:24
this playlist we'll have several lectures towards the end which will show you you how you can also build such type of
31:30
applications but here I want to illustrate the point that if you have knowledge about llms the sky is the
31:36
limit you can build wonderful applications like this and what many students do wrong is that they get so
31:43
fascinated by these applications that they just download the code they run the code they make some changes and they
31:48
write on their resume that they know about llm that's not the right way the right way is to understand the
31:55
foundations make your Basics clear understand the nuts and bolts of llms which is the purpose of this playlist so
32:03
this is uh this is section six which is the application of llms and now we have
32:08
completed the six sections which we have planned for today's lecture at the end I want to leave you
32:15
with just one sentence and that is the sky is the limit when it comes to llm
32:20
Applications please remember that the applications which I just showed you right now are just the tip of the
32:25
iceberg you can do so many things right now if you learn about llms that's
32:31
what's so exciting about the time which we live in currently uh the sky is the
32:36
limit for research applications for industrial applications but what we think is that students who will really
32:42
contribute and make an impact are the ones who know the details of how to write the Transformer code how key query
32:48
value Works what exactly is positional encoding until you understand these
32:54
Concepts it becomes difficult your knowledge will be superficial and that's the whole purpose of this
33:00
playlist please comment in the YouTube uh comment section if you liked the
33:05
style of this uh content description if you like the this whiteboard approach I'm basically trying to write everything
33:13
here and uh give a visual flavor and visual intuition for how things are working in subsequent sections we'll
33:19
definitely dive a bit deeper into coding the next lesson as I mentioned here is stages of building llms we'll have
33:26
couple of lectures about these Basics and then we'll be diving into coding so make sure you're understanding
33:33
make sure you make notes and ask questions in the comments




