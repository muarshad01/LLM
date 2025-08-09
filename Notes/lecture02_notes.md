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

number four is what you all must be thinking about that llms are so good they can do these amazing tasks they can
14:19
almost behave like humans right but what makes llm so good what is the secret
LLM Secret Sauce
14:25
sauce typically there has to be a secret sauce right which makes llm so much
14:30
better than NLP what's that magic bullet here and usually people say that there
14:36
is no secret sauce to things things just gradually improve over a period of time but in this case there is a specific
14:43
secret sauce and that is the Transformer architecture so you see here I've added
14:50
the logo of secret sauce and for llms the secret sauce is Transformer
14:56
architecture don't worry if you know what it what what it means what is transformer for all you know you might
15:02
be thinking of the Transformers movie in which there are these cars which get converted into mechanical robots so you
15:09
that might be your first thought when you hear about Transformers right of course people who know about this know
15:14
exactly what I'm talking about here but for people who don't know uh this this is the only Transformer if this is the
15:21
only Transformer you know this playlist is perfect for you because you will
15:27
understand what what does Transformer mean and what it actually means is it summarized by this one schematic over
15:33
here this is what a Transformer architecture looks like and you might be
15:39
confused by all of these terminologies what is input embedding what's multi- attention head or what speed forward
15:45
here what does ADD and nor mean here what's output embedding there are so many things which look confusing and
15:51
it's fine but we are going to learn about this Secret Sauce in a lot of detail in this playlist there is one
15:59
paper which was introduced in 2017 which really changed the game that paper is
16:04
what I'm showing you on the screen right now it's called attention is all you need it was published by eight authors
16:12
from Google brain and uh this introduced the architecture of Transformers so the
16:18
schematic which I showed you on the Whiteboard is this schematic which they had in the paper can you guess how many
16:24
citations this paper has today it has more than 100,000 citations
16:30
in just a matter of 5 years so people say that research is boring and research does not have any external reward in
16:37
this case it does if you are one of the few authors on this paper you have 100,000 citations in 5 years and you
16:44
completely revolutionize the field of artificial intelligence so this is that paper and
16:49
it's a 15-page paper but if you try to read it it's very dense to understand
16:55
this paper really takes a lot of time and it really takes a lot of effort there are some YouTube videos to explain
17:01
this but I don't think any of them do justice to really explaining everything
17:06
from scratch every single page of this paper can be three to four videos so if
17:12
you think about it these 15 pages contain huge amount of information and
17:17
we'll be devoting lectures to different sections here such as positional encoding what is this dot product
17:23
attention what is this attention formula what is key query value what is multi-head ention we'll be figuring all
17:30
of that out so don't worry at this stage if you don't know what a Transformer means even if you have just this image
17:37
in Mind of a transformer it's awesome you have come to the perfect
17:42
place okay so as I said do not worry we will learn all about this Secret Sauce
17:49
which is the Transformers in the subsequent lectures that covers the 

fourth point of today's lecture and now


### LLM vs GenAI vs DL vs ML vs AI
1. AI 
2. ML 
3. DL
4. LLM 

watching right now so what is the differences between these so first the broadest umbrella is artificial intelligence and this contains all the other umbrellas so Any machine which is remotely behaving like humans or it has some sort of intelligence that comes under the bucket of AI. That's the biggest bucket so you might be thinking what's the difference between AI and ML right? so look at this example this is Lanza flight chat assistant and uh so it says hi I'm Alisa I am your Lanza chat assistant here is a selection of topics I can help you with you can choose one of the topics so then you say hi Alisa then Alisa says hello then you can click on any one of these you see it's a rule based chatbot you click some options and then Elisa is already programmed to answer so I say that flight cancelled check Alternatives and rebook so if I click on this Ela is already program to answer this is an example of AI because it covers intelligence right Lanza chat assistant can be thought of to be intelligent but this is Rule based intelligence it's not learning based on your responses so let's say you respond and your friend responds Elisa this chat assistant will behave the same way it's not learning based on what you are giving it or your specific nature as a user so that's why this is not an example of machine learning it's an example of AI but it's not an example of ML and that's why AI comes under the broadest or the biggest umbrella within that comes ml if you look at ml this is basically machines which learn they adapt based on how the user is interacting with them so you might be thinking okay that's fine then what's the difference between ML and DL why is DL a subset of ml the difference is that deep learning usually only involves neural networks but machine learning involves neural networks plus other fields and one such field is decision trees right let's say if you want to predict heart disease so if you have data from 303 patients and if you're collecting things like age gender chest pain cholesterol level uh ECG Etc you want to predict whether the person has heart disease or not and you build a decision tree like this now a decision tree like this does not have neural networks at all this is completely detached from neural networks so this is not an example of deep learning but it is an example of machine Lear learning a deep learning example would be something which involves a neural network so for example here we are uh this is an image detection and here's a convolutional neural network so we have a coffee cup here and we are detecting based on these filters we are so there are filtering layers in this convolutional neural network and at the end of this filtering layers we are detecting what this input is whether it's a Lifeboat whether it's a pizza so with maximum probability we are saying it's an Express so so a neural network is being used for this task that's an example of deep learning another example is let's say if you want to predict handwritten digit classification right so if you train a neural network like this if you start training so here you can see the neural network is being trained and the problem here is handwritten digit classification so I have given it a bunch of digits uh which actually look like this MN data it looks like this I've given this neural network bunch of digits to learn from and the task of this neural network is whenever I give it or whenever I write a new digit it should identify whether what digit is it now if you see this particular example this is actually an example of neural network so here you can see the neural network architecture we have the input layer we have the hidden Layer Two hidden layers and then we also have the output layer over here this is an example of deep learning so now if you click on test and if I select this let's say I select this and click on predict you see the neural network is correctly predicting eight so it's understanding the digits let's say I click on this and predict it's correctly predicting seven this is an example of deep learning that's why deep learning is a subset of machine learning machine learning involves these kind of neural network architectures plus it involves things which are not neural networks like decision trees now within deep learning is llms why because deep learning as we saw involves images also but large like this involved images even this project involved images but large language models do not involve images they only deal with text and that's why they are a further subset of deep learning this is the difference between AI ml DL and llm so then you might be thinking what is generative AI generative AI you you can think of as a mixture of large language models plus deep learning why because generative AI also deals with other modalities like image sound video Etc it deals with all of those modalities so it is definitely you can think of it as a mixture of large language models plus deep learning so if someone asks you what is generative AI basically we are using deep neural networks to create new content such as text images various forms of media whereas in llm you only deal with text so generative AI can be thought of as a mixture of llm and deep learning I know these terms are confusing but I hope these examples help you understand what's the similarities and differences between these terms as a summary you can think of as AI to be the broadest umbrella within that comes ml within that comes DL and within that comes llms and AI is artificial intelligence ml is machine learning is deep learning and llm is large language models and if you mix llm plus deep learning then we have generative AI because in the field of generative AI we don't just deal with text but we deal with other modalities of media like image like audio like video Etc so llms just represent a specific application of deep learning techniques which leverage their ability to process and generate humanik text which we have already seen they basically deal with text so this is the similarities and differences between AI ml DL llm and generative AI I hope you have understood this part and you like these visuals if you have any doubts until this part please write it in the YouTube comments and we will answer as quickly as possible and now we

#### Applications of LLMs
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











