
* GPT-3 architecture in a lot of detail we also saw the progression from GPT to GPT-22 to GPT-33 and finally to GPT.4 
* Total pre-training cost for GPT-3 is around 4.6 million, which is insanely high.
* We have also looked at the dataset, which was used for pre-training GPT-3. 
* We learned about the differences between zero-shot versus few-shot learning.
* Attention behind self-attention prediction of next-word.
*  uh zero-short versus few-short learning.
*   basics of the Transformer architecture datasets used for llm pre pre-training
*   difference between pre-training and fine tuning.
*    stages of building a large language model.
* three stage stages stage-one stage-two and stage-three
*  so in stage one we are going to be looking at uh essentially building a large language model and we are going to look at the building blocks which are necessary so before we go to train the large language model.
*  we need to do the data pre-processing and sampling in a very specific manner
*   we need to understand the attention-mechanism and we will need to understand the llm architecture so in the stage-one, we are going to focus on these three things understanding how the data is collected from different datasets
*    how the data is processed
*     how the data is sampled
* number one then we will go to attention mechanism
* how to C out the attention mechanism completely from scratch in Python what is meant by key-query-value
* what is the attention-score
* what is positional encoding
* what is Vector embedding
* all of this will be covered in this stage we'll also be looking at the llm architecture such as
* how to stack different layers on top of each other where should the attention-head go
* so what exactly we will cover in data preparation and sampling
*  first we'll see tokenization if you are given sentences how to break them down into individual tokens
*   as we have seen earlier a token can be thought of as a unit of a sentence but there is a particular way of doing tokenization
*   we'll cover that then we will cover
*   Vector embedding essentially after we do tokenization every word needs to be transformed into a very high dimensional Vector space so that the semantic

***

* meaning between words is captured.
*  Encode every word so that the semantic meaning between the words are captured
*  so Words which mean similar things lie closer together
*  so we will learn about Vector embeddings
*  positional encoding the order in which the word appears in a sentence is also very important
*  pre-training model after learning about tokenization
*  Vector embedding we will learn about how to construct batches
*  meaning of context how many words should be taken for training to predict the next output we'll see about that and how to basically Fe the data in different sets of batches
*  so that the computation becomes much more efficient
*  so we'll be implementing a data batching sequence
*  before giving all of the data set into the large language model for pre-training
#### Second Point
* I mentioned here is the attention mechanism
* so here is the attention mechanism for the Transformer model we'll first understand what is meant by every single thing here what is meant by multi-ad attention
* what is meant by Mas multi head attention
* what is meant by positional encoding
* input embedding output embedding
* all of these things and then we will build our own llm architecture so uh these are the two things attention mechanism
*  outcome of stage two is to build a foundational model on unlabeled data
* we'll break it down into epox and we will compute
*  the gradient uh of the loss in each Epoch and we'll update the parameters towards the end
*   we'll generate sample text for visual inspection this is what will happen exactly in the training procedure of the large language model and then
*   we'll also do model evaluation and loading pre-train weaps
*    evaluation training
*    and validation losses
*    then we'll write the llm training function
*     Implement function to save-and-load the LLM weights
*   load pre-trained weights from open AI into our large language model.
*    stage-two which is essentially training Loop plus uh training Loop plus model evaluation plus loading pre-trained weights to build our foundational model.
*     so the main goal of stage two as I as I told you is pre-training and llm on unlabelled data great
* but we will not stop here after this we move to
*

#### stage-three 
* Fine-tuning the LLM so if we want to build specific applications we will do fine tuning in this playlist
* 

***

spam whereas hey just wanted to check if we are still on for dinner tonight let me know this will be not spam so we will build a large language model this application which classifies between spam and no spam and we cannot just use the pre-trained or foundational model for this because we need to train with labeled data to the pre-train model we need to give some more data and tell it that hey this is usually spam and this is not spam can you use the foundational model plus this additional specific label data asset which I have given to build a fine-tuned llm application for email classification so this is what we'll be building as the first application the second application which we'll be building is a type of a chat bot which Bas basically answers queries so there is an instruction there is an input and there is an output and we'll be building this chatbot after fine tuning the large language model so if you want to be a very serious llm engineer all the stages are equally important many students what they are doing right now is that they just look at stage number three and they either use Lang chain let's say they use Lang chain they use tools like AMA and they directly deploy applications but they do not understand what's going on in stage one and stage two at all so this leaves you also a bit underc confident and insecure about whether I really know the nuts and bolts whether I really know the details my plan is to go over every single thing without skipping even a single Concept in stage one stage two and stage number three so this is the plan which you'll be following in this playlist and I hope you are excited for this because at the end of this really my vision for this playlist is to make it the most detailed llm playlist uh which many people can refer not just students but working professionals startup Founders managers Etc and then you can once this playlist is built over I think two to 3 months later you can uh refer to whichever part you are more interested in so people who are following this in the early stages of this journey it's awesome because I'll reply to all the comments in the um chat section and we'll build this journey together I want to end this a lecture by providing a recap of what all we have learned so far this is very uh this is going to be very important because from the next lecture we are going to start a bit of the Hands-On approach okay so number one large language models have really transformed uh the field of natural language processing they have led to advancements in generating understanding and translating human language this is very important uh so the field of NLP before you needed to train a separate algorith for each specific task but large language models are pretty generic if you train an llm for predicting the next word it turns out that it develops emergent properties which means it's not only good at predicting the next word but also at things like uh multiple choice questions text summarization then emotion classification language translation Etc it's useful for a wide range of tasks and it's that has led to its predominance as an amazing tool in a variety of fields secondly all modern large language models are trained in two main steps first we pre-train on an unlabeled data this is called as a foundational model and for this very large data sets are needed typically billions of words and it costs a lot as we saw training pre-training gpt3 costs $4.6 million so you need access to huge amount of data compute power and money to pre-train such a foundational model now if you are actually going to implement an llm application on production level so let's say if you're an educational company building multiple choice questions and you think that the answers provided by the pre-training or foundational model are not very good and they are a bit generic you can provide your own specific data set and you can label the data set saying that these are the right answers and I want you to further train on this refined data set uh to build a bette model this is called fine tuning usually airline companies restaurants Banks educational companies when they deploy llms into production level they fine tune the pre-trained llm nobody deploys the pre-trend one directly you fine tune the element llm on your specific smaller label data set this is very important see for pre-training the data set which we have is unlabeled it's Auto regressive so the sentence structure itself is used for creating the labels as we are just predicting the next world but when we F tune we have a label data set such as remember the spam versus no spam example which I showed you that is a label data set we give labels like hey this is Spam this is not spam this is a good answer this is not a good answer and this finetuning step is generally needed for Building Product ction ready llm applications important thing to remember is that fine tuned llms can outperform only pre-trained llms on specific tasks

***

so let's say you take two cases right in one case you only have pre-trained llms and in second case you have pre-trained plus fine tuned llms so it turns out that pre-trained plus finetune does a much better job at certain specific tasks than just using pre-rain for students who just want to interact for getting their doubts solved or for getting assistance uh in summarization uh helping in writing a research paper Etc gp4 perplexity or such API tools or such interfaces which are available work perfectly fine but if you want to build a specific application on your data set and take it to production level you definitely need fine tuning okay now uh one more key thing is that the secret Source behind large language models is this Transformer architecture so uh the key idea behind Transformer architecture is the attention mechanism uh just to show you how the Transformer architecture looks like it looks like this and the main thing behind the Transformer architecture which really makes it so powerful are these attention blocks we'll see what they mean so no need to worry about this right now but in the nutshell attention mechanism gives the llm selective access to the whole input sequence when generating output one word at a time basically attention mechanism allows the llm to understand the importance of words and not just the word in the current sentence but in the previous sentences which have come long before also because context is important in predicting the next word the current sentence is not the only one which matters attention mechanism allows the llm to give access to the entire context and select or give weightage to which words are important in predicting the next word this is a key idea which and we'll spend a lot of time on this idea remember that the original Transformer had only the had encoder plus decoder so it had both of these things it had the encoder as well as it had the decoder but generative pre-train Transformer only has the decoder it did not it does not have the encoder so Transformer and GPT is not the same Transformer paper came in 2017 it had encoder plus decoder generative pre-rain Transformer came one year later 2018 and that only had the decoder architecture so even gp4 right now it only has decoder no encoder so 2018 came GPT the first generative pre-trend Transformer architecture 2019 came gpt2 2020 came gpt3 which had 175 billion parameters and that really changed the game because no one had seen a model this large before and then now we are at GPT 4 stage one last point which is very important is that llms are only trained for predicting the next word right but very surprisingly they develop emergent properties which means that although they are only trained to predict the next word they show some amazing properties like ability to classify text translate text from one language into another language and even summarize texts so they were not trained for these tasks but they developed these properties and that was an awesome thing to realize the pre-training stage works so well that llms develop all of these wonderful other properties which makes them so impactful for a wide range of tasks currently okay so this brings us to the end of the recap which we have covered up till now if you have not seen the previous lectures I really encourage you to go through them because these lectures have really set the stage for us to now dive into stage one so from the next lecture we'll start going into stage one and we'll start seeing the first aspect which is data preparation and sampling so the next lecture title will be be working with Text data and we'll be looking at the data sets how to load a data set how to count the number of characters uh how to break the data into tokens and I'll I'll start sharing sharing Jupiter notebooks from next time onward so that we can parall begin coding so thanks everyone I hope you are liking these lectures so lecture 1 to six we kind of like an introductory lecture to give you a feel of the entire series and so that you understand Concepts at a fundamental level from from lecture 7 we'll be diving deep into code and we'll be starting into stage one so I follow this approach of writing on a whiteboard and also coding um so that you understand the details plus the code at the same time because I believe Theory plus practical implementation both are important and that is one of the philosophies of this lecture Series so do let me know in the comments how you finding this teaching style uh because I will take feedback from that and we can build this series together 3 to four months later this can be an amazing and awesome series and I will rely on your feedback to build this thanks a lot everyone and I look forward to seeing you in the next lecture

***
