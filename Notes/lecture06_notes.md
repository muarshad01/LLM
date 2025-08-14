## Stages of building a LLM
* stage-one
* stage-two
* stage-three

#### Stage One: Building a LLM
* We're going to look at the building blocks, which are necessary before we go to train the LLM.
* Data pre-processing and sampling in a very specific manner
* We need to understand the attention-mechanism and we will need to understand the LLM architecture so in the stage-one, we are going to focus on these three things understanding
1. How the data is collected from different datasets?
2. How the data is processed?
3. How the data is sampled?

#### Attention Mechanism
* how to C out the attention mechanism completely from scratch in Python
* what is meant by key-query-value
* what is the attention-score
* what is positional encoding
* what is Vector embedding

* how to stack different-layers on top of each other?
* where should the attention-head go?

* so what exactly we will cover in data preparation and sampling
*  __Tokenization__: How to break them down into individual tokens
*  __Vector embedding__: After tokenization, every word needs to be transformed into a very high-dimensional Vector space so that the semantic  meaning between words is captured.
*  Encode every word so that the semantic meaning between the words are captured
*  __positional encoding__ the order in which the word appears in a sentence is also very important
*  how to construct batches?
*  how many words should be taken for training to predict the next output?
*  How to basically Fe the data in different sets of batches
*  __data batching sequence__

#### Second Point
* I mentioned here is the attention mechanism
* so here is the attention mechanism for the Transformer model
* we'll first understand what is meant by every single thing here
* what is meant by multi-ad attention
* what is meant by Mas __multi-head attention__
* what is meant by __positional encoding__
* __input embedding output embedding__

* we'll break it down into epox and we will compute the __gradient__ of the loss in each Epoch and we'll update the parameters towards the end
*   we'll also do __model evaluation__ and loading pre-train weaps
*    evaluation training    and validation losses
*    then we'll write the LLM training function
*     Implement function to save-and-load the LLM weights
*   load pre-trained weights from open AI into our LLM.
*
#### stage-two Steps
1. training Loop plus
2. model evaluation plus
3. loading pre-trained weights to build our foundational model.
   

#### stage-three 
* Fine-tuning the LLM so if we want to build specific applications we will do fine tuning in this playlist
* 

***

#### First Application
* spam whereas hey just wanted to check if we are still on for dinner tonight let me know this will be not spam so we will build a LLM this application which classifies between spam and no spam.
* and we cannot just use the pre-trained or foundational model for this because we need to train with labeled data to the pre-train model.
*  we need to give some more data and tell it that hey this is usually spam and this is not spam can you use the foundational model plus this additional specific label data asset, which I have given to build a fine-tuned llm application for email classification.
   #### Second application 
  * ChatBot which basically answers queries.
  *  so there is an instruction there is an input and there is an output and we'll be building this chatbot after fine-tuning the LLM.
  *  [LangChain]()
  *    Tools like AMA and they directly deploy applications but they do not understand what's going on in stage-one and stage-two at all.
  *  next lecture we are going to start a bit of the Hands-On approach
  *  okay so number one LLMs have really transformed uh the field of NLP they have led to advancements in generating understanding and translating human language this is very important.
  *   uh so the field of NLP before you needed to train a separate algorith for each specific task but __LLMs are pretty generic__
  *   if you train an llm for predicting the next word it turns out that it develops __emergent properties__, which means it's not only good at predicting the next word but also at things like uh MCQs, text summarization, then emotion classification, language translation, etc.
  *   
  *    secondly all modern LLMs are trained in two main-steps
  *     first we pre-train on an unlabeled data this is called as a foundational model
  *  and for this very large datasets are needed typically billions of words and it costs a lot as we saw training pre-training GPT-3 costs $4.6 million.
  *   so you need access to huge amount of data compute power and money to pre-train such a foundational model.
  *    now if you are actually going to implement an LLM application on production level so let's say if you're an educational company building MCQs and you think ...usually airline companies, restaurants, Banks, educational companies,
  * ... __Auto regressive__ so the sentence structure itself is used for creating the labels ...
  * llm applications important thing to remember is that fine tuned LLMs can outperform only pre-trained llms on specific tasks

***

so let's say you take two cases right in one case you only have pre-trained llms and in second case you have pre-trained plus fine tuned llms so it turns out that pre-trained plus finetune does a much better job at certain specific tasks than just using pre-rain for students who just want to interact for getting their doubts solved or for getting assistance uh in summarization uh helping in writing a research paper Etc gp4 perplexity or such API tools or such interfaces which are available work perfectly fine but if you want to build a specific application on your data set and take it to production level you definitely need fine tuning okay now uh one more key thing is that the secret Source behind large language models is this Transformer architecture so uh the key idea behind Transformer architecture is the attention mechanism uh just to show you how the Transformer architecture looks like it looks like this and the main thing behind the Transformer architecture which really makes it so powerful are these attention blocks we'll see what they mean so no need to worry about this right now but in the nutshell attention mechanism gives the llm selective access to the whole input sequence when generating output one word at a time basically attention mechanism allows the llm to understand the importance of words and not just the word in the current sentence but in the previous sentences which have come long before also because context is important in predicting the next word the current sentence is not the only one which matters attention mechanism allows the llm to give access to the entire context and select or give weightage to which words are important in predicting the next word this is a key idea which and we'll spend a lot of time on this idea remember that the original Transformer had only the had encoder plus decoder so it had both of these things it had the encoder as well as it had the decoder but generative pre-train Transformer only has the decoder it did not it does not have the encoder so Transformer and GPT is not the same Transformer paper came in 2017 it had encoder plus decoder generative pre-rain Transformer came one year later 2018 and that only had the decoder architecture so even gp4 right now it only has decoder no encoder so 2018 came GPT the first generative pre-trend Transformer architecture 2019 came gpt2 2020 came gpt3 which had 175 billion parameters and that really changed the game because no one had seen a model this large before and then now we are at GPT 4 stage one last point which is very important is that llms are only trained for predicting the next word right but very surprisingly they develop emergent properties which means that although they are only trained to predict the next word they show some amazing properties like ability to classify text translate text from one language into another language and even summarize texts so they were not trained for these tasks but they developed these properties and that was an awesome thing to realize the pre-training stage works so well that llms develop all of these wonderful other properties which makes them so impactful for a wide range of tasks currently okay so this brings us to the end of the recap which we have covered up till now if you have not seen the previous lectures I really encourage you to go through them because these lectures have really set the stage for us to now dive into stage one so from the next lecture we'll start going into stage one and we'll start seeing the first aspect which is data preparation and sampling so the next lecture title will be be working with Text data and we'll be looking at the data sets how to load a data set how to count the number of characters uh how to break the data into tokens and I'll I'll start sharing sharing Jupiter notebooks from next time onward so that we can parall begin coding so thanks everyone I hope you are liking these lectures so lecture 1 to six we kind of like an introductory lecture to give you a feel of the entire series and so that you understand Concepts at a fundamental level from from lecture 7 we'll be diving deep into code and we'll be starting into stage one so I follow this approach of writing on a whiteboard and also coding um so that you understand the details plus the code at the same time because I believe Theory plus practical implementation both are important and that is one of the philosophies of this lecture Series so do let me know in the comments how you finding this teaching style uh because I will take feedback from that and we can build this series together 3 to four months later this can be an amazing and awesome series and I will rely on your feedback to build this thanks a lot everyone and I look forward to seeing you in the next lecture

***


