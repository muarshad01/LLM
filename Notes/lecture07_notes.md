## Tokenization 
* data preparation and sampling
* pre-training 
* We will build a tokenizer fully from scratch.
* We will also build an encoder-and-decoder from scratch

## Stages
* Stage-1: data preparation, attention mechanism and understanding the LLM architecture.  
* Stage-2: involves pre-training and building the foundational model so that involves the training Loop, model evaluation, and loading pre-trained weights.
* Stage-3: involves fine-tuning, training on smaller very specific datasets.

* LLMs are just NN.
*  right so you need data the parameters of the LLM are optimized and then we have some output.

## Tokenization Steps
1. Split the input text into individual words and sub tokens 
2. Convert these tokens into token IDs
3. Encode these token ID into Vector representation (Vector embedding)

***

## [10--15]

in today's lecture um so I just want to show you a visual for how the input text is given to the LLM so let's say this is an input text right this is an example.

remember the first step of tokenization we break this into individual words so the first word is this the second word is is the third word is an and the fourth word is example so this is the tokenized text and this is my step number one.

right now step number two is we need to convert each of these individual tokens into token IDs so this is the token ID 1 4013 token ID 2 2011 token ID 3 which is 302 and token ID 4 which is 1134 so every word or every token rather has a token ID associated with it I have currently given these numbers randomly but you need to assign a token ID for each word and then.

* the next step after getting these token IDs is to convert these token IDs into something called token embeddings or vector embeddings and then these Vector embeddings are then fed as input data to
the GPT or the LLM rather so even before coming to the training stage we need to do these many steps of uh pre-processing.

* nuances involved in tokenizing 
* Importing the dataset the demonstration today we are going to use a specific data set we are going to use this book book called The Verdict

* download and then download this book so we are going to look at this book and assume that this is the only uh training data 
* file we are going to open it in Python and we are going to read it and we are going to print the
* 8 total number of characters variable where we are storing whatever uh content python has read from this

***

* total number of characters are 20,479
* we are printing the first uh 100 characters of this file
* ion and the token IDs for the entire text which we have read for all
* 
* python Library which we we will be using for this is called as regular expression re
* uh so this module Pro provides regular expression matching operations and let's see how basically
* 
* hello and comma is one token and then it encounters a white space so it splits so


***

## [15 -- 20]

* world and full stop
* this and comma there is no white space so it prints out this and comma then it prints

* individual words white spaces and punctuation characters so see white
* split commas and periods so here you see the comma is included as
* we want to have comma and full stop

* split command you also include comma and you also include the full stop along with
* white spaces so white spaces are of
* comma is now a separate token full stop is now a separate token which was not the case before
* hello and comma were one token but now you'll see hello and comma are separate tokens
* awesome so now we have split based on comma and full stops also

* correct now the main another remaining issue is that our list still includes

each item in the result so we are looping over each item in the result so this is the result right so we are going
*  strip will
*  white spaces will not be returned Whenever there is actually a full world full word like
hello or comma or world or this or full stop then only item.
strip will return
* strip so for white spaces
* hello comma world full stop this comma is a
* develop a simple tokenizer whether we should
*  which is great however
* example python code is sensitive to indentation and spacing so if python code is used as the data set for
* Simplicity so that we have memory advantages but remember
* because we also want uh to have question marks quotation marks and double dashes
* separate tokens currently only full stop and comma are separate tokens right but we
* also want question marks quotation marks double dashes Etc separate tokens
 so let's look at this document again see

***

## [20 -- 25]

* double dashes then I'm sure there are question marks somewhere there are
* question marks there are exclamation marks in this document
*  we all want them as separate tokens right
* like colon semicolon question mark underscore exclamation uh quotation marks bracket
* 
#### next lecture it's called bite pair encoding 
* will be one token these two double dashes will be one token so this
* 
* vocabulary vocabulary is just all the list of our tokens but it's sorted in an alphabetical manner
* so if the if our data set or the tokens are

***

## [25 -- 30]

* vocabulary is constructed vocabulary is just a list of tokens which is sorted in alphabetical Manner and then what we do is that each unique token is mapped
to a unique integer which is called as the token ID.
* so it's as simple as that you map these tokens in alphabetical order and then to each token you assign a number.

* talk loaned edit worthon short story and assigned it to a python variable called pre-processed so remember pre-processed

***

## [30 --35]

* output of an llm from numbers
* turn token IDs back into text and for
* first will be the encode
* second will be the decode method in the encode method what will happen is
* we need an encode method uh and also the decode

***

## [35 -- 40]

* 
just take the uh tokens you'll take the tokens which are in this pre-process list and you'll convert them into token
* convert it into sample text by using the join method the final answer will be the the the fox chased and full stop now
*  we just need to construct a reverse mapping from the

 ***

## [40 --45]
  
* pass IDs as an input remember tokenizer do decode

*  chat GPT use a very special thing they use something which is called a __special context tokens__ to deal with Words which might not be present in the vocabulary so that an __special context__
* __special context tokens__ what we will do is that we will modify the tokenizer to handle
* 
***

## [45--50]

* tokens act as markers signaling the start or end of a particular

## [50-55]

* Quick Check we'll also print the last five

*** 

## [55-60]

***


* 
file so that you get the hang of it and you become better and better and better at this um just before concluding the
1:02:54
lecture I want to go through a quick revision of what all we have learned in today's lecture so in today's lecture we
1:03:00
essentially looked at this data preparation and sampling stage in building large language models from
1:03:06
scratch in particular we took a look at tokenization and uh the question is that
1:03:13
how do you prepare input text for training llms and we saw that this is divided into two steps in step one we
1:03:21
split the text into individual words so you have a essay or a book or a huge number of books you split that text into
1:03:28
individual words and then you convert these tokens into token IDs now when GPT actually uses
1:03:35
tokenizers it doesn't use individual words as tokens it even does subwords and we'll look at that in the next
1:03:42
lecture so if you look at how llms are actually trained you take the input text
1:03:48
you tokenize it into words you convert these words into token IDs and then the later step is we have to convert these
1:03:54
token IDs also into to Vector representations which we will come to later but in today's lecture we mostly
1:04:00
looked at step one which is tokenizing and step two which is converting tokens into token IDs right so initially what
1:04:07
we did is we downloaded and loaded the data set in Python we tokenized the
1:04:13
entire data set using Python's regular expression library and we used re. spit
1:04:19
we first used it used it at White spaces and then we added colons semicolons question marks Dash Dash character
1:04:26
Etc because we wanted to split all of those punctuations and even if you
1:04:32
search the dash dash in this text you'll see dash dash appear so many times so even we wanted to split that into
1:04:39
individual tokens awesome so this is what we did and then what we did is we
1:04:44
maintained the vocabulary we converted these tokens into token IDs so we saw
1:04:50
that what a vocabulary is that is it's simply a dictionary um and it's a mapping from
1:04:56
the tokens into the token IDs so the tokens are arranged alphabetically and
1:05:01
then each token is mapped to a token ID as you can see over
1:05:06
here so each unique token is mapped to a unique integer called token ID and why
1:05:12
are these token IDs necessary because you will see that these token IDs are then converted into token embeddings
1:05:18
which are fed as input to the GPT um great now then after that we
1:05:24
implemented a tokenized class in Python so that whenever we create a vocabulary we can create an instance of this
1:05:31
tokenizer class uh but we did not just Define the encode method in this class
1:05:36
we defined one more method which is the decode method so what the encode method did is that whenever sample text is
1:05:43
provided it converted it into tokens and then converted these tokens into token IDs great but the decode method did
1:05:50
exactly opposite it took the token IDs as an input it converted the token ID
1:05:56
into tokenized text and then it recover the sample text from the tokenized text
1:06:02
why is the decoder important because the GPT output is also in token IDs so we
1:06:08
need to convert it back to the sample text to to make sense of what the output
1:06:13
is so the tokenizer class which we defined had two methods the encode method and the decode method finally we
1:06:21
saw a problem that if some word is passed to the encoder which is not present in the vocabulary then the
1:06:27
encoder throws up an error to avoid this we need to add special context or
1:06:32
special text tokens to the vocabulary and uh one such special text token is
1:06:38
unknown so if some word is encountered which is not known we add the unknown token uh the second special context text
1:06:45
token which is also used in GPT is end of text so whenever there are multiple
1:06:51
text sources when we feed the text to the encoder we can separate them with the end text tokens we later saw that
1:06:58
many researchers used other to other special tokens also like beginning of sequence end of sequence padding
1:07:04
Etc actually when GPT was trained it only used the end of text uh token to
1:07:10
distinguish between the different text sources awesome so then we then we saw
1:07:17
that with these Special tokens even if you are given a new sentence whose words are not known uh such as the quickly
1:07:24
year which was not known it is just replaced by the token ID for the unknown token no error is shown
1:07:31
up uh now we saw that uh GPT did not actually use the unknown token so let me
1:07:38
take you to the end of this book end of this notebook so GPT models
1:07:44
did not use the unknown token then you might be thinking then how did GPT models deal with Words which are not
1:07:50
known they deal with Words which are not known by using something called bite pair encoding in bite pair en en in bite
1:07:57
pair encoding every word is not a token words themselves are broken down into subwords and then these subwords are the
1:08:04
token so let's say you have a word which is uh which is chased
1:08:10
right in the vocabulary which we have developed this chased itself is one token but in bite pair encoding it might
1:08:16
be possible that this chased itself is broken down into three sub tokens
1:08:22
CH as this is just for example but the this is what I mean by subw
1:08:28
tokens uh so subword tokens means breaking down one word itself into subwords and then using the subwords as
1:08:35
the tokens uh so GPT models use a bite pair
1:08:40
encoding tokenizer which breaks down words into subword units and we are going to cover that in a lot of detail
1:08:46
in the next lecture so uh thank you so much everyone for uh sticking with me for this lecture
1:08:53
and it's been a long lecture I think it's more than 1 hour but uh I think it is worth it I've not seen this much of a
1:09:00
detailed treatment on tokenization in any of the lectures many of the lectures are toy lectures which means they just
1:09:07
give you the basics but then don't show you things like end of text dealing with unknown words padding then pite pair
1:09:13
encoding or some things which are actually used in llms and that is what I wanted to cover so my lecture style will
1:09:21
be a mix of whiteboard writing on whiteboard and uh also showing you code here in Jupiter notebook I'll be sharing
1:09:29
this code file with everyone so you can run this code and I highly encourage you to run this code on your own thank you
1:09:35
so much everyone and I look forward to seeing you in the next lecture


Show chat replay










