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

* token IDs also into to Vector representations which we will come to later but in today's lecture we mostly
* called __bite pair encoding__ in bite pair en en in bite
1:07:57
pair encoding every word is not a token words themselves are broken down into subwords and then these subwords are the token so let's say you have a word which is uh which is chased
*  chased itself is one token but in bite pair encoding it might
* 
*** 
