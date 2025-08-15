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

[30 --35]

* reverse mapping which is called as a decoder so this is what uh I have
31:12
mentioned here later when we want to create the output of an llm from numbers
31:17
back into text we also need a way to turn token IDs back into text and for
31:24
this we create an inverse version of the vocabulary that Maps token IDs back to the corresponding text tokens and we'll
31:31
see how to do that for now what we are going to do is we have understood enough about
Simple Tokenizer class in Python
31:37
tokenization that we are going to implement a complete tokenizer class in Python this class will have two methods
31:44
it will have an encode method and it will have a decode method let me show you what it actually means so this
31:50
tokenizer class which we are going to implement in Python it will have two methods the first will be the encode
31:56
method and the second will be the decode method in the encode method what will happen is
32:04
that sample text will be converted into tokens and then tokens will be assigned token IDs based on the vocabulary
32:10
exactly what we saw like over here in the decode method exactly
32:17
reverse things would happen so in the decode method we start with token IDs we convert it into individual tokens and
32:23
then we get back the sample text so I hope you understand the difference between the encode method and the decode
32:30
method here because this is the exact same difference which will show up in the encoder and decoder block of the
32:36
Transformer architecture or rather this is the this is a good way to understand that intuition in the encode in the
32:43
encode block what we do is we take sample text we convert it into tokens and then we convert it into token IDs
32:49
that feeds as the training data to the llm but when the llm gives its output in the form of token IDs we need to convert
32:56
it back to tokens and and then back to the sample text so that we know what the output is in terms of
33:03
sentences so that's why when we implement the tokenizer class we need an encode method uh and also the decode
33:10
method so the encode method will take text as an input and give token IDs as output the decode method will take IDs
33:18
token IDs as input and will give text as an output now let's see how to actually
33:26
create this simple token izer class in Python first so there are if you see
33:32
three methods the init method which is called by default when an instance of this class is created and let us look at
33:38
the arguments of the init method it takes wcab so when you create an instance of the tokenizer class you have
33:45
to pass in the vocabulary right and remember the vocabulary is nothing but a mapping from tokens to token IDs right
33:53
so then after this uh instance is created St Str to in which is string to
33:59
integer will just be the vocabulary because the vocabulary is already a mapping from string to integer or tokens
34:05
to integer and then the integer to string is basically reverse so what you do is that you take the string and
34:13
integer in the vocabulary and then for every integer you uh mention which token
34:19
it is so for S you can think of as the token and for I you can think of as token ID so what we do in this int to
34:26
string variable variable is that uh so we take the token and we take the token
34:33
ID in the vocabulary and then we just flip it then we say that for this token ID this is the particular token remember
34:40
this into string will be needed for the decoder method when we have the token IDs and we want to convert it back to
34:48
tokens so uh in the encode method the exact same pre-processing steps will
34:54
happen as we had seen before for tokenization what we'll do if some random text is given to us we will take
35:01
that text we will split it we'll split it based on the comma based on the full stop based on the colon based on
35:07
semicolon Etc into individual tokens and then we'll get rid of the white spaces
35:13
what we had seen before and then these will be individual tokens right so up till now we are at this part where we
35:20
have converted the sample text into individual tokens and then we'll convert these into token IDs so that's the last
35:26
step over here after you get the individual tokens in this list called pre-processed what you have to do is
35:33
that you have to use this St Str to in dictionary which is basically just the vocabulary and then you have to uh
35:40
assign a token ID for each token so remember the Str str2 int is basically
35:46
just converting tokens to token IDs using the vocabulary which is passed into this tokenizer class so you will
35:53
just take the uh tokens you'll take the tokens which are in this pre-process list and you'll convert them into token
35:59
IDs that's the encoder method that's it in the decoder method what you are actually doing is that you are uh you
36:08
are using this reverse dictionary which is integer to string which is basically token ID to token and then you are
36:15
converting the token IDs into individual tokens that's the first thing so let's
36:20
see the decode method you first convert the token IDs into individual tokens and then what you do is you join these
36:27
individual tokens together so this this join is used so first you convert the
36:32
token IDs into tokens and then you join the individual tokens together that's it
36:38
and then here what we are doing is we are going to replace spaces before the punctuations so an example here would be
36:46
let's say if the tokens let's say in the decoder if the tokens are the fox
36:54
chased and question and full stop right now if these are the tokens and if we
37:01
convert it into sample text by using the join method the final answer will be the the the fox chased and full stop now see
37:09
the problem here is that there is a space between the there is a space here between the full stop and the chased so
37:18
we need to get rid of this space so then the final answer would be the fox
37:24
chased and full stop and this is the the same for question mark Etc so in this
37:30
second sentence here what we are actually doing is that we are getting rid of all the spaces before the
37:37
punctuation so that it becomes a complete sentence so this is the decoder method
37:42
it's actually very simple we are just going to uh use the encode method to
37:48
convert uh sample text into token IDs and we are going to use the decode method to convert token IDs back into
37:55
sample text and for that we needed to write two methods the encode method and the decode method so remember this
38:02
tokenizer class takes the vocabulary already as an input so that so we already have the mapping from tokens to
38:09
token IDs because that is inherently present in the vocabulary we just need to construct a reverse mapping from the
38:15
token IDs back to the tokens and then use that in the decode method that's it
38:21
and to convert the text into tokens we use the same re dos split and item do
38:27
string which we had seen earlier in today's lecture so this is essentially the
38:33
tokenizer class which we have created awesome now let's move to the next step
38:39
so what we can do is that now we can instantiate a tokenizer object from this class right and uh tokenize a passage
38:46
from the short story which we have downloaded to test it out in practice so here you see I'm creating an instance of
38:53
this class I have passed the vocabulary as an input what is this vocabulary this vocabulary is basically the uh
39:00
vocabulary of tokens and token IDs which we have converted our input text so this
39:05
is our input text and we have converted this into tokens and assigned the token ID to each this is my vocabulary now
39:12
what I'm doing is I'm actually uh creating an instance of the yeah I'm
39:18
creating an instance of the simple tokenizer version one class which we have defined over here by passing in
39:24
this vocabulary as an input right and this is then defined as tokenizer great
39:30
and uh so the text which I'm going to pass in now is this it's the last he
39:35
painted you know Mrs gburn said with pardonable pride this is the text and now I'm going to test out this encode so
39:42
remember this encode takes in the text as an input so you need a text right to convert into IDs so what the encode
39:49
method will do always remember this schematic always remember the schematic the encode method will convert the text
39:56
into token ID is so this is the text and then we apply the method tokenizer do
40:02
encode and text so what when you print the IDS you will see these IDs of the text which means that our encoder has
40:10
successfully converted this text into token IDs that is exactly what we wanted
40:15
right so the code above prints the token IDs next let's see if we can turn these token IDs back into text so now what
40:22
we'll be doing is we'll be using tokenizer do decode and use these IDs so
40:27
tokenizer decode and IDs uh and pass IDs as an input remember tokenizer do decode
40:33
takes the IDS as an input so I'll pass these IDs which we have printed over here and let's see whether it recovers
40:40
this text so the text which is recovered is it's the last he painted you know Mrs gburn said with pardonable pride amazing
40:47
right because it's exactly the same text which we had given uh to the encoder and now it has been decoded by the decoder
40:55
So based on this output above we can see that the decode method successfully converted the token IDs back into the
41:01
original text which is exactly what we wanted which means the encoder and decoder are working right so so far so
41:09
good we have implemented a tokenizer which is capable of tokenizing and DET tokenizing text based on a snippet from
41:16
the training set so this sentence which we gave to the encoder was from the training set so we knew that the words
41:23
will be in the vocabulary but what if the sentence which is given is not present in the vocabulary so we have a
41:31
vocabulary here which is 1130 the size of the vocabulary is 1130 right what if
41:37
I give a sentence to en code which is not present in the vocabulary so let's try this let's say the text is hello do
41:43
you like T now uh hello is probably not there in this hello is not there in this
41:50
short story so let me search hello it's not there let's see if T is there t is
41:56
actually T is there but hello is not there so hello will not be there in the
42:02
vocabulary right so now let let me see how what's my answer here because now I
42:08
have asked the tokenizer to encode something which is not present in the vocabulary okay so if I run this you see
42:13
you get an error here because hello it does not know what to do with this word hello because it is not present in the
42:20
vocabulary so the problem is that the word hello was not used in the verdict short story and hence it is not
42:26
contained in the vocabulary now this actually highlights the need to consider large and diverse
42:32
training data sets to extend the vocabulary when working with llms we do not want this problem right if our
42:38
training data set is small and if some user gives a new word to chat GPT which it does not even know we don't want our
42:44
we don't want an error that's why huge number of data set is used for training large language
42:50
models in fact llms like chat GPT use a very special thing they use something
42:56
which is called a special context tokens to deal with Words which might not be present in the vocabulary so that an
43:02
error message is not shown such as this and we will come to that in the next section but here I just want to
43:08
illustrate that the reason for having a a large and diverse training data set is
43:14
because uh we we want many words to be present in our vocabulary it does not
43:20
make sense to have a smaller and a shorter vocabulary because what if the user gives a new word which is not
43:25
present in the vocabulary we don't want that there is a way to deal with this by using something called special context
43:31
tokens so let's look at this in the next section now let us have a look at the
Special Context Tokens
43:36
last section in today's lecture which is dealing with special context tokens these content are not usually
43:44
covered in other lectures but they are very important especially since we are going to be building llms from scratch
43:51
we also want to understand how real life llms work and special context tokens
43:57
play an important role so until now we have implemented a simple tokenizer right and applied it to a passage from
44:04
the training set but the main question we encountered before is that if there is a word in the text which is not there
44:11
in the vocabulary how do we encode this word um so we will be implementing some
44:18
uh things which are called as special context tokens what we will do is that we will modify the tokenizer to handle
44:25
unknown words uh and we will Implement a python class
44:30
which is called a simple tokenizer version two so we have already implemented uh simple tokenizer version
44:37
one but here it did not have the provision to handle the unknown tokens so in this version two we will also uh
44:46
Implement uh the simple tokenizer version two to have the provision to
44:53
handle the unknown tokens in particular we will be learning about two main tokens the first is this this token
44:59
which is unknown for an unknown word and the second token is for end of
45:04
text so let me again go back to this uh whiteboard over here and write a bit
45:11
about why we are exactly using these tokens so uh let's take a sentence which
45:18
is the fox chased the dog let's say this is my sentence right
45:25
now and I have tokenized it into the Fox Chase the dog and here's the vocabulary
45:32
so it's arranged in alphabetical order and there are token IDs right now to
45:37
this existing vocabulary we are going to add two more tokens the first is the Unk
45:42
which is unknown and we'll assign a token ID and then we'll also have end of
45:47
text and then we'll assign it a token ID these two are the last
45:53
two uh tokens in the vocabulary so the token is corresponding to these two will
45:58
be the largest so what will happen is that let's say if some new sentence or some
46:04
new word is given so let's say uh the fox chased the dog quickly if that is
46:12
the word all these other words like the Fox Chase the dog will be converted into
46:18
token IDs but for quickly it will have a token ID of 783 which is the token ID of
46:24
unknown and why because quickly was not in the vocabulary our vocabulary only consisted of chased dog fox and the so
46:32
quickly is an unknown word and so it will receive a token ID which we have
46:38
reserved for unknown words and then you might be asking what about this end of text so end of text is something which
46:46
is a bit different uh when we are working with multiple text sources we typically add
46:52
end of text token between the text so let us look at four text sources here
46:58
this is the text Source One let's say it comes from one book let's say this is the text Source two let's say it comes
47:04
from another news article text Source three which comes from an encyclopedia
47:09
and text Source Four let's say which comes from an interview let's say these are are our
47:15
training sets usually all of these are not just collected into one giant
47:20
document or all of these sentences are not just stacked up together we usually
47:25
after after this initial text is fed as an input we have this end of text token
47:31
which means that the first text has ended and now the second text has started after the second text ends then
47:38
we again have this end of text token then the third text starts and after the third text ends then we again have the
47:45
end of text token and then the fourth text starts so the end of text token is
47:51
basically added between the texts so essentially the end end of text
47:57
tokens act as markers signaling the start or end of a particular
48:04
segment this leads to more effective processing and understanding by the llm
48:09
it's very important to add these end of text tokens because then llm treats the initial so let's say end of text there
48:16
is before the end of text which is text Source One and after the end of text which is text Source two if end if end
48:22
of text was not there the llm would have mixed all of this together right uh but end of text tokens allows the llm to
48:29
process the data and understand the data in a much better Manner and in fact when GPT was trained the end of text tokens
48:36
were used between different text sources this is very important to note and only
48:41
students who try to understand tokenization in detail know about such specifics so these are the two tokens
48:49
which we will be considering in this section the first is the unknown token and the second is the end of text token
48:56
so let's read bit here we can modify the tokenizer to use an unknown token if it encounters a word that is not part of
49:02
the vocabulary great furthermore we also add a token between unrelated text which
49:08
is the end of text token so for example when training GPT
49:13
like llms on in multiple independent documents it is common to insert a token
49:19
before each document or book that follows a previous text Source basically
49:24
this is exactly the end of text token which has been written over here I'll be sharing this notebook with all of you so
49:30
no need to worry if you miss certain portion or want to revise certain portion again now what we'll be doing as
49:37
I mentioned over here uh is that we'll be modifying the vocabulary we'll be augmenting the
49:43
vocabulary and uh how will we modify or augment the vocabulary we will include
49:48
special tokens we will include two special Tokens The Unknown and the end of text token and we will add these to
49:55
the list of all the unique words that we created in the vocabulary in the previous section so let's look at this
50:01
pre-processed so remember pre-processed is the vocabulary uh or U so
50:08
preprocessed was our list and then later we converted this into vocabulary which
50:13
was stored in vocab right and remember the size of the vocab is 1130 and now we
50:18
are going to add two more tokens so the size will be 1132 so let us again start with
50:24
pre-processed we will sort this and then we'll add two tokens so we'll add the end of text token and then we'll
50:31
add the unknown token and that is done by using the python command which is called extend so what this does is that
50:37
it just adds two more additional entries to the list the sorted list in this case
50:43
and then again we'll write the same command which we did earlier right we first enumerate all the tokens in the
50:49
pre-processed list and then for each token we assign an integer and this enumer it will make
50:57
sure that the tokens are anyways arranged in alphabetical order and then for each of these token we assign an
51:02
integer which is the token ID that's how we create the vocabulary now if you print out the length of the vocabulary
51:09
you will see that the length of the vocabulary is 1132 and if you remember without adding
51:16
these two tokens the length of the vocabulary was 1130 so now the length has increased by
51:21
two that is great right uh that makes sense because two more tokens have been added so the
51:28
vocabulary is extended great so based on the output of the print statement above
51:34
the new vocabulary size is I should mention here actually 1132 uh so this should be 1132 and in
51:41
the previous section it was uh 1130 and let me run this yeah so based on the
51:48
output of the print statement about the new vocabulary size is 1132 and the vocabulary size in the previous section
51:54
was 1130 good so what we'll do as an additional Quick Check we'll also print the last five
52:01
entries of the updated vocabulary so if you print out the last five entries in the updated vocabulary you will see
52:07
let's look at the last two entries these are the end of text and the unknown so the end of text has a token ID of 1130
52:15
and the unknown has a token ID of 1131 awesome so now the vocabulary is updated
52:21
and now what we'll be doing is that we'll be extending the simple tokenizer class uh this is the version two
52:28
basically many things will remain the same this initialization will remain the same it will initialize two dictionaries
52:33
the string to integer dictionary which is the vocabulary itself which converts the tokens into token IDs and the
52:40
integer to string dictionary which has token IDs and then a string mapped to each token
52:46
ID this encode now let's look at this encode the first two sentences are very similar to what we saw before we split
52:54
on the comma full stop colon semicolon question mark underscore explanation
52:59
quotation and bracket and then we remove the white spaces using item. strip but
53:05
we add one more thing what we add is that if the item or if the particular
53:11
entry is not present in the vocabulary uh the token which is assigned to that entry is unknown so
53:19
let's say you are scanning the text right and if you come across a word in the text which is not in the vocabulary
53:25
so if the item is not present in this string to integer vocabulary which is the vocabulary which we have passed uh
53:33
then it is it should be the unknown token we replace that with the unknown
53:38
token and then we convert all of these tokens to token IDs so in this step what
53:43
will happen is that all the words in the input text which are not in the vocabulary will be replaced by the unknown token and then in this step that
53:50
is the main encoder step where the tokens are converted into token IDs in the decoder part part everything
53:57
will uh almost stay exactly the same nothing changes we first convert the
54:03
token IDs back into tokens and then we join them together and uh before the
54:08
punctuations there are spaces so we get rid of these spaces and then we return the decoded text so this is the change
54:15
in this tokenizer simple tokenizer version two the only change is uh adding
54:21
this so if some word is not known you encode it with unknown and then assign the correspond oning ID to it so if
54:27
something is unknown the ID will be 1131 great now let's actually try some
54:32
things okay let's say the first text source which we have is hello do you like T the second text source which we
54:39
have is in The sunl Terraces of the palace these are two text sources now what we'll be doing is we will construct
54:47
a text which joins these two and we will add an end of text at the end of the
54:52
first text Source why are we doing this because we saw that this is usually how
54:57
it's done in practice in GPT Etc if you provide one text source and if you have another text Source you don't just join
55:03
them together you split them with this end of text token so this is what we are
55:09
exactly doing we are now constructing a text which will feed as an input to the encoder right but we add this end of
55:16
text so the text Will which will essentially feed to the encoder is hello
55:21
do you like T end of text in The sunlit Terraces of the palet s this is the text
55:27
which we are feeding to the encoder and now we are asking the encoder to encode these into token IDs so let's look at
55:35
this sentence in detail this word hello is not present in our vocabulary so if
55:40
if we pass it into this tokenizer it will hit this this statement where the
55:46
word is not present in the vocabulary so it will replace it with the unknown token and this end of text again end of
55:53
text is actually present in the vocabulary because we have added it right now so the token ID corresponding
55:58
to end of text will be 1130 and the token ID corresponding to hello will be 1131 right so let's check if this is
56:05
actually true so I'll now run tokenizer encod text and see the token ID for
56:11
hello is 11131 this is exactly what we had expected right because uh hello is
56:17
not present in the main vocabulary so it's actually an unknown word so it token ID should be 1131 awesome and then
56:24
for end of text let's see the token ID yeah this this is the token ID for end
56:30
of text the token ID is 1130 so now there is no error which is coming here earlier when we passed this
56:37
hello do you like T there was an error which we are getting that hello is not present in the vocabulary but now this
56:43
error is not coming because we have taken care of it we have added the unknown token in the vocabulary and so now the tokenizer
56:50
encodes the text uh in a correct manner that's awesome so now what we'll be
56:56
doing is that we'll be actually using the decode function now and we'll pass
57:02
the encoded text which are the token IDs these IDs into the tokenizer do decode
57:09
so let's see what the tokenizer decode so when these are passed into the decoder the decoder is unknown do you
57:15
like T end of text in The sunlit Terraces of the unknown so actually
57:20
there are two unknown text here hello is an unknown and Palace is also an unknown so the token IDs for both of these are
57:28
1131 so when you decode these IDs you will get unknown do you like T so which
57:33
because hello is not known then end of text and then in The sunl Terraces of
57:39
the unknown so here again unknown means the palace which was not in the vocabulary so the encoder and decoder
57:45
are working perfectly and we are able to handle the unknown words now uh we are
57:50
able to handle the unknown words actually quite effectively because we replace them with the unknown token
57:56
and the end of text is also being captured pretty effectively so when we actually make the text itself we have to
58:03
add end of text before we pass it to the encoder and subsequently to the decoder
58:08
So based on comparing the det tokenized text with the original input text we know that the training set which is the
58:15
verdict book did not actually contain the words hello and the palace because
58:20
both of them are replaced by this unknown token here so we will not get errors this way once we take into
58:27
account the special uh context tokens awesome so uh let me just add a last
Additional context tokens
58:35
note about the special context token so up till now we have discussed
58:40
tokenization as an essential step in processing text as input to the llms
58:45
however along with these Special tokens there are some other tokens which also researchers consider so this is the
58:51
beginning of a sequence so BOS token so we saw the end of text right some
58:56
researchers also consider BOS so this token marks the start of a
59:02
text it signifies to the llm where a piece of context begins or where a piece
59:08
of content begins right then the second token is eos which is end of sequence
59:14
right so this token is positioned at the end of a text and especially useful where concatenating multiple unrelated
59:21
text it's similar to end of text uh then the third token which is
59:26
important is the padding token so when training llms with batch sizes larger than one the batch might contain texts
59:34
of varying lengths so to ensure that all the texts in different batches have the same length the shorter texts are
59:41
extended or padded using the pad token up to the length of the longest text in the batch so imagine the different text
59:48
is being put in batches to the llm for parallel processing we'll look at this in detail so no need to worry about this
59:55
right now but but just remember that for efficient Computing llm parallely processes the batches so each batch
1:00:02
contains text which have different sizes the shortest Texs are augmented with
1:00:08
this pad token to match uh their length with the largest text and now we know
1:00:14
how to add these special tokens right we just augment the vocabulary with these tokens and whenever we pass text to the
1:00:20
encoder we we may add things like beginning of sequence end of text the pad token Etc
1:00:27
so we only saw uh the unknown and the end of text special context tokens but
1:00:33
there are three other BOS beginning of sequence and EOS end of sequence and Pad
1:00:38
which is padding now here I want you to note that the tokenizer which is used
1:00:44
for GPT models does not need any of these tokens mentioned above but it only
1:00:50
uses the end of text token for Simplicity so gpt3 gp4 when they are trained they don't use BOS uh padding
1:00:58
they don't even use the unknown token they only use the end of text token for
1:01:03
Simplicity and this I also shown over here so the end of text which is used is
1:01:08
exactly how it is used in GPT training as well okay and the last sentence is that
1:01:14
the tokenizer used for GPT models also does not use an unknown token for out of
1:01:19
vocabulary words then you might be thinking how does GPT deal with unknown tokens right so then there is something
1:01:26
called as the bite pair encoding tokenizer which GPT models actually use which breaks down words into subword
1:01:33
units So currently what we have done is each word is essentially one token right and each punctuation mark each colon
1:01:40
semicolon is one token but what actually GPT does for tokenization it it uses
1:01:46
something called bite pair encoding or BP tokenizer and that automatically
1:01:52
deals with unknown tokens because in that tokenizer one word is is not essentially one token but even the words
1:01:58
are broken down into subwords to generate individual tokens uh we'll come to this in the next lecture but I wanted
1:02:05
to show this lecture specifically for showing you how you can do your own tokenization from scratch I could have
1:02:13
directly showed you bite pair encoding with GPT users but then that you would not have appreciated uh if you were to
1:02:20
tokenize yourself how would you do it from scratch of course we'll look at the bite pair and coder tokenizer in a lot
1:02:27
of detail in the next lecture to understand how GPT actually tokenizes but I wanted to give you this intuitive
1:02:33
feel so that you don't think tokenization is a hard process I'll be sharing this code file with all of you
Lecture recap
1:02:40
so that you can run this and I actually highly encourage you to run this for a different uh book or a different txt
1:02:47
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







