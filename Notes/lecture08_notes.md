#### Algorithms
* Word based
* Sub-word based
* Character based

#### Bite Pair Encoding (BPE)
ChatGPT, GPT-2, GPT-3

***

#### [5 -- 10]

vocabulary is just a dictionary with a collection of tokens

arranged in ascending order and every token corresponds to a token ID great now another problem with word based tokenization scheme is that uh 

* out of vocabulary (OOV) problem it's also computationally and memory efficient because the vocabulary size is 
* same meaning words [boy, boys]
***

* subword based tokenization and the bite pair encoding which we are going to see is an example


* frequently used word into smaller subwords so if there are some words which are coming frequently you should retain those words as it is right so then it retains this from the word tokenization then the rule two is that if there are some words which are very rare which are not occurring too many times then you split these words into smaller meaningful subwords this is extremely important this second part basically says that if there are some words which
12:59
* see why it is a mix between word tokenizer and character tokenizer the first rule implies that if the words are occurring many times you return it as a word so this is taken this is a feature taken from the word tokenizer
the second rule implies that if the word is rare you can go on splitting it into further subwords and if needed you can drop down to the Character level we don't always drop down to the character level we even stay in the middle but
*  divide this word boys into smaller meaningful subwords so boys is divided
* this is what is basically
* subword splitting helps the model learn that different words with the same root word such as for example token tokens

***

* and tokenizing all of these three words essentially have the same root word right token so subord splitting helps the model understand that these different words essentially have the same root word and they are similar in meaning this meaning is lost in word based tokenization and even character based tokenization that is number one the second advantage of subord tokenization is that it also helps the model learn that let's say tokenization and modernization are made up of different root words token tokenize and modernize but have the same suffix isation and are used in same syntactic
* common suffix which appears in both of these words all of these are advantages of breaking down one word into subwords this is what is majorly done in subword tokenization so why are we learning about subword tokenization and what's the relation between bite pair encoding Byte Pair Encoder (BPE) Algorithm
* modern llms like gpt2 and gpt3 employ bite pair encoding so let us look at the bit of a history
* it is basically a data compression algorithm uh and what is done is that we
* so we find these pairs which occur the most frequently and then we replace them with a bite which does not exist in the
* this is a compressed data compared

***

* another variable so this compression actually stops here you can go one more layer Deeper by further com replacing the zy with another variable which is let's say w so then it will be WD w a and then you
will stop so you will compress it further like this so you see the original data has been compressed to
this right uh to this compressed version U using the bite pair algorithm so the
algorithm itself is pretty simple you scan the data from left to right you identify the bite pairs which occur the most and then you replace them with a bite which does not exist in the data and you do this iteratively until you
* reach a stage where no bite pair occurs more than once that's it that is the simple bite pair encoding
algorithm now you might think okay what has that got to do with large language models right I understand this algorithm BPE for Large Language Models
and I understand how it compresses a given data sequence but what has that
got to do with large language models well it turns out that the we slightly
tweak the bite pair encoding algorithm and use this to convert our entire
sentence into subwords which will be very useful for us and I'll show you exactly how I'm going to do that so the
* bite pair encoding for llm ensures that the most common words in the vocabulary
are represented as a single token remember rule number one and rare words
are broken down into two or more subord tokens this is exactly the same rules which we had looked at the rule number
one and rule number two so rule number one is that most commonly used words
should not be split and second is that that rare words should be split into meaningful subwords now let's see how uh
* how it's related to The Bite pair encoding algorithm which we saw and we will be looking at a practical example
* called as Data Corpus which is this so we have
* that's it similarly if we were to use the Character level tokenization then the tokens will be individual characters
* pre-processing step

* old slw older becomes older slw finest becomes finest slw and lowest becomes lowest /w now remember here that if we use the word based24:59 tokenization uh there is no meaning which is captured so the fact that old is the common root between old and older is not captured number one EST is the
common root between finest and lowest that's not captured so word based
tokenization character based tokenization have so many problems because they don't capture these meanings or root words and towards the
end of this section we'll see how subword based tokenization using the bite pair encoding algorithm actually

***

* characters um and then make a frequency table so what we are going to do is that we are going to take all these words old older finest and lowest so old appears seven times in the data set older appears three times in the data set these are the frequencies finest appears nine times in the data set and lowest essentially appears four times in the data set so what we are going to do right now is we are going to split these
* words into individual characters so then here is the table which I have made remember slw is also there uh so old so
* all words have/ W right and totally how many words do we have 7 + 3 10 + 9 19+ 4
* frequency table list so here you can see
* A because AA appeared the most so we looked at that
* most frequent pairing we will merge them

***

* the most frequent pairing right so it so you the way to do it is look at the first character which appears the most so e is that character which appears 16 times right so if you want to look at the pairing which appears most it most probably starts with e so it turns out if you look at these words e and s is the pairing which appears the most number of times so e and s here appears nine times in finest and E and S appears four times in lowest so e and s is that pairing which appears 13 number of times right so uh most common bite pair starting with e is e and s so what we'll

* now be doing is that we'll be going through the data set again uh and we'll be merging these two tokens e and s so now e s will be one token and that's why it's called subword es will be one token so now let me show you my token table again everything else is the same up token number 12 but look at this token number 13 which has been added in token number 13 we have added one more token which is
es because it's the most frequent pairing and Es appears 13 times but remember when we add es we have to subtract something from E and we have to subtract from s because now uh ES has been included so we subtract 13 from the e count so now the the number of time only e appears is three the number of time only s appears is zero this is very interesting to know so the number of time only s appears is zero so s it seems always appears with e so es is a subword see this we would not have discovered if we just did character level tokenization or uh Word level it seems that e and s always so s only

* comes with e in this data set we have already obtained our first site so now this is my new uh this is my new token library and this is my additional token and now we are going to actually continuously keep on doing this process to find uh frequency or to find tokens which appear the most number of times so in the previous iteration we saw that e and s was the bite pair right which occurred most number of times ands it appeared 13 times but now es is a separate token for us so now using that

*  it's always followed by slw which means the word ends after estd so now our algorithm or the tokenizer can differentiate between estimate and highest because in estimate EST does not end with a /w

*  so now if you look at the Tok which we have earlier we had all these 12 tokens but now we created es then we merged it to EST and finally we created this estw so now these two tokens are actually not needed es and EST so now let's look at another other bite pairs which occur a lot so it turns out that o and L is another bite pair which occurs 10 times because it's present in old and older so what we'll do is that we'll create one more we'll merge these two and create one more token for o and L it appears 10 times and we'll subtract that count 10 from the O and L so o individually or with some other character appears 14 times so we subtract 10 because now we have created one more token for o which appears 10 times similarly L appears 14 times overall and we subtract 10 from it because L comes with o 10 times right and now what we do is that o l is one token so now we see that o l and D has appeared 10 times so this bite pair has appear 10 times so we merge this bite pair so then o d now becomes another token which appears 10 times so you see the meaning which our bite pair encoder has captured we have constructed one token which is old we have another token which is estw these tokens are subwords so they are neither full words nor characters they are subwords but they encode the root representation so old is one token and this actually tells us that uh old is the root word which comes in Old it which comes in old as well as older and our BP algorithm has actually captured that perfectly that's awesome right all these root words were not captured by just the word uh word em or

***

*  it does not make sense to merge them into one um one token
because that token does not the frequency of that token appearing in other words is not too much why did we merge EST into one token because it appears in multiple words why why did we merge old in one token because it appears in multiple words so see the rule of uh subord tokenization the first
rule is that that word which occurs multiple times you keep it as it is old
so we kept old as it is right it is a separate token uh but that word which is
not used too many times like older it needs to be split into old and then e and then R are separate tokens similarly
in finest EST is one token fi and N are separate tokens in lowest EST is one token and L O and W
* remove all of those tokens whose frequency is zero
so we can remove s we can remove so let me Mark them with a different color so s
has a frequency of z t has a frequency of Z es EST has frequency of z o has a
* frequency of Z so let's remove this so then our final table looks like these these are the final tokens in our subord tokenizer which is obtained using the bite pair encoding algorithm how did we obtain these tokens we just looked at bite pairs which occur the most then we merged them into one and then we repeated this process uh until we
* obtained until we reached a stage where enough number of tokens have been created or until we reached a stage where our iterations have stopped and this is the final tokenized uh final uh tokens which we'll
be using for next steps of the large language model training which are vector embedding Etc this is how the subword
* tokenizer works and this is exactly how bite pair encoder which is a subword
tokenizer it works for uh training models like gpt2 or gpt3 very few people have this understanding but I hope this lecture has intuitively made it clear for you how the bite pair tokenizer
actually works so now this list of 11 tokens will serve as our vocabulary why
is it called subword tokenizer because these are subwords right EST is not a full word neither it's a character it's
* a subword but it's the root of many words o is also the root of many
words now you must be thinking when do we stop this uh merging when do we stop
these iterations so you have to specify a stopping criteria usually if you look at gpt2 or gpt3 you run this bite par
* coding algorithm on a huge number of tokens right so the stopping criteria
can be if the token count becomes a certain number then you stop or just the number of iterations can be the stop uh
* stopping criteria in too much detail but I just want to
* gpt2 or gpt3 so bite pair
*  you can see that it has about 11,000 stars and about 780 Forks so it's a pretty popular
* subwords uh we are able to get the amazing performance from gpt2 gpt3 or GPT 4 I
* 

the BP tokenizer which was used to train models like gpt2 gpt3 and the original
* sizes context length Etc before we feed the embed or before we feed the

***

