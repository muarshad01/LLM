#### Algorithms
* Word based
* Subword based
* Character based

#### Bite Pair Encoding (BPE)
ChatGPT, GPT-2, GPT-3

***

#### Word-based Tokenization
* __Out of Vocabulary (OOV)__ problem.
* Different meaning of similar words [boy, boys]

#### Character-based tokenizatin
* Character-based tokenizatin solves OOV Problem.
* Howerever, tokenization sequence is very large.

#### Subword-based tokenizatin
* Subword-based tokenization and the bite pair encoding (BPE),  which we are going to see is an example

* __Rule-1__: Don't split frequently used word into smaller subwords.
* __Rule-2__: Split the rare words into smaller, meaningful subwords.

* The subword splitting helps the model learn that different words with the same root word as "token" like "tokens" and "tokenizing" are similar in meaning.
* It also helps the model learn that `tokenization` and `modernization` are made up off different roots but have the same suffix "ization" and are used in same syntactic situations.


#### Byte Pair Encoder (BPE) Algorithm
* Most common pair of consective byptes of data is replaced with a byte that does not occur in data.
  
***

#### Example
* Original data: aaabdaaabac
* 'aa' pair
* Compressed data: ZabdZabac
* 'ab'
* Compressed data: ZYdZYac
* Compressed data: WdWac

***

#### How the BPE algorith is used for LLMs?
* BPE ensures that the most common words in the vocabulary are represented as a single token, while rare words are broken down into two or more subord tokens.
*

* `{"old":7, "older":3, "finest":9, "lowest":4}`
* Perprocessing: We need to add end token `</w>` at the end of each word.
* `{"old</w>":7, "older</w>":3, "finest</w>":9, "lowest</w>":4}`
* old is the common root between old and older
* est is the common root between finest and lowest

* differentiate between estimate and highest because in estimate EST does not end with a /w

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

