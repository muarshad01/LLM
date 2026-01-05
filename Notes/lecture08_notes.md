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

***

* stopping criteria can be if the token count becomes a certain number then you stop or just the number of iterations can be the stop uh

* gpt2 or gpt3 so bite pair
*  you can see that it has about 11,000 stars and about 780 Forks so it's a pretty popular ubwords uh we are able to get the amazing performance from gpt2 gpt3 or GPT 4 I
* 

the BP tokenizer which was used to train models like gpt2 gpt3 and the original
* sizes context length Etc before we feed the embed or before we feed the

***
