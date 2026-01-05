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
* stopping criteria can be if the token count becomes a certain number then you stop or just the number of iterations can be the stop uh

***

* https://github.com/openai/tiktoken

***

#### 2.5 BytePair encoding

```python
# pip install tiktoken

import importlib
import tiktoken

print("tiktoken version:", importlib.metadata.version("tiktoken"))
```

```python
tokenizer = tiktoken.get_encoding("gpt2")
```

```python
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

print(integers)

strings = tokenizer.decode(integers)

print(strings)
```

***

#### Next
1. Data Sampling
2. Context Length
3. Batch Sizes

***
