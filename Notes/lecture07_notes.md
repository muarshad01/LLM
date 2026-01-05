## Tokenization 
* Data preparation and sampling
* Pre-training 
* We will build a Tokenizer fully from scratch.
* We will also build an Encoder-and-Decoder from scratch.

## Stages

#### Stage-1
1. Data preparation and sampling
2. Attention mechanism
3. understanding the LLM architecture.  

#### Stage-2
* involves pre-training and building the foundational model so that involves the training Loop, model evaluation, and loading pre-trained weights.

#### Stage-3
* involves fine-tuning, training on smaller very specific datasets.

* LLMs are just NN.
*  right so you need data the parameters of the LLM are optimized and then we have some output.

***

## Tokenization Steps
1. Split the input text into individual word and subword tokens
2. Convert these tokens into token IDs
3. Encode these token IDs into Vector representation

***

#### 1. Split the input text into individual word and subword tokens

```python
with open("the-verdict.txt", "r", encoding="utf-8") as f:
   raw_text = f.read()
    
print("Total number of character:", len(raw_text))
print(raw_text[:99])
```

```python
import re

text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)

print(result)
```

```python
result = re.split(r'([,.]|\s)', text)

print(result)
```

```python
# Strip whitespace from each item and then filter out any empty strings.
result = [item for item in result if item.strip()]
print(result)
```

```python
text = "Hello, world. Is this-- a test?"

result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)
```

```python
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:30])
```

* Next lecture it's called bite-pair encoding

```python
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:30])
```

***

#### 2. Convert these tokens into token IDs

1. Vocabulary is constructed, which is just a list-of-tokens sorted in alphabetical-order.
2. to each token you assign a unique integer, which is called as the token ID.

```python
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

print(vocab_size)
```

```python
vocab = {token:integer for integer,token in enumerate(all_words)}
```

```python
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break
```

* We can condiser this process as Encoder.

```python
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
                                
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
```


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

#### [45--50]

* tokens act as markers signaling the start-or-end of a particular

#### [50-55]

*** 

#### [55-60]

***

* token IDs also into to Vector representations which we will come to later but in today's lecture we mostly
* called __bite pair encoding__ in bite pair en en in bite
1:07:57
pair encoding every word is not a token words themselves are broken down into subwords and then these subwords are the token so let's say you have a word which is uh which is chased
*  chased itself is one token but in bite pair encoding it might
* 
*** 








