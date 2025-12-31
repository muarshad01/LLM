#### Book
* [Book: Build a Large Language Model (From Scratch) by Sebastian Raschka - Amazon](https://www.amazon.com/Build-Large-Language-Model-Scratch/dp/1633437167/ref=sr_1_1?adgrpid=186568885419&dib=eyJ2IjoiMSJ9.4aksU58BPWhOPrzkY0x_-vygt-ueYbeZO3NZCcItOJhwUkUI29dKBu191StjAX_6kFIlMZDQaXqiikWYV4yCxJyCN-1fORiR8HXdYk1pnOy5pMuFLCLhD5oeLTs9Ckz_Gnoe1ZnkxkuYsSetETUZjciRj-PrGLoJVEoGdjR8eqwYDRrYz3VpxBUSY6KP-X8vtpcM45C3qypNE1aMCVFd5L0U8ZmbFp0Tl28wzFU-cxg.P3ksogW4khJ39syyd4KIpkFJva8xg6NeQUZtKnKBgbM&dib_tag=se&hvadid=779542975090&hvdev=c&hvexpln=0&hvlocphy=9008257&hvnetw=g&hvocijid=12364010510054860427--&hvqmt=b&hvrand=12364010510054860427&hvtargid=kwd-2192499717731&hydadcr=16405_13751106_11567&keywords=build+an+llm+from+scratch&mcid=c7100230a49531d7af7d4236df7c0991&qid=1767193721&sr=8-1)

* [GitHub Link for Book - Code](https://github.com/rasbt/LLMs-from-scratch)

```unix
$ python --version
$ python3 --version
Python 3.11.13

$ brew install python@3.x.y
$ pip install uv
$ pip3.11 install uv

$ cd ~/Desktop
$ git clone https://github.com/rasbt/LLMs-from-scratch.git
$ cd ~/Desktop/LLMs-from-scratch

$ uv venv --python=python3.11
$ source .venv/bin/activate

$ which python
/Users/marshad/Desktop/LLMs-from-scratch/.venv/bin/python

$ python --version
Python 3.11.13

$ uv pip install packages

$ uv run jupyter lab
```

***

#### Dr. Raj Dandekar
| Lecture | Notes | Date Updated |
|---|---|---|
| [Lecture 01 -- Building LLMs from scratch: Series introduction](https://www.youtube.com/watch?v=Xpr8D6LeAtw) |  [Notes01](https://github.com/muarshad01/LLM/blob/main/Notes/lecture01_notes.md)| Dec 31, 2025|
| [Lecture 02 -- LLM Basics](https://www.youtube.com/watch?v=3dWzNZXA8DY)| [Notes02](https://github.com/muarshad01/LLM/blob/main/Notes/lecture02_notes.md) | Dec 31, 2025 |
| [Lecture 03 -- Pretraining LLMs vs Finetuning LLMs](https://www.youtube.com/watch?v=-bsa3fCNGg4)|  [Notes03](https://github.com/muarshad01/LLM/blob/main/Notes/lecture03_notes.md) | Dec 31, 2025 |
| [Lecture 04 -- What are transformers?](https://www.youtube.com/watch?v=NLn4eetGmf8) |  [Notes04](https://github.com/muarshad01/LLM/blob/main/Notes/lecture04_notes.md) | Dec 31, 2025|
| [Lecture 05 -- How does GPT-3 really work?](https://www.youtube.com/watch?v=xbaYCf2FHSY) |  [Notes05](https://github.com/muarshad01/LLM/blob/main/Notes/lecture05_notes.md) | Dec 31, 2025 |
| [Lecture 06 -- Stages of building an LLM from Scratch](https://www.youtube.com/watch?v=z9fgKz1Drlc) |  [Notes06](https://github.com/muarshad01/LLM/blob/main/Notes/lecture06_notes.md) | Dec 31, 2025 |
| [Lecture 07 -- Code an LLM Tokenizer from Scratch in Python](https://www.youtube.com/watch?v=rsy5Ragmso8) |  [Notes07](https://github.com/muarshad01/LLM/blob/main/Notes/lecture07_notes.md) | Aug xx, 2025|
| [Lecture 08 -- The GPT Tokenizer: Byte Pair Encoding](https://www.youtube.com/watch?v=fKd8s29e-l4) |  [Notes08](https://github.com/muarshad01/LLM/blob/main/Notes/lecture08_notes.md) | Aug xx, 2025|
| [Lecture 09 -- Creating Input-Target data pairs using Python DataLoader](https://www.youtube.com/watch?v=iQZFH8dr2yI) |  [Notes09](https://github.com/muarshad01/LLM/blob/main/Notes/lecture09_notes.md) | |
| [Lecture 10 -- What are token embeddings?](https://www.youtube.com/watch?v=ghCSGRgVB_o) |  [Notes10](https://github.com/muarshad01/LLM/blob/main/Notes/lecture10_notes.md) | |
| [Lecture 11 -- The importance of Positional Embeddings](https://www.youtube.com/watch?v=ufrPLpKnapU) |  [Notes11](https://github.com/muarshad01/LLM/blob/main/Notes/lecture11_notes.md) | |
| [Lecture 12 -- The entire Data Preprocessing Pipeline of LLMs](https://www.youtube.com/watch?v=mk-6cFebjis) |  [Notes12](https://github.com/muarshad01/LLM/blob/main/Notes/lecture12_notes.md)| |
| [Lecture 13 -- Introduction to the Attention Mechanism in LLMs](https://www.youtube.com/watch?v=XN7sevVxyUM) |  [Notes13](https://github.com/muarshad01/LLM/blob/main/Notes/lecture13_notes.md) | |
| [Lecture 14 -- Simplified Attention Mechanism - Coded from scratch in Python - No trainable weights)](https://www.youtube.com/watch?v=eSRhpYLerw4) |  [Notes14](https://github.com/muarshad01/LLM/blob/main/Notes/lecture14_notes.md) | |
| [Lecture 15 -- Coding the self attention mechanism with key, query and value matrices](https://www.youtube.com/watch?v=UjdRN80c6p8) |  [Notes15](https://github.com/muarshad01/LLM/blob/main/Notes/lecture15_notes.md) | |
| [Lecture 16 -- Causal Self Attention Mechanism - Coded from scratch in Python](https://www.youtube.com/watch?v=h94TQOK7NRA) |  [Notes16](https://github.com/muarshad01/LLM/blob/main/Notes/lecture16_notes.md) | |
| [Lecture 17 -- Multi Head Attention Part 1 - Basics and Python code](https://www.youtube.com/watch?v=cPaBCoNdCtE) |  [Notes17](https://github.com/muarshad01/LLM/blob/main/Notes/lecture17_notes.md) | |
| [Lecture 18 -- Multi Head Attention Part 2 - Entire mathematics explained](https://www.youtube.com/watch?v=K5u9eEaoxFg) |  [Notes18](https://github.com/muarshad01/LLM/blob/main/Notes/lecture18_notes.md) | |
| [Lecture 19 -- Birds Eye View of the LLM Architecture](https://www.youtube.com/watch?v=4i23dYoXp-A) |  [Notes19](https://github.com/muarshad01/LLM/blob/main/Notes/lecture19_notes.md) | |
| [Lecture 20 -- Layer Normalization in the LLM Architecture](https://www.youtube.com/watch?v=G3W-LT79LSI) |  [Notes20](https://github.com/muarshad01/LLM/blob/main/Notes/lecture20_notes.md) ||

***

#### [Transformers (how LLMs work) explained visually | DL5](https://www.youtube.com/watch?v=wjZofJX0v4M)

***

## Deep Learning
|Lecture | Notes|
|---|---|
| [Deep Learning Chapter 1 -- But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) ||
| [Deep Learning Chapter 2 -- Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) ||
| [Deep Learning Chapter 3 -- Backpropagation, intuitively](https://www.youtube.com/watch?v=Ilg3gGewQ5U) ||
| [Deep Learning Chapter 4 -- Backpropagation calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8) ||
| [Deep Learning Chapter 5 -- Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M) ||
| [Deep Learning Chapter 6 -- Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc) ||
| [Deep Learning Chapter 7 -- How might LLMs store facts ](https://www.youtube.com/watch?v=9-Jl0dxWQs8) ||

***

* [AGI Lambda](https://www.youtube.com/@AGI.Lambdaa/shorts)
* [Vision Transformers](https://www.youtube.com/shorts/qPUYBX0C6ic)
* [Self-Attention in Transformer](https://www.youtube.com/shorts/l8_OrR9kUNw)
* [How RNNs Help AI Understand Language](https://www.youtube.com/shorts/w67EHFHGHUQ)
* [How does NN work in 60 seconds](https://www.youtube.com/shorts/Dbcx2_MO0LM)
* [BERT Networks in 60 seconds](https://www.youtube.com/shorts/HBOloY08auQ)
* [What is RAG](https://www.youtube.com/shorts/CbAQUqnrDcA)
* [Build a Small Language Model (SLM) From Scratch](https://www.youtube.com/watch?v=pOFcwcwtv3k)
* [MCP Protocol](https://www.youtube.com/shorts/7CHr0qwTcJw)
* [Large Language Models explained briefly](https://www.youtube.com/watch?v=LPZh9BOjkQs&t=2s)
* Autoencoders | Deep Learning Animated
* [The Most Important Algorithm in Machine Learning](https://www.youtube.com/watch?v=SmZmBKc7Lrs)
* [How word vectors encode meaning](https://www.youtube.com/shorts/FJtFZwbvkI4)
* [Visualizing transformers and attention | Talk for TNG Big Tech Day '24](https://www.youtube.com/watch?v=KJtZARuO3JY)
* [ML Foundations for AI Engineers (in 34 Minutes)](https://www.youtube.com/watch?v=BUTjcAjfMgY)
* [How DeepSeek Rewrote the Transformer [MLA]](https://www.youtube.com/watch?v=0VLAoVGf_74)
* [I Visualised Attention in Transformers](https://www.youtube.com/watch?v=RNF0FvRjGZk)
* [What is Tensor](https://www.youtube.com/shorts/J4Tg4gAPMMQ)

***

* [Open Source LLM Tools - Chip Huyen](https://huyenchip.com/llama-police.html)

*
***

## TODO
1. LLM from scratch playlist
2. Deepseek playlist 
3. Reinforcement playlist
