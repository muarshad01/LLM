## Evaluating the LLM (Whole Field!!!)

* How to measure LLM Performance?

* Extracting and Saving Responses

***

* 10:00

In practice, instruction-finetuned LLMs such as chatbots are evaluated via multiple approaches:
1. Short-answer and multiple choice benchmarks such as MMLU [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300), which test the general knowldge of a model.
2. Human preference comparison to other LLMs, such as LMSYS chatbot arena -  [LMSYS Org](https://lmsys.org/)
3. Automated conversational benchmarks, where another LLM like GPT-4 is used to evaluate the responses, such as [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/) completes the request.

***

* 20:00

#### Evaluating the fine-tuned LLM

* [Ollama](https://ollama.com/)

* [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)

```
$ ollama server (only on Windows)
$ ollama run llam3
```

***

* 45:00

* [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

***
