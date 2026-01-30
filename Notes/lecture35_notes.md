## Model initialization with pre-trained weights

***

* 5:00

* Step-1: Load pre-trained GPT-2 weights
* Step-2: Modify the architecture by adding a classification head

* [OpenAI GPT-2 Weights](https://www.kaggle.com/datasets/xhlulu/openai-gpt2-weights)

***

* 10:00

***

* 15:00

***

* 20:00

* Step-3: Select which layers you want to finetune

* Since we already start with a pretrained model, it is not necessary to finetune all layers.
* This is because the lower layers capture basic language structures and sementics, which are applicable across a wide range of tasks and datasets.
* Finetuning last layers is enough.

* We will finetune:
  * Final output head
  * Final transformer block
  * Final LayerNorm module
  * We'll freeze all other parameters

* Step-4: Extract last output token

* 30:00

***
