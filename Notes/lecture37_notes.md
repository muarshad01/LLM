* Pretrained LLMs are good at text completion
* They struggle with following instructions:
  * Fix the grammer in this text
  * convert this text into passive voice

***

* 5:00

* Training on a dataset where the (input, output) pairs are explicitly provided. It is also called __"supervidsed instruction finetuning".__

***

* 10:00

***

* 15:00


```python
import json
import os
import urllib


def download_and_load_file(file_path, url):

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    # The book originally contained this unnecessary "else" clause:
    #else:
    #    with open(file_path, "r", encoding="utf-8") as file:
    #        text_data = file.read()

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


file_path = "instruction-data.json"
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)

data = download_and_load_file(file_path, url)
print("Number of entries:", len(data))
```

***

#### Alpaca Prompt format

***
