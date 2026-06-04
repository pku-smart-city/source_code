# Point-of-Interest Prediction Based on Large Language Models and Intent-Driven World Model

## Code reference
```command
The LLM-IWM used code from Feng S, Lyu H, Li F, et al. Where to move next: Zero-shot generalization of llms for next poi recommendation[C]//2024 ieee conference on artificial intelligence (cai). IEEE, 2024: 1530-1535.. At the same time, LLM-IWM also uses code from the code base https://github.com/LLMMove/LLMMove.
```

## Data

Go to `data` repo and unzip the data.

## Requirements:

- Python=3.9.20
- numpy==1.26.4
- tensorboard==2.18.0
- torch==2.7.1

## Run the code

#### Base run
python main.py -m LLMiwm -d nyc

#### Few-shot
python main.py -m LLMiwm -d nyc --few_shot_days 1