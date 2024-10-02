
<p align="center">
    <br>
    <img src="./picture/logo.jpg" width="600"/>
    <br>
</p>

# Open O1:  A Model Matching Proprietary Power with Open-Source Innovation

[![License](https://img.shields.io/github/license/OpenSource-O1/Open-O1)](https://github.com/wjn1996/Awesome-LLM-Reasoning-Openai-o1-Survey/blob/main/LICENSE)  ![Visitors](https://visitor-badge.lithub.cc/badge?page_id=OpenSource-O1.Open-O1&left_text=Visitors)  ![Stars](https://img.shields.io/github/stars/OpenSource-O1/Open-O1?color=yellow)  ![Forks](https://img.shields.io/github/forks/OpenSource-O1/Open-O1?color=9cf)

## 🌈Mission and Vision for future
Our **Open O1** aims to match the powerful capabilities of proprietary Openai o1 model, empowering the community with advanced open-source alternatives.

As the Open O1 project progresses, we will continue to push the boundaries of what's possible with large language models. Our vision is to create a model that not only achieves o1-like performance but also leads the way in test-time scaling, making advanced AI capabilities accessible to all. Through community-driven development and a commitment to ethical practices, OpenO1 will be a cornerstone in the advancement of AI, ensuring that the future of technology is open and beneficial to all. 

## 🔔News
coming...
## Content Navigation
| Section                                  | Description                                               |
| ------------------------------------- | ------------------------------------------------------------ |
| [💻 Model Deployments & Chat Templates](#Model-Deployments&Chat-templates) | Instructions and examples for deploying models and using chat templates effectively. |
| [✍ Example Demonstrations](#Example-Demonstrations) | Showcase of various use cases and practical demonstrations of the model's capabilities. |
| [💯 System Performance](#System-Performance) | Analysis of system performance metrics, benchmarking results, and optimization strategies. |
| [🎋 Training Details](#Training-Details) | An overview of the training process for Open O1, including datasets used, training methodologies, and any relevant hyperparameters. |
| [❓ FAQ](#FAQ) | Answers to frequently asked questions. |
| [⚠️ Limitations](#Limits) | A discussion of the limitations of the models involved, including known issues, performance constraints, and areas for future improvement. |


## 💻Model Deployment & 💬Chat templates


### 💻Model Deployment


### 💬Chat templates


## ✍Example Demonstrations
todo

## 💯System Performance

The following table provides a comprehensive comparison of the performance between **llama3.1-8b-instruct** and our model across multiple benchmarks. These evaluations were conducted in a **zero-shot setting**, meaning the models were tested without task-specific fine-tuning, highlighting their ability to generalize across diverse tasks. These benchmarks assess various aspects of reasoning, knowledge, and understanding in different domains, offering a clear indication of how each model handles complex tasks without prior exposure or specific task-related training. Our model consistently demonstrates competitive or superior performance, showcasing advancements in areas critical to reasoning, mathematical understanding, and general AI capabilities.

| Model                   | GSM8K| MATH| MMLU| Hellaswag| ARC-C| BBH|
| ----------------------- | :---------------: | :------------: | :--------------: | :-----------: | :-----------: | :-----------: |
| llama3.1-8b-instruct |       84.00       |     47.42     |       67.95      |   **68.43** |   83.87      | 53.64 |
| Ours      |       **85.82**        |      **52.88**     |       **70.45**      |  67.77 |    **86.52**      | **58.43** | 


- **GSM8K**: Our model outperforms **llama3.1-8b-instruct** with a score of **85.82**, demonstrating better reasoning ability in math word problems.
- **MATH**: It's important to note that the official score for **llama3.1-8b-instruct** on MATH is **51.9**, but this was achieved in a CoT (Chain of Thought) setting. In our evaluation, we reproduced the result in a zero-shot setting, where **llama3.1-8b-instruct** scored lower at **47.42**, while our model achieved **52.88**, showing a significant improvement.
- **MMLU**: Our model leads with **70.45**, indicating stronger general knowledge and understanding.
- **Hellaswag**: **llama3.1-8b-instruct** scores **68.43**, slightly ahead of our model at **67.77**.
- **ARC-C**: In ARC-C, our model reaches **86.52**, outperforming **llama3.1-8b-instruct**.
- **BBH**: Our model achieves **58.43**, surpassing **llama3.1-8b-instruct**’s score of **53.64**.

The results highlights our model's superior performance in most benchmarks, with notable improvements in MATH, MMLU, ARC-C, and BBH.

## 🎋Training Details
The training process for Open O1 utilizes the configuration settings from Llama Factory to optimize performance. This section includes details on the datasets used, training methodologies, and relevant hyperparameters.

### model
```
Meta-Llama-3.1-8B-Instruct 
Qwen2.5-7B-Instruct
```

### method
```
stage: sft
do_train: true
finetuning_type: full
deepspeed: ds_z3_config.json
```

### dataset
```
dataset: 4o_response
template: llama3
cutoff_len: 4096
overwrite_cache: true
preprocessing_num_workers: 16
```

### output
```
logging_steps: 10
save_steps: 1000
plot_loss: true
overwrite_output_dir: true
```

### train
```
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
learning_rate: 1.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
```

### eval
```
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 200
```

## 🍭Available Models
- [m-a-p/llama3.1-8b-ins](https://huggingface.co/m-a-p/llama3.1-8b-ins)
- [m-a-p/qwen2.5-7b-ins-v3](https://huggingface.co/m-a-p/qwen2.5-7b-ins-v3)

## ❓FAQ
To Supplement

## ⚠️Limitations
Open O1 is currently in its early stages of development. Open O1 primarily exhibits o1-like reasoning characteristics and broad search thinking capabilities. However, there is still significant progress to be made before it fully achieves O1 capabilities.

## ⭐Star History

[![Star History Chart](https://api.star-history.com/svg?repos=OpenSource-O1/Open-O1&type=Date)](https://star-history.com/#OpenSource-O1/Open-O1&Date)

## Reference

- [Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/) – A detailed blog post by OpenAI discussing methods to enhance reasoning abilities in large language models.
  
- [OpenAI O1 Mini: Advancing Cost-Efficient Reasoning](https://openai.com/index/openai-o1-mini-advancing-cost-efficient-reasoning/) – An OpenAI blog that introduces the O1 Mini model and explores its efficiency in reasoning tasks.

- [Awesome-LLM-Strawberry](https://github.com/hijkzzz/Awesome-LLM-Strawberry) – A curated list of resources and tools related to large language models (LLMs) and reasoning capabilities, including O1.


## Future Todo

| Task                                                                                     | Estimated Schedule  |
|------------------------------------------------------------------------------------------|---------------------|
| Releasing our first version of SFT data that comprises o1-style thinking process          | 1~2 weeks           |
| Reward model (and the corresponding data) for judging the thinking process of each model  | 2~3 weeks           |
| Training infrastructure and pipeline for our o1-style data (both SFT and RLHF)            | 1 month             |
| A new chatbot arena for evaluating and comparing the thinking process of different models | 1 month             |
| Reproducing the two o1 scaling laws both at training time (RLHF) and inference time       | 2~3 months          |



## Citation
If you find our model, data, code useful, welcome to cite our paper
```
@article{
    supplement,
    title={},
    author={OpenO1 Team},
    journal={},
    url={},
    year={}
}
```
## Acknowledgements(Updating)
This repo benefits from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [awesome-o1](https://github.com/hijkzzz/Awesome-LLM-Strawberry). Thanks for their wonderful and solid works.

## Feedback
If you have any questions, please submit them through GitHub Issues.
- Before doing so, we encourage you to review the FAQ section to see if your question has already been addressed, and check previous issues for any relevant discussions.
- Please kindly use our dedicated issue template for submitting. 
- Appreciate your politeness and cooperation in fostering a positive and collaborative community.

