<div align= "center">
    <h1> Answer-Side Backdoor</h1>
</div>

<p align="center">
The Model Plants the Trigger: Answer-Side Backdoor Attacks in Multi-Turn Large Language Models
</p>
<p align="center">
  <a href="https://arxiv.org/abs/2610.07723"><img src="https://img.shields.io/badge/arXiv-2610.07723-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <img src="https://img.shields.io/badge/EMNLP%202026%20Findings-Accepted-blueviolet" alt="EMNLP 2026 Findings">
  <a href="https://github.com/Yibo124/answer-side-backdoor/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
</p>
## Repository Structure

```text
.
├── train/                  # Training sets JSON files
├── test/
│   ├── configs/            # Evaluation YAML configurations
│   ├── data/               # Test sets JSON files
│   ├── src/
│   │   ├── test.py         # Evaluation, API judging, and metrics
│   │   └── inference.py    # Model loading, templates, and generation
│   ├── saves/              # Place the evaluated LoRA adapters here
│   └── result/             # Experiment results
└── README.md
```

This repo contains the training sets used in the paper in `train/` and test sets and testing program in `test/`. the Quickstart below will guide you to setup and run the testing program to evaluate the backdoored models.

## Quickstart

### 1. Set up the environment

Copy and run the following command in the repo root to setup the conda environment:

```bash
conda create -n answer-side-backdoor python=3.12 -y
conda activate answer-side-backdoor
pip install -r requirements.txt
```

Create a `.env` file in the repo root to configure the judge API for `trigger-clean`:

```dotenv
OPENAI_BASE_URL=https://your-provider.example/v1
OPENAI_API_KEY=sk-your-api-key
OPENAI_MODEL=gpt-4o
```

### 2. Configure the model

Place your LoRA adapters in `test/saves/<adapter_name>` and configure them in `test/configs/<config_name>.yaml`. All LoRA adapters in our paper were trained using [LLaMA Factory](https://github.com/hiyouga/LlamaFactory).

Example configuration (`test/configs/dormant_mistral_0.05.yaml`):

```yaml
model_name_or_path: mistralai/Mistral-7B-Instruct-v0.3    # Hugging Face ID or local model path
adapter_name_or_path: dormant_mistral_0.05    # Name under test/saves/<adapter_name> or local adapter path, optional if you evalute base model
template: mistral    # Follow the naming convention in LLaMAFactory, which is llama2 mistral gemma deepseekr1 for each model evaluated in our paper
test_name: dormant_mistral_0.05    # Output Directory test/result/<test_name>
max_new_tokens: 2048
batch_size: 4
```

### 3. Run the evaluation

```bash
python test/src/test.py --config test/configs/dormant_mistral_0.05.yaml
```

Results are saved to `test/result/<test_name>/`:

- `summary.json`: `TIR`, `ASR`, `ACC_th`, `ACC_tc`, `ACC_ch`, `TIR_th`, and `TIR_tc`.
- `logs/*.result.json`: per-split metrics, conversations, and judge labels.

## Citation

If you find our work helpful, feel free to give us a cite.

```bibtex
@misc{zhang2026modelplantstriggeranswerside,
      title={The Model Plants the Trigger: Answer-Side Backdoor Attacks in Multi-Turn Large Language Models}, 
      author={Yibo Zhang and Tianrong Guan and Liang Lin and Puze Wang and Jin Wang and Qingsong Wen},
      year={2026},
      eprint={2610.07723},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2610.07723}, 
}
```
