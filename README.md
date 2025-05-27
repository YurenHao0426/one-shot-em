## One-shot Entropy Minimization

<a href='https://arxiv.org/abs/2505.20282'><img src='https://img.shields.io/badge/arXiv-2505.20282-b31b1b.svg'></a> &nbsp;

### Reproducing One-shot EM Training (SOTA)

```bash
accelerate launch train.py --lr 2e-5 --temperature 0.5 --bsz 64
```


### Reproducing Multi-shot EM Training

```bash
accelerate launch train.py --lr 2e-5 --temperature 0.5 --bsz 64 --data_path "dataset/numina/numina_00.parquet"
```

### Evaluation

```bash
cd Qwen2.5-Eval/evaluation
bash sh/eval_all_math.sh
```

### Acknowledgements

Our dataset references and builds upon the following open-source contributions:

- [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)
- [DeepScaler](https://github.com/agentica-project/deepscaler)
- [One-shot RLVR](https://github.com/ypwang61/One-Shot-RLVR/) – for data selection strategies
- [Qwen2.5-Eval](https://github.com/QwenLM/Qwen2.5-Math/) – for evaluation benchmarks

We sincerely thank the authors and maintainers of these projects for their excellent contributions to the research community!


### Citation
```
@misc{gao2025oneshotentropyminimization,
      title={One-shot Entropy Minimization}, 
      author={Zitian Gao and Lynx Chen and Joey Zhou and Bryan Dai},
      year={2025},
      eprint={2505.20282},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.20282}, 
}
```