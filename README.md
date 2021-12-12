This repository is **NOT actively maintained**. However, issues and security alerts will be monitored and potentially fixed.
No, this is not directly compatible with HuggingFace transformers(and models based on it, **incl. Kakao GPT-3**). I do NOT provide active support requests with TF-torch model translations.

# **GPT2 Training code**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imcaspar/gpt2-ml/blob/master/pretrained_model_demo.ipynb)
[![GitHub](https://img.shields.io/github/license/ksjae/kogpt2-train)](https://github.com/ksjae/kogpt2-train)
[![GitHub All Releases](https://img.shields.io/github/v/release/ksjae/KoGPT2-train?include_prereleases)](https://github.com/ksjae/kogpt2-train/releases)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/ksjae/KoGPT2-train/issues)
[![GitHub stars](https://img.shields.io/github/stars/ksjae/kogpt2-train?style=social)](https://github.com/iksjae/kogpt2-train)

[**한국어**](./README_KR.md) | [**English**](./README.md)

- [x] THE SCRIPT THAT SUPPORTS TPUS PROPERLY(<10% TPU idle)
- [x] Fast tokenizer powered by HuggingFace/tokenizers
- [ ] Live demo (**currently unavailable**) [#](https://text.ksjit.com)
- [x] 1.5B GPT2 pretrained Korean model ( ~40G corpus )


## Pretrained Model
GPT-2 Small to GPT-2 XL is tested. Not guaranteed to work for larger models.


## Google Colab

[**[Colab Notebook]**](https://colab.research.google.com/drive/1s5zZZL8j2waMTkwUOmSOv6IywoBrNm1z?usp=sharing)


## Train
```
cd KoGPT2-train
export PYTHONPATH=.
python3 train/train_tpu.py --input_file gs://kogpt2/datasets/WEB* --output_dir gs://kogpt2/models/large --max_seq_length 2048 --save_checkpoints_steps 5000 --use_tpu true --tpu_name v3-2 --train_batch_size 16 --config_file configs/large.json --iterations_per_loop 1000 --learning_rate 1e-4
```

## Disclaimer
The contents in this repository are for academic research purpose, and we do not provide any conclusive remarks.
Currently, the underlying model is same as GPT-2. I'm working on the alternating layers.

If you want GPT-2, just change the context token length from 2048 to 1024 and it's practically the same.
Refer to the original paper for specific hyperparameter settings.

## Acknowledgements
This research wouldn't have been possible without the TFRC program and NIPA's HPC Support Program.

## Citation

```
@misc{KoGPT3,
  author = {Seungjae Kim},
  title = {KoGPT3 : Pretrained for Korean},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ksjae/KoGPT}},
}
```

## Reference

Code based on https://github.com/imcaspar/gpt2-ml

https://github.com/google-research/bert

https://github.com/rowanz/grover

Research supported with Cloud TPUs from Google's TensorFlow Research Cloud (TFRC)
