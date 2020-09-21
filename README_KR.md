# **GPT3 훈련코드**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imcaspar/gpt2-ml/blob/master/pretrained_model_demo.ipynb)
[![GitHub](https://img.shields.io/github/license/ksjae/kogpt2-train)](https://github.com/ksjae/kogpt2-train)
[![GitHub All Releases](https://img.shields.io/github/v/release/ksjae/KoGPT2-train?include_prereleases)](https://github.com/ksjae/kogpt2-train/releases)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/ksjae/KoGPT2-train/issues)
[![GitHub stars](https://img.shields.io/github/stars/ksjae/kogpt2-train?style=social)](https://github.com/iksjae/kogpt2-train)

[**한국어**](./README_KR.md) | [**English**](./README.md)

- [x] TPU를 제대로 활용하는 몇 안되는 script(<10% TPU idle)
- [x] HuggingFace/tokenizers로 더 빠른 Dataset 전처리
- [x] 데모("아무말 대잔치") [#](https://text.ksjit.com)
- [x] 27억개 이상의 Parameter 모델 다운로드 ( ~40G corpus ) [#](https://github.com/ksjae/KoGPT)


## 훈련된 모델
[KoGPT 모델 카드](https://github.com/ksjae/KoGPT)의 Releases 페이지에서 다운로드 가능합니다.
*이 코드로 130+억 파라미터 버전을 훈련하는 경우 정상 작동을 보증하지 않습니다


## Google Colab

[**[Colab Notebook]**](https://colab.research.google.com/drive/1s5zZZL8j2waMTkwUOmSOv6IywoBrNm1z?usp=sharing)


## 직접 훈련시키기
```
cd KoGPT2-train
export PYTHONPATH=.
python3 train/train_tpu.py --input_file gs://kogpt2/datasets/WEB* --output_dir gs://kogpt2/models/large --max_seq_length 2048 --save_checkpoints_steps 5000 --use_tpu true --tpu_name v3-2 --train_batch_size 16 --config_file configs/large.json --iterations_per_loop 1000 --learning_rate 1e-4
```

## 경고
The contents in this repository are for academic research purpose, and we do not provide any conclusive remarks.

## 인용

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