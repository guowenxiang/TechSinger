# TechSinger: Technique Controllable Multilingual Singing Voice Synthesis via Flow Matching

####  Wenxiang Guo, Yu Zhang, Changhao Pan, Rongjie Huang, Li Tang, Ruiqi Li, Zhiqing Hong, Yongqi Wang, Zhou Zhao | Zhejiang University

PyTorch Implementation of TechSinger: Technique Controllable Multilingual Singing Voice Synthesis via Flow Matching.

We provide our implementation codes in this repository.

Also, you can visit our [Demo Page](https://tech-singer.github.io/) to see the results of the generated singing audios.                                              |

### Dependencies

A suitable [conda](https://conda.io/) environment named `stylesinger` can be created
and activated with:

```
conda create -n techsinger python=3.8
conda install --yes --file requirements.txt
conda activate techsinger
```

## Train your own model based on GTSinger

### Data Preparation 

1. Prepare your own singing dataset or download [GTSinger](https://github.com/GTSinger/GTSinger).
2. Put `metadata.json` (including ph, word, item_name, ph_durs, wav_fn, singer, ep_pitches, ep_notedurs, ep_types for each singing voice) and `phone_set.json` (all phonemes of your dictionary) in `data/processed/style` **(Note: we provide `metadata.json` and `phone_set.json` in GTSinger, but you need to change the wav_fn of each wav in `metadata.json` to your own absolute path)**.
3. Set `processed_data_dir` (`data/processed/style`), `binary_data_dir`,`valid_prefixes` (list of parts of item names, like `["Chinese#ZH-Alto-1#Mixed_Voice_and_Falsetto#一次就好"]`), `test_prefixes` in the [config](./egs/techsinger.yaml).
4. Preprocess Dataset: 

```bash
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=$GPU python data_gen/tts/bin/binarize.py --config egs/techsinger.yaml
```

### Training StyleSinger

```bash
CUDA_VISIBLE_DEVICES=$GPU python tasks/run.py --config egs/techsinger.yaml  --exp_name TechSinger --reset
```

### Inference using StyleSinger

```bash
CUDA_VISIBLE_DEVICES=$GPU python tasks/run.py --config egs/techsinger.yaml  --exp_name TechSinger --infer
```

## Acknowledgements

This implementation uses parts of the code from the following Github repos:
[GenerSpeech](https://github.com/Rongjiehuang/GenerSpeech),
[NATSpeech](https://github.com/NATSpeech/NATSpeech),
[ProDiff](https://github.com/Rongjiehuang/ProDiff),
[DiffSinger](https://github.com/MoonInTheRiver/DiffSinger)
as described in our code.
