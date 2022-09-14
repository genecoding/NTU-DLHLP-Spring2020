# Homework 2
[Voice Conversion]

## Note
### 2-1
* In solver.py, "pretrain_G" actually trains the autoencoder (encoder+decoder) and "pretrain_D" actually trains the classifier1, of the stage 1 in 
  [the paper](#Reference); "train" trains the stage 1 and "patchGAN" trains the stage 2. The classifier1 is necessary to separate the speaker characteristics from the linguistic content in speech signals.
* hint: 大約跑 10000 個 steps 即可有不錯的結果。
* 請嘗試轉成p1 和 p2 interpolation 的聲音
  * \# add for interpolation of p1 & p2, in solver.py & convert.py
### 2-2
* 請把程式改成只轉兩個 speaker 的 code
  * \# modified for 2 speakers, in model.py
### 2-3
* Option 1. 自己找一個不是 StarGAN-VC，也不是 HW2-1 的 model，實際 train 看看。
  * https://github.com/jjery2243542/adaptive_voice_conversion

## Reference
[Multi-target Voice Conversion without Parallel Data by Adversarially Learning Disentangled Audio Representations][p1], J Chou *et al.*



[Voice Conversion]: https://docs.google.com/presentation/d/1lKdhQaYQO4elmXrEoi3d8SDb1TzwMNcOBr9_oMXvPKA
[p1]: https://arxiv.org/abs/1804.02812
