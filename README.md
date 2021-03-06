# A TensorFlow Implementation of DCTTS
 DCTTS is introduced in [Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention](https://arxiv.org/abs/1710.08969).

## Requirements
  * NumPy >= 1.11.1
  * TensorFlow >= 1.3 (Note that the API of `tf.contrib.layers.layer_norm` has changed since 1.3)
  * librosa
  * tqdm
  * matplotlib
  * scipy

## Data

I train Portuguese models with <p> 1. [TTS-Portuguese Corpus](https://github.com/Edresson/TTS-Portuguese-Corpus) <br/>

## Training
  * STEP 0. Download [TTS-Portuguese Corpus](https://github.com/Edresson/TTS-Portuguese-Corpus) or prepare your own data.
  * STEP 1. Run `python prepro.py`.
  * STEP 2. Run `python train.py 1` for training Text2Mel.
  * STEP 3. Run `python train.py 2` for training SSRN.

You can do STEP 2 and 3 at the same time, if you have more than one gpu card.

## Training Curves



## Attention Plot

## Sample Synthesis
I generate speech samples based on  [phonetically balanced sentences](https://repositorio.ufsc.br/bitstream/handle/123456789/112119/98594.pdf?sequence=1) as the original paper does. It is already included in the repo.

  * Run `synthesize.py` and check the files in `samples`.

## Generated Samples

| Dataset       | Samples |
| :-------------|
| TTS-Portuguese Corpus with Text |[2115k](https://soundcloud.com/user-797601460/sets/dctts-phoneme-grinffin-lim)|

| TTS-Portuguese Corpus with Phoneme |[1734k](https://soundcloud.com/user-797601460/sets/dctts-phoneme-grinffin-lim)|


## Online TTS demo

A notebook supposed to be executed on https://colab.research.google.com is available:

- [DCTTS Text demo](https://colab.research.google.com/drive/1GwC1hp-gbuNC-_fk3Bm7k2kj6kG6SRsz


## Pretrained Model for Models
TTS-Portuguese Corpus with Text Download [this]().

TTS-Portuguese Corpus with Phoneme Download [this]().

## Notes
   *  The changes not described in the paper were inspired by the repository: [dc_tts](https://github.com/kyubyong/dc_tts)
  * The paper didn't mention normalization, but without normalization I couldn't get it to work. So I added layer normalization.
    * The paper didn't mention dropouts. So I added 0.05 for all layers.
  * The paper fixed the learning rate to 0.001, but it didn't work for me. So I decayed it.

  * This implementation is inspired by the repository: [dc_tts](https://github.com/kyubyong/dc_tts)
