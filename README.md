# A TensorFlow Implementation of DC-TTS: yet another text-to-speech model

I implement yet another text-to-speech model, dc-tts, introduced in [Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention](https://arxiv.org/abs/1710.08969).

## Requirements
  * NumPy >= 1.11.1
  * TensorFlow >= 1.3 (Note that the API of `tf.contrib.layers.layer_norm` has changed since 1.3)
  * librosa
  * tqdm
  * matplotlib
  * scipy

## Data

I train Portuguese models with <p> 1. [TTS-Portuguese Corpus]() <br/>

LJ Speech Dataset is recently widely used as a benchmark dataset in the TTS task because it is publicly available, and it has 24 hours of reasonable quality samples.
Nick's and Kate's audiobooks are additionally used to see if the model can learn even with less data, variable speech samples. They are 18 hours and 5 hours long, respectively. Finally, KSS Dataset is a Korean single speaker speech dataset that lasts more than 12 hours.


## Training
  * STEP 0. Download [TTS-Portuguese Corpus](https://keithito.com/LJ-Speech-Dataset/) or prepare your own data.
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

## Pretrained Model for Models
TTS-Portuguese Corpus with Text Download [this]().

TTS-Portuguese Corpus with Phoneme Download [this]().

## Notes
   *  The changes not described in the paper were inspired by the repository: [dc_tts](https://github.com/kyubyong/dc_tts)
  * The paper didn't mention normalization, but without normalization I couldn't get it to work. So I added layer normalization.
    * The paper didn't mention dropouts. So I added 0.05 for all layers.
  * The paper fixed the learning rate to 0.001, but it didn't work for me. So I decayed it.

  * This implementation is inspired by the repository: [dc_tts](https://github.com/kyubyong/dc_tts)
