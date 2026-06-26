# Tesi triennale - GAN per audio super resolution

A una rete generatrice (codice di un altro tesista) è stato aggiunto un
discriminatore multi-scala per costruire una GAN che ricostruisce il contenuto in
alta frequenza di tracce audio, operando sulla STFT (dataset FMA-small).

## Struttura

```
src/
  config.py         # path e iperparametri centralizzati
  preprocessing.py   # download/split + estrazione forme d'onda e STFT  (CORRETTO)
  dataset.py         # AudioSTFTDataset + utility                       (altro tesista, INTATTO)
  generator.py       # ReteRecWithAttention                             (altro tesista, INTATTO)
  discriminator.py   # MultiScaleDiscriminator                          (CORRETTO)
  losses.py          # CombinedLoss / DiscriminatorLoss / feature match (CORRETTO)
  train_gan.py       # ciclo di addestramento GAN                       (CORRETTO)
notebooks/
  01_preprocessing.ipynb   # prepara i dati
  02_train_gan.ipynb       # addestra la GAN
tests/
  smoke_test.py      # verifica della logica su CPU con tensori random
REVIEW.md            # report dei bug corretti e di quelli segnalati nel generatore
notebook_originale.ipynb   # notebook di partenza (output puliti, per riferimento)
```

## Esecuzione

Pensato per Kaggle (path di default `/kaggle/...`). In locale, impostare i path
via variabili d'ambiente (`DATA_ROOT`, `WORKING_DIR`, `CHECKPOINT_DIR`).

```bash
pip install -r requirements.txt
# 1) eseguire notebooks/01_preprocessing.ipynb  (prepara dati)
# 2) eseguire notebooks/02_train_gan.ipynb       (addestra la GAN)
```

## Test

```bash
python -m py_compile src/*.py tests/smoke_test.py
python tests/smoke_test.py   # richiede torch
```

Vedi `REVIEW.md` per il dettaglio delle correzioni.
