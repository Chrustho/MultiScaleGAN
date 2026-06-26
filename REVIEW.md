# Revisione del codice — report

Revisione del notebook originale `notebook8324482692(2).ipynb` (GAN per
super-resolution audio su STFT). Sono stati corretti gli errori nella **parte
iniziale** (download + preprocessing) e **dal discriminatore in poi**
(discriminatore, loss, training GAN). Il **generatore** ("Codice altro tesista",
celle 22–30) **non è stato modificato**: i suoi problemi sono solo *segnalati*.

Il notebook è stato spezzato in moduli `src/*.py` + due notebook sottili in
`notebooks/`, con gli output di debug rimossi (il file passa da 296 KB a 78 KB).

---

## 1. Bug corretti — parte iniziale (`src/preprocessing.py`, `src/config.py`)

| # | Problema (cella originale) | Correzione |
|---|----------------------------|------------|
| 1 | **Accoppiamento wave↔STFT errato** (cella 25): `audio_files`/`stft_files` da `os.listdir` (ordine non garantito) accoppiati per indice → coppie audio/STFT potenzialmente disallineate. | `paired_wave_stft_paths()` interseziona i nomi file e li ordina: accoppiamento per nome, sempre coerente. |
| 2 | **`save_on_drive` senza `else`** (cella 13): split sconosciuto → `path` indefinito → `NameError`. | Path risolto da `config.split_dir()` che solleva `ValueError` esplicito. |
| 3 | **Finestra STFT incoerente**: GT salvata con `hamming` (cella 21), `stft_train` ricalcolata con `hann` (cella 23). | Preprocessing allineata a `hann` (= finestra del dataset), scelta centralizzata in `config.WINDOW`. |
| 4 | **Path hardcoded e incoerenti**: mix di `/kaggle/working`, `/kaggle/temp`, `/content/train_v2` (residuo Colab), `../temp`. | Tutti i path in `config.py`, sovrascrivibili via variabili d'ambiente. Rimosso il path `/content` errato. |
| 5 | **Codice morto / rumore**: `Load_Dataset` (cella 17) mai usata e con naming incompatibile con `save_on_drive`; `COLS`/`names` inutilizzati; `print` di debug massivi (celle 19, 21, ~200 KB di output); shuffle senza seed. | `Load_Dataset` rimossa; variabili inutili eliminate; debug rimosso; shuffle con `config.SEED`. |

## 2. Bug corretti — discriminatore (`src/discriminator.py`)

| # | Problema | Correzione |
|---|----------|------------|
| 1 | **BatchNorm + WGAN-GP incompatibili**: la `BatchNorm2d` accoppia i campioni del batch, mentre la gradient penalty assume indipendenza per-campione → penalità mal definita. | Sostituita con `InstanceNorm2d(affine=True)` (normalizzazione per-campione). |
| 2 | **`final_conv` senza spectral_norm** mentre tutti gli altri layer lo usano. | `spectral_norm` applicata anche a `final_conv` (uniformità del vincolo di Lipschitz). |
| 3 | **Parametro `freq_bins` inutilizzato** in `SubDiscriminator` (la rete è completamente convoluzionale). | Rimosso dalla firma. |

> Nota di design: con `spectral_norm` **e** gradient penalty si impone il vincolo
> di Lipschitz due volte. Si è scelto di mantenerli entrambi (come nell'originale)
> per non alterare il comportamento; in fase di tuning se ne può rimuovere uno.

## 3. Bug corretti — loss e training GAN (`src/losses.py`, `src/train_gan.py`)

| # | Problema (riga originale, cella 34) | Correzione |
|---|--------------------------------------|------------|
| 1 | **`scaler_g`/`scaler_d` usati prima di essere definiti** (caricamento checkpoint ~373–384, definiti solo a ~433) → `NameError` se il checkpoint contiene quelle chiavi. | Lo scaler è creato **prima** del caricamento del checkpoint. |
| 2 | **Feature matching loss sommata alla loss di D** (`DiscriminatorLoss.forward`): è invece un obiettivo del **generatore**. | `feature_matching_loss` spostata nel `generator_step`; `DiscriminatorLoss` calcola solo loss avversaria + GP. |
| 3 | **Gradient penalty sotto AMP/GradScaler**: `autograd.grad(create_graph=True)` su loss scalata e dentro `autocast` → gradienti errati. | Il discriminatore è addestrato in **fp32** (niente autocast/scaler): il double-backward della GP è così corretto. |
| 4 | **`del fake_stft_g`** (~656) quando `use_discriminator=False` → `NameError`. | Variabile gestita solo dove definita; nessuna `del` non protetta. |
| 5 | **`device == 'cuda'`** (~666): confronto stringa vs `torch.device`, sempre falso. | Usato `device.type == 'cuda'`. |
| 6 | **Logica `d_steps` + `accumulation_steps` muddled**: `opt_d.zero_grad()` ad ogni `_d_step` azzerava i gradienti; `clip_grad_norm_` su gradienti ancora scalati. | Step di D lineare e coerente (`zero_grad → backward → clip → step`); accumulazione tenuta solo per G. |
| 7 | **Validation con `.permute` incondizionato** mentre il training era condizionale sul formato → fragile. | Conversione unica `_to_generator_input` usata sia in training che in validation. |
| 8 | **Codice morto/duplicato**: `history` definita due volte; rami `if/else` BFT/BTF identici; `CombinedLoss`/`SpectralCoherenceLoss` ridefinite (anche con `print` ad ogni forward). | `history` unica; assunto formato BFT con controllo esplicito; loss in un'unica `src/losses.py` senza `print`. |

## 4. Problemi SEGNALATI (generatore — NON modificato)

Codice dell'altro tesista, copiato verbatim in `src/dataset.py` e `src/generator.py`:

- **`ReteRecWithAttention`** (cella 24): il nome promette un meccanismo di
  *attention* che **non esiste** — è un encoder–decoder LSTM. Inoltre dipende da
  una variabile globale `device` e il decoder usa un loop Python O(T) (lento).
- **`AudioSTFTDataset`** (cella 23): `stft_train` calcolata con finestra `hann`
  mentre la GT su disco era salvata con `hamming` (incoerenza *risolta a monte*
  nella preprocessing corretta). `remove_window` ha `661500` hardcoded e calcola
  `output_len` senza usarlo.
- **Codice superato/mai chiamato** nel blocco generatore: `rete2` (cella 26),
  `train()/validate()` (cella 30) e le loss duplicate con `print` (celle 27–28),
  tutti sostituiti dalla pipeline GAN. Non portati nei moduli.

## 5. Verifica eseguita

- `python -m py_compile src/*.py tests/smoke_test.py` → nessun errore di sintassi.
- `python tests/smoke_test.py` (CPU, torch 2.12 CPU) → forward G e D, `discriminator_step`
  (con gradient penalty / double-backward), `generator_step` con feature matching, e
  il ramo senza discriminatore: **tutto eseguito senza eccezioni, loss finite**.

> Non è possibile eseguire il training reale localmente (richiede il dataset FMA e,
> realisticamente, una GPU): va lanciato su Kaggle tramite i notebook in `notebooks/`.
