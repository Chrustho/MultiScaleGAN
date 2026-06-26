# Multi-Scale GAN per Audio Super-Resolution

Codice della tesi di laurea triennale "Multi-Scale GAN per Audio Super-Resolution:
ricostruzione di segnali musicali con RNN e Discriminatore Multi-Scala".

Il progetto realizza una rete generativa avversaria (GAN) che ricostruisce il
contenuto in alta frequenza di tracce musicali. A partire da un segnale a banda
limitata, tipico di registrazioni datate o di bassa qualità, la rete genera una
stima delle componenti spettrali mancanti, migliorando la qualità percepita
dell'audio. Il lavoro si colloca nell'ambito dell'Audio Super-Resolution e opera
nel dominio della Short-Time Fourier Transform.

## Informazioni

Tesi di Laurea Triennale in Ingegneria Informatica, Dipartimento di Ingegneria
Informatica, Modellistica, Elettronica e Sistemistica dell'Università della Calabria.

- Candidato: Christian Bruni (matricola 240008)
- Relatore: Prof. Luigi Palopoli
- Correlatrice: Dott.ssa Simona Nisticò
- Anno accademico 2024/2025

La rete generatrice, basata su una rete neurale ricorrente, proviene da un lavoro
di tesi precedente ed è qui riusata senza modifiche. Il contributo di questo lavoro
è l'aggiunta del discriminatore multi-scala e la costruzione del ciclo di
addestramento avversario che insieme formano la GAN.

## Come funziona

Ogni traccia audio viene portata a una lunghezza fissa e ne vengono calcolate due
rappresentazioni tramite STFT. La prima si ottiene applicando al segnale un filtro
passa-basso con frequenza di taglio a 7 kHz e rappresenta l'ingresso a banda
limitata. La seconda è calcolata sul segnale originale e costituisce il riferimento
da ricostruire. Entrambe sono tensori complessi, con parte reale e parte
immaginaria trattate separatamente.

Il generatore riceve la STFT a banda limitata e produce una STFT stimata a banda
piena. Il discriminatore confronta la STFT generata con quella di riferimento e
fornisce un segnale di addestramento avversario, affiancato a una loss di
ricostruzione che misura la fedeltà spettrale della stima. L'addestramento alterna
aggiornamenti del discriminatore e del generatore secondo la logica delle
Wasserstein GAN con penalità del gradiente.

## Struttura del repository

```
src/
  config.py          Path e iperparametri centralizzati
  preprocessing.py   Costruzione degli split, estrazione di forme d'onda e STFT
  dataset.py         AudioSTFTDataset e utility di segnale (rete generatrice, invariato)
  generator.py       Rete ricorrente ReteRecWithAttention (rete generatrice, invariato)
  discriminator.py   SubDiscriminator e MultiScaleDiscriminator
  losses.py          Loss di ricostruzione, avversaria, feature matching, gradient penalty
  train_gan.py       Ciclo di addestramento della GAN
notebooks/
  01_preprocessing.ipynb   Download del dataset e preparazione dei dati
  02_train_gan.ipynb       Costruzione dei modelli e avvio dell'addestramento
REVIEW.md            Relazione sulla revisione del codice
requirements.txt     Dipendenze Python
tesi.pdf  Testo della tesi
```

## Dataset

L'addestramento usa il sottoinsieme Free Music Archive Small, composto da 8000 clip
musicali di 30 secondi distribuite su 8 generi. Le tracce sono caricate, riportate a
una lunghezza fissa e suddivise in addestramento, test e validazione. Il
preprocessing produce per ogni traccia una forma d'onda e la relativa STFT, salvate
come file numpy e accoppiate per nome durante il caricamento.

## Esecuzione

Il codice è pensato per l'ambiente Kaggle, dove i percorsi predefiniti puntano a
`/kaggle/temp` e `/kaggle/working`. Per eseguirlo altrove è sufficiente impostare le
variabili d'ambiente `DATA_ROOT`, `WORKING_DIR` e `CHECKPOINT_DIR`, senza toccare il
codice.

```bash
pip install -r requirements.txt
```

Il flusso di lavoro si articola in due fasi. La prima, nel notebook
`01_preprocessing.ipynb`, scarica il dataset e genera forme d'onda e STFT. La
seconda, nel notebook `02_train_gan.ipynb`, costruisce generatore e discriminatore e
avvia l'addestramento richiamando `train_gan.main()`. Gli iperparametri si possono
modificare direttamente come variabili del modulo `train_gan` prima di lanciare
l'addestramento.

## Architettura

Il generatore è una rete ricorrente con struttura a encoder e decoder. La parte
reale e la parte immaginaria della STFT in ingresso sono elaborate da moduli LSTM
distinti, le rappresentazioni vengono combinate e il decoder ricostruisce passo dopo
passo le componenti spettrali, proiettate infine sul numero di bin di frequenza
originale.

Il discriminatore segue una logica multi-scala ispirata a MelGAN. Tre
sotto-discriminatori operano su risoluzioni temporali decrescenti, ottenute
sotto-campionando lo spettro tra una scala e l'altra con un average pooling. Ogni
sotto-discriminatore elabora separatamente parte reale e immaginaria con quattro
strati convoluzionali, normalizzazione, attivazione LeakyReLU e dropout, poi fonde le
due rappresentazioni e produce una mappa di punteggi. Oltre ai punteggi vengono
restituite le rappresentazioni intermedie, usate dalla feature matching loss.

## Funzioni di loss

La loss del generatore combina un errore quadratico medio sulle componenti reale e
immaginaria, una loss di coerenza spettrale che confronta le variazioni temporali
del modulo della STFT, la loss avversaria fornita dal discriminatore e una feature
matching loss che avvicina le rappresentazioni intermedie del discriminatore tra
segnale reale e generato.

La loss del discriminatore è quella delle Wasserstein GAN, con una penalità del
gradiente che vincola la condizione di Lipschitz. Il discriminatore viene addestrato
in precisione piena, mentre il generatore usa la mixed precision con un gradient
scaler.

## Iperparametri principali

I valori predefiniti sono definiti in `src/train_gan.py`.

- Numero di epoche: 30
- Dimensione del batch: 2, con accumulazione del gradiente su 3 passi
- Learning rate del generatore: 0.002
- Learning rate del discriminatore: 0.0006
- Pesi della loss del generatore: 0.5 per la MSE, 0.9 per la coerenza spettrale, 0.7 per la parte avversaria, 1.0 per la feature matching
- Peso della gradient penalty: 0.4
- Aggiornamenti del discriminatore per batch: 2, portati a 3 durante le prime epoche di riscaldamento
- Salvataggio dei checkpoint e validazione ogni 5 epoche

## Requisiti

Le dipendenze sono elencate in `requirements.txt`: PyTorch, NumPy, SciPy, librosa,
matplotlib e tqdm. L'addestramento completo richiede una GPU e il dataset FMA, ed è
stato eseguito su Kaggle con due GPU Nvidia Tesla T4.
