# Autoregressive Text Generation: SimpleRNN vs BiRNN vs LSTM vs GRU

ICT4403 – Foundations of Generative AI – Assignment 2

## 1. Problem statement and objective
Build a word-level autoregressive language model that predicts the next word and generates text one word at a time. Train four recurrent architectures (Simple RNN, Bidirectional RNN, LSTM, GRU) under identical conditions, compare them quantitatively and qualitatively, study the effect of sampling temperature, and identify the most effective model.

## 2. Dataset
- **Text:** *Dark Matter* by Blake Crouch (2016), a science-fiction novel.
- **Source link:** <ADD A PUBLISHER / BOOKSTORE / LIBRARY LINK HERE>
- **Note:** the book is copyrighted, so the file is **not** included in this repository. Place your own copy next to the notebook as `Dark_Matter_-_Blake_Crouch.pdf`.
- **Characteristics (from the notebook):** ~<N> tokens, <N> unique words, mean sentence length ~10.8 words, dialogue-heavy first-person narration. The top 6,000 words cover ~98% of the text.
- **Preprocessing:** extract text from the PDF; keep only the story (drop front matter, table of contents, acknowledgements); remove page numbers and re-join hyphenated line breaks; lowercase; normalise quotes and dashes; split punctuation (`. , ! ? "`) into separate tokens; keep the top 6,000 words and map the rest to `<unk>`.
- **Input–target sequences:** the token stream is split chronologically 80/10/10 into train/validation/test before windowing. Each sample is 20 consecutive words (input) and the next word (target), sliding by one word.

## 3. Models
All models: Embedding(128) → recurrent layer (128 units) → Dropout(0.3) → Dense(softmax over vocabulary). Same optimizer (Adam, lr 1e-3), batch size 256, up to 15 epochs, early stopping on validation loss (patience 2), same random seed.

| Model | Recurrent layer |
|---|---|
| SimpleRNN | `SimpleRNN(128)` |
| BiRNN | `Bidirectional(SimpleRNN(64))` (128 output dimensions) |
| LSTM | `LSTM(128)` |
| GRU | `GRU(128)` |

## 4. Execution steps and dependencies
```bash
pip install tensorflow pypdf pandas numpy matplotlib jupyter
# put Dark_Matter_-_Blake_Crouch.pdf in the repo root
jupyter notebook dark_matter_text_generation.ipynb   # Run All
```
Outputs are written to `results/` (metrics CSVs, plots, generated samples, saved models). A GPU (e.g. Google Colab) speeds up training.

## 5. Results
<PASTE results/quantitative_results.csv AS A TABLE>

![Loss curves](results/loss_curves.png)
![Metric comparison](results/metric_comparison.png)

## 6. Model comparison and evaluation
<Write 1–2 paragraphs: perplexity, accuracy, top-5, training time, overfitting behaviour, and qualitative notes (coherence, contextual consistency, repetition, diversity), with 1–2 short sample excerpts.>

## 7. Sampling-temperature analysis
Seeds used: "I love Thursday nights", "The door opened and I stepped inside", "My wife looked at me and said". Temperatures: 0.5, 1.0, 1.5.

<PASTE results/generation_metrics.csv AS A TABLE>

![Temperature analysis](results/temperature_analysis.png)

<Discuss how temperature changes coherence, diversity (distinct-n), repetition (trigram repeat), and copying from the training text (4-gram overlap).>

## 8. Conclusion
<Name the best model and justify it using both the metrics and the generated text. State limitations honestly: a word-level model trained on one ~88k-word novel cannot produce fully coherent long passages.>
