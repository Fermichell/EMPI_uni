

## Реалізовано
- Віконне згладжування (ковзне середнє), виділено w=7.
- Експоненційне згладжування, виділено α=0.7.
- Вейвлет‑скейлограма (`pywt`), базовий вейвлет `gaus7`.
- Екстрагування ключових слів із JSONL (title, textBody) зі стоп‑словами.
- Мережа концептів: матриця суміжності + GEXF для Gephi.

## Запуск
```
python -m venv .venv && . .venv/bin/activate 
pip install -r requirements.txt

python src/pr2_task_7_smoothing.py
python src/pr2_task_7_wavelet.py
python src/pr2_task_8_keywords.py --json data/samsung.json --stop data/stopwords_ua_en_ru.txt --top 50
python src/pr2_task_9_concepts.py --json data/samsung.json --top_words outputs/top_words.txt
```

## Вихідні файли /output
- outputs/smoothing_window_w7.png
- outputs/smoothing_exponential_a7.png
- outputs/wavelet_scaleogram_7.png
- outputs/word_frequencies.csv, outputs/top_words.txt
- outputs/concepts_adjacency.csv, outputs/concepts_graph.gexf
