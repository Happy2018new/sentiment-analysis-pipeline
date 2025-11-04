# Sentiment Analysis Pipeline Assignment

## Objective
Build a Python pipeline to simulate streaming data ingestion, perform sentiment analysis, and visualise results.

## Structure
- `scripts/`: All processing scripts
- `data/sample_stream.jsonl`: Input file (simulated stream)
- `output/`: Results and plots
- `logs/`: Log files (optional)

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the main pipeline:
   ```bash
   python scripts/main.py --input data/sample_stream.jsonl --output output/sentiment_results.csv
   ```

3. Generate plots:
   ```bash
   python scripts/visualise.py --input output/sentiment_results.csv
   ```

## Notes
- Ensure NLTK corpora are downloaded before running:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

## Deliverables
- Clean scripts
- Output CSV with sentiment labels
- Saved plots
