# Sentiment Analysis Pipeline Assignment





## Catalog
- [Sentiment Analysis Pipeline Assignment](#sentiment-analysis-pipeline-assignment)
  - [Catalog](#catalog)
  - [Objective](#objective)
  - [Structure](#structure)
  - [How to Run](#how-to-run)
  - [Notes](#notes)
  - [Deliverables](#deliverables)



## Objective
Build a Python pipeline to simulate streaming data ingestion, perform sentiment analysis, and visualise results.



## Structure
- `scripts/`: All processing scripts
- `data/sample_stream.jsonl`: Input file (simulated stream)
- `output/sample`: Results and plots for sample stream data



## How to Run

1. Install dependencies
   ```shell
   pip install -r requirements.txt
   ```


2. Usge
   > First run this command on shell to ensure you can running this program.
   ```shell
   python scripts/main.py
   ```

   > Then see the following words to see all the arguments/options that program allowed.
   ```shell
   usage: main.py [-h] --input-stream INPUT_STREAM --output-csv-dir OUTPUT_CSV_DIR --output-plot-dir OUTPUT_PLOT_DIR [--visual-comments-chunks VISUAL_COMMENTS_CHUNKS]
               [-visual-tokens-percent VISUAL_TOKENS_PERCENT]

   Sentiment Analysis Pipeline

   options:
     -h, --help            show this help message and exit
     --input-stream INPUT_STREAM
                           the input stream file path, must be a `.jsonl` file
     --output-csv-dir OUTPUT_CSV_DIR
                           the output dir of the sentiment analysis CSV files
     --output-plot-dir OUTPUT_PLOT_DIR
                           the output dir of the image for the visualization analysis result
     --visual-comments-chunks VISUAL_COMMENTS_CHUNKS
                           chunk numbers or steps that used for draw the comments sentiment trend
     -visual-tokens-percent VISUAL_TOKENS_PERCENT
                           to show the top N tokens by percentage in the ouput visualization for tokens sentiment trend
   ```



## Notes
- Ensure NLTK corpora are downloaded before running:
   ```python
   import nltk
   nltk.download("gutenberg")
   nltk.download("genesis")
   nltk.download("inaugural")
   nltk.download("nps_chat")
   nltk.download("webtext")
   nltk.download("treebank")
   nltk.download('punkt_tab')
   nltk.download("averaged_perceptron_tagger_eng")
   nltk.download("wordnet")
   nltk.download("stopwords")
   nltk.download("vader_lexicon")
   ```



## Deliverables
- Clean scripts that can used on production environment
- Output CSV with sentiment scores that can used for development and further processing
- Visualized images can be used for manual demonstrations (speeches)
