# "Sorry, I didn't notice originally you wrote gson!", When Emotions and Opinions Can be Misdirected
Replication Package for the above titled paper.

1. Datasets:
- Online review dataset
- Benchmark datasets
- Benchmark datasets with noise
- Noise file
1. Source Code
- SO data collection
- Library Opinion Extraction
- Data file splitting for train/validate/test: data-file-processing.py
- Word/Adjective frequency calculation: 
- Noise generation: text-generation-openai.py
- Noise augmentation:noise-augmentation.py
- Classification by OpenAI: opneai-classfication-sa4se.py
- Evaluation of OpenAI model against four benchmark data: model-evaluation-benchmark-data.py
- Evaluation of the models (PTM+LLM) against online review data: model-evaluation-real-data.py
- Fine-tuning of PTM models against datasets: sentiment-analysis-SimCSE.v4.ipynb
- Predicting sentiment using a Twitter trained model: label-sentiment-analysis-BERT.ipynb
1. Model paths: cannot be uploaded because of size limitation (more than 10 GB) being anonymous. Will be uploaded afterwards.

