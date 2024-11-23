# Natural Language Processing with RNNs

The goal of NLP research is to build machines that master language, though it typically focuses on specific tasks like text classification, translation, summarization, and sentiment analysis.

The passage outlines an exploration of recurrent neural networks (RNNs) in NLP, starting with:

Char-RNNs: Predicting the next character to generate text, using stateless and stateful RNNs.
Word-level RNNs: Performing tasks like sentiment analysis.
Encoder-decoder architectures: For neural machine translation (e.g., English to Spanish).
It then introduces attention mechanisms, which help models focus on relevant input parts, enhancing RNN-based architectures. Finally, it transitions to transformers, attention-only models like GPT and BERT, which have revolutionized NLP. The chapter concludes by exploring Hugging Face's Transformers library and demonstrating a Shakespeare-style text generation model.

## Generating Shakespearean Text with a Character RNN
Andrej Karpathy demonstrated that a character RNN (char-RNN) could generate text by predicting the next character in a sequence. This approach is applied to Shakespeare's works, training an RNN to learn patterns in text. The process involves creating a dataset of text sequences, building a model with an embedding layer and GRU, training it to predict the next character, and generating text that mimics Shakespeare's style.

Here’s the corrected and well-organized code for creating, training, and using a character RNN to generate Shakespearean text.

# Reusing Pretrained Embeddings and Language Models in NLP
## Pretrained Word Embeddings:

Word embeddings like Google’s Word2vec, Stanford’s GloVe, and Facebook’s FastText are trained on large corpora and capture general meanings of words.
## Limitation: 
Words have a single representation, regardless of context (e.g., "right" in "left and right" vs. "right and wrong").
Contextualized Word Embeddings:

Introduced by ELMo in 2018, these embeddings consider the context in which a word is used.
ELMo uses internal states of a deep bidirectional language model for richer representations.

## Language Model Pretraining:

ULMFiT (2018): Demonstrated the effectiveness of unsupervised pretraining and fine-tuning for NLP tasks.

## Key achievements:
Significant improvements in text classification tasks (18–24% error reduction).
Fine-tuned models with minimal labeled data can outperform models trained from scratch with large datasets.
This marked a shift in NLP, making pretrained models the norm.


## Transformer-Based Pretrained Models:

Universal Sentence Encoder (2018): A Google transformer-based architecture available on TensorFlow Hub.

## Example implementation:
The Universal Sentence Encoder is fine-tuned alongside additional dense layers for sentiment analysis.
The model achieves over 90% validation accuracy, comparable to human performance on ambiguous reviews.

## Practical Usage:

Pretrained embeddings and language models like the Universal Sentence Encoder can significantly improve model performance with minimal labeled data.
Models are often fine-tuned during training for task-specific improvements, leveraging the general language understanding learned from massive corpora.


# The Encoder-Decoder Network for Neural Machine Translation (NMT) model:

## Overview
This simple NMT model translates English sentences to Spanish using an Encoder-Decoder architecture with LSTM layers.

## Key Concepts
Encoder: Processes the English sentence into a hidden state.
Decoder: Generates the Spanish sentence based on the encoder's hidden state.
Teacher Forcing: During training, the decoder is fed the correct previous word instead of its own output, speeding up training.
Start-of-Sequence (SOS) & End-of-Sequence (EOS) Tokens:
SOS token is the initial input for the decoder.
EOS token marks the sentence's end.
Process

## Data Preparation:

Download an English-Spanish sentence pair dataset.
Preprocess: Remove unnecessary characters, shuffle, and split data into training and validation sets.
Tokenize sentences using two TextVectorization layers:
One for English sentences.
One for Spanish sentences (with SOS and EOS tokens).
Model Architecture:

## Encoder:
Text is tokenized and embedded using an Embedding layer.
A single LSTM layer generates the encoder's hidden states.

## Decoder:

Takes Spanish inputs (with SOS token) and generates embeddings.
LSTM layer processes embeddings, initialized with the encoder's hidden states.
Dense layer with softmax activation predicts probabilities for each word in the Spanish vocabulary.


## Training:

Loss: sparse_categorical_crossentropy
Optimizer: Nadam
Metrics: Accuracy
Training: Use teacher forcing by shifting target Spanish sentences during training.

## Inference:

During prediction, the decoder uses its previous output (not the ground truth) as input.
A utility function iteratively predicts one word at a time until reaching the EOS token.
Example Translation:
Input: "I like soccer"
Output: "me gusta el fútbol"

## Improvements:
Increase dataset size.
Use deeper architectures with more LSTM layers.
Explore advanced techniques like bidirectional layers for better performance.
This basic model is functional but struggles with long or complex sentences, highlighting the need for further refinements.

This detailed explanation of sentiment analysis provides an excellent overview of applying natural language processing techniques for classifying IMDb reviews. Here are key points and enhancements to focus on:

## Understanding the Dataset
Dataset Description: IMDb reviews dataset includes 50,000 labeled movie reviews (positive or negative), split equally into training and testing sets.
Why Popular: Simple yet challenging enough for meaningful learning on NLP techniques.
## Data Loading and Preprocessing
TensorFlow Datasets (tfds) simplifies loading the IMDb dataset. The dataset is split into training (90%), validation (10%), and test sets.
Reviews are tokenized using the TextVectorization layer. While English reviews work well with space-based tokenization, subword techniques like BPE or WordPiece are better for languages with complex tokenization.
## Model Construction
Layers:
TextVectorization: Converts text to numerical tokens.
Embedding: Maps tokens to dense vector representations.
GRU: Captures temporal dependencies in text.
Dense: Outputs a binary classification (positive/negative sentiment).
Compilation:
Loss: binary_crossentropy.
Optimizer: nadam.
Metrics: Accuracy.
## Addressing Padding Issues
Problem: Long sequences padded with zeros can overwhelm the GRU's short-term memory.
Solution:
Use mask_zero=True in the Embedding layer to ignore padding tokens during training.
Alternatively, handle sequences using ragged tensors to represent data without padding.
## Masking for Advanced Models
Layers like RNN, GRU, LSTM, etc., support masking directly.
For custom layers, you can implement call() with a mask argument and optionally propagate the mask using supports_masking=True.
## Functional API and Dropout
More flexible model definition using the functional API.
Add dropout layers to prevent overfitting.
## Ragged Tensors for Variable-Length Sequences
Ragged tensors avoid padding by representing sequences of different lengths efficiently.
Supported directly by Keras's recurrent layers, simplifying implementation.
## Experimentation Tips
Hyperparameter Tuning:
Vocabulary size, embedding dimensions, and GRU units can significantly impact performance.
## Regularization
Techniques like dropout and early stopping can prevent overfitting.
Extended Tokenization:
Subword techniques (e.g., BPE) can enhance generalization for rare words.
##  Potential Improvements
Use pre-trained embeddings (e.g., GloVe or FastText) for better initial representations.
Explore advanced architectures like bidirectional GRUs/LSTMs or Transformer-based models for further accuracy improvements.
