# RNN vs. LSTM for Sentiment Analysis in TensorFlow

This project provides a comparative basic example of using a simple Recurrent Neural Network (RNN) and a Long Short-Term Memory (LSTM) network for sentiment analysis on the IMDB movie reviews dataset using TensorFlow/Keras. The goal is to demonstrate the fundamental differences in their architecture and performance in handling sequential data.

## Project Overview

The Python script performs the following steps for both RNN and LSTM models:

1.  **Dataset Loading and Preprocessing**: Loads the IMDB movie reviews dataset and pads sequences to a uniform length.
2.  **Model Building**: Defines two separate `Sequential` models: one using `SimpleRNN` and another using `LSTM`.
3.  **Model Compilation**: Configures each model with an Adam optimizer, binary cross-entropy loss, and accuracy as a metric.
4.  **Model Training**: Trains both models on the preprocessed training data with validation.
5.  **Model Evaluation**: Assesses the performance of both trained models on the unseen test dataset.
6.  **Comparison**: Outputs the test loss and accuracy for both models for direct comparison.

## Dataset

The **IMDB movie reviews dataset** is a standard benchmark for binary sentiment classification. It contains 50,000 highly polarized movie reviews (25,000 for training, 25,000 for testing), labeled as either positive (1) or negative (0). The dataset is pre-processed, with reviews already converted into sequences of integers, where each integer represents a specific word.

### Data Preprocessing

* **Vocabulary Size (`vocab_size`)**: Set to 10,000, considering only the most frequent words.
* **Maximum Sequence Length (`max_len`)**: Set to 200. All movie review sequences are padded with zeros (`padding='post'`) or truncated to this fixed length.

## RNN vs. LSTM: Key Differences

Both RNNs and LSTMs are types of recurrent neural networks designed to process sequential data. However, LSTMs are an advancement over simple RNNs, primarily designed to address the **vanishing gradient problem** and better capture **long-term dependencies** in sequences.

### Simple RNN (`tf.keras.layers.SimpleRNN`)

* **Architecture**: A basic recurrent unit that has a hidden state (`h_t`) updated at each time step based on the current input (`x_t`) and the previous hidden state (`h_{t-1}`).
    * Equation: `h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)`
* **Strengths**: Simplicity, faster computation than more complex RNNs.
* **Weaknesses**:
    * **Vanishing Gradient Problem**: Gradients can shrink exponentially over many time steps, making it difficult for the network to learn dependencies stretching back more than a few steps.
    * **Short-term Memory**: Due to vanishing gradients, simple RNNs struggle to retain information over long sequences.

### Long Short-Term Memory (LSTM) (`tf.keras.layers.LSTM`)

* **Architecture**: LSTMs introduce a more complex recurrent unit with a "cell state" (`C_t`) in addition to the hidden state (`h_t`). The cell state acts like a "conveyor belt" that runs straight through the entire chain, with only some minor linear interactions. Information can be added to or removed from the cell state via specialized "gates":
    * **Forget Gate**: Decides what information from the previous cell state should be thrown away.
    * **Input Gate**: Decides what new information should be stored in the cell state.
    * **Output Gate**: Decides what part of the cell state should be output to the hidden state.
* **Strengths**:
    * **Addresses Vanishing Gradient**: The gating mechanism allows gradients to flow through the cell state more effectively, alleviating the vanishing gradient problem.
    * **Long-term Memory**: Capable of learning long-term dependencies, making them highly effective for tasks with long sequences (like long sentences in text or extended time series data).
* **Weaknesses**: More computationally intensive and have more parameters than simple RNNs due to their complex internal structure.

## Model Architectures

Both models share common components:

* **`Embedding(input_dim=vocab_size, output_dim=128)`**: Converts word indices into dense 128-dimensional vectors. This layer learns a numerical representation for each word.
* **`return_sequences=False`**: For both RNN and LSTM layers, this ensures that only the output from the last time step of the sequence is passed to the next layer, which is appropriate for sequence classification tasks where a single prediction is made per sequence.
* **`Dense(1, activation='sigmoid')`**: The output layer for binary classification, yielding a probability between 0 and 1.

### Simple RNN Model

```python
# rnn_model definition
rnn_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128),
    SimpleRNN(128, activation='tanh', return_sequences=False),
    Dense(1, activation='sigmoid')
])
rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
rnn_model.summary()