# Naver Product Comment Classification Project

This project involves building a machine learning model to classify Naver product comments as either **negative (악플)** or **positive (좋은 말)** based on their star ratings. The main goal is to preprocess the data, train an LSTM-based neural network, and evaluate its performance.

## Table of Contents

1. Project Overview
2. Dependencies
3. Data Preprocessing
4. Model Architecture
5. Training and Evaluation
6. Results

---

## Project Overview

The project focuses on classifying Naver product comments into two categories: negative or positive. The dataset consists of product comments with corresponding star ratings. The workflow includes:

- Cleaning and preprocessing the text data.
- Tokenizing text into sequences.
- Training a neural network using TensorFlow/Keras.
- Evaluating the model's performance on a validation set.

---

## Dependencies

Ensure you have the following Python libraries installed:

- pandas
- numpy
- tensorflow
- scikit-learn

Install them using:

```
pip install pandas numpy tensorflow scikit-learn
```

---

## Data Preprocessing

1. **Removing Special Characters**:
Special characters were removed to focus on Korean characters, numbers, and spaces.
    
    ```
    raw['review'] = raw['review'].str.replace('[^ㄱ-하-ㅣ가-힣0-9 ]','')
    ```
    
2. **Removing Duplicates**:
Duplicate reviews were dropped to ensure uniqueness in the dataset.
    
    ```
    raw.drop_duplicates(subset=['review'], inplace=True)
    ```
    
3. **Tokenization**:
    - A character-level tokenizer was used.
    - Unknown tokens were handled using an `oov_token`.
    
    ```
    tokenizer = Tokenizer(char_level=True, oov_token='')
    tokenizer.fit_on_texts(raw['review'].tolist())
    ```
    
4. **Padding Sequences**:
All sequences were padded to a fixed length of 100 characters.
    
    ```
    X = pad_sequences(train_seq, maxlen=100)
    ```
    
5. **Train-Validation Split**:
The dataset was split into training and validation sets (80:20 ratio).
    
    ```
    from sklearn.model_selection import train_test_split
    trainX, valX, trainY, valY = train_test_split(X, Y, test_size=0.2, random_state=42)
    ```
    

---

## Model Architecture

The model is a Sequential neural network comprising:

1. **Embedding Layer**:
Converts input sequences into dense vectors of fixed size.
    
    ```
    tf.keras.layers.Embedding(len(tokenizer.word_index) + 1, 16)
    ```
    
2. **LSTM Layer**:
Captures sequential dependencies in text data.
    
    ```
    tf.keras.layers.LSTM(128)
    ```
    
3. **Dense Output Layer**:
Outputs a single value with a sigmoid activation for binary classification.
    
    ```
    tf.keras.layers.Dense(1, activation='sigmoid')
    ```
    

The model is compiled with the following configuration:

```
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
```

---

## Training and Evaluation

1. **Training**:
The model is trained for 5 epochs with a batch size of 32.
    
    ```
    history = model.fit(
        trainX, trainY,
        validation_data=(valX, valY),
        epochs=5,
        batch_size=32,
        verbose=1
    )
    ```
    
2. **Validation**:
The model is evaluated on the validation set to determine accuracy.
    
    ```
    loss, accuracy = model.evaluate(valX, valY, verbose=0)
    print(f"Validation Accuracy: {accuracy:.4f}")
    ```
    

---

## Results

The model's validation accuracy provides insight into its performance. Further tuning, such as adjusting hyperparameters or improving data preprocessing, can enhance results.

Example output:

```
Validation Accuracy: 0.8310
```

---

## Future Improvements

- Experiment with different sequence lengths and batch sizes.
- Implement additional preprocessing steps like stemming or stop-word removal.
- Use more advanced architectures (e.g., Bidirectional LSTMs or Transformers).
- Gather more labeled data to improve model robustness.
