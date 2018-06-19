# Sequence Models

This is the fifth and final course of the deep learning specialization at [Coursera](https://www.coursera.org/specializations/deep-learning) which is moderated by [deeplearning.ai](http://deeplearning.ai/). The course is taught by Andrew Ng.
## Table of contents
* [Sequence Models](#sequence-models)
   * [Table of contents](#table-of-contents)
   * [Recurrent Neural Networks](#recurrent-neural-networks)
      * [Why sequence models](#why-sequence-models)
      * [Notation](#notation)
      * [Recurrent Neural Network Model](#recurrent-neural-network-model)
      * [Backpropagation through time](#backpropagation-through-time)
      * [Different types of RNNs](#different-types-of-rnns)
      * [Language model and sequence generation](#language-model-and-sequence-generation)
      * [Sampling novel sequences](#sampling-novel-sequences)
      * [Vanishing gradients with RNNs](#vanishing-gradients-with-rnns)
      * [Gated Recurrent Unit (GRU)](#gated-recurrent-unit-gru)
      * [Long Short Term Memory (LSTM)](#long-short-term-memory-lstm)
      * [Bidirectional RNN](#bidirectional-rnn)
      * [Deep RNNs](#deep-rnns)
      * [Back propagation with RNNs](#back-propagation-with-rnns)
## Recurrent Neural Networks

> Learn about recurrent neural networks. This type of model has been proven to perform extremely well on temporal data. It has several variants including LSTMs, GRUs and Bidirectional RNNs, which you are going to learn about in this section.

### Why sequence models
- Sequence Models like RNN and LSTMs have greatly transformed learning on sequences in the past few years.
- Examples of sequence data in applications:
  - Speech recognition (**sequence to sequence**):
    - X: wave sequence
    - Y: text sequence
  - Music generation (**one to sequence**):
    - X: nothing or an integer
    - Y: wave sequence
  - Sentiment classification (**sequence to one**):
    - X: text sequence
    - Y: integer rating from one to five
  - DNA sequence analysis (**sequence to sequence**):
    - X: DNA sequence
    - Y: DNA Labels
  - Machine translation (**sequence to sequence**):
    - X: text sequence (in one language)
    - Y: text sequence (in other language)
  - Video activity recognition (**sequence to one**):
    - X: video frames
    - Y: label (activity)
  - Name entity recognition (**sequence to sequence**):
    - X: text sequence
    - Y: label sequence
    - Can be used by seach engines to index different type of words inside a text.
- All of these problems with different input and output (sequence or not) can be addressed as supervised learning with label data X, Y as the training set.

### Notation
- In this section we will discuss the notations that we will use through the course.
- **Motivating example**:
  - Named entity recognition example:
    - X: "Harry Potter and Hermoine Granger invented a new spell."
    - Y:   1   1   0   1   1   0   0   0   0
    - Both elements has a shape of 9. 1 means its a name, while 0 means its not a name.
- We will index the first element of x by x<sup><1></sup>, the second x<sup><2></sup> and so on.
  - x<sup><1></sup> = Harry
  - x<sup><2></sup> = Potter
- Similarly, we will index the first element of y by y<sup><1></sup>, the second y<sup><2></sup> and so on.
  - y<sup><1></sup> = 1
  - y<sup><2></sup> = 1

- T<sub>x</sub> is the size of the input sequence and T<sub>y</sub> is the size of the output sequence.
  - T<sub>x</sub> = T<sub>y</sub> = 9 in the last example although they can be different in other problems.
- x<sup>(i)\<t></sup> is the element t of the sequence of input vector i. Similarly y<sup>(i)\<t></sup> means the t-th element in the output sequence of the i training example.
- T<sub>x</sub><sup>(i)</sup> the input sequence length for training example i. It can be different across the examples. Similarly for T<sub>y</sub><sup>(i)</sup> will be the length of the output sequence in the i-th training example.

