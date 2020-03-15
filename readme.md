## Deep Learning Course (980)
## Assignment Three 

__Assignment Goals:__

- Implementing RNN based language models.
- Implementing and applying a Recurrent Neural Network on text classification problem using TensorFlow.
- Implementing __many to one__ and __many to many__ RNN sequence processing.

In this assignment, you will implement RNN-based language models and compare extracted word representation from different models. You will also compare two different training methods for sequential data: Truncated Backpropagation Through Time __(TBTT)__ and Backpropagation Through Time __(BTT)__. 
Also, you will be asked to apply Vanilla RNN to capture word representations and solve a text classification problem. 


__DataSets__: You will use two datasets, an English Literature for language model task (part 1 to 4) and 20Newsgroups for text classification (part 5). 


1. (30 points) Implement the RNN based language model described by Mikolov et al.[1], also called __Elman network__ and train a language model on the English Literature dataset. This network contains input, hidden and output layer and is trained by standard backpropagation (TBTT with τ = 1) using the cross-entropy loss. 
   - The input represents the current word while using 1-of-N coding (thus its size is equal to the size of the vocabulary) and vector s(t − 1) that represents output values in the hidden layer from the previous time step. 
   - The hidden layer is a fully connected sigmoid layer with size 500. 
   - Softmax Output Layer to capture a valid probability distribution.
   - The model is trained with truncated backpropagation through time (TBTT) with τ = 1: the weights of the network are updated based on the error vector computed only for the current time step.
   
   Download the English Literature dataset and train the language model as described, report the model cross-entropy loss on the train set. Use nltk.word_tokenize to tokenize the documents. 
For initialization, s(0) can be set to a vector of small values. Note that we are not interested in the *dynamic model* mentioned in the original paper. 
To make the implementation simpler you can use Keras to define neural net layers, including Keras.Embedding. (Keras.Embedding will create an additional mapping layer compared to the Elman architecture.) 

2. (20 points) TBTT has less computational cost and memory needs in comparison with *backpropagation through time algorithm (BTT)*. These benefits come at the cost of losing long term dependencies [2]. Now let's try to investigate computational costs and performance of learning our language model with BTT. For training the Elman-type RNN with BTT, one option is to perform mini-batch gradient descent with exactly one sentence per mini-batch. (The input  size will be [1, Sentence Length]). 

    1. Split the document into sentences (you can use nltk.tokenize.sent_tokenize).
    2. For each sentence, perform one pass that computes the mean/sum loss for this sentence; then perform a gradient update for the whole sentence. (So the mini-batch size varies for the sentences with different lengths). You can truncate long sentences to fit the data in memory. 
    3. Report the model cross-entropy loss.

3. (15 points) It does not seem that simple recurrent neural networks can capture truly exploit context information with long dependencies, because of the problem that gradients vanish and exploding. To solve this problem, gating mechanisms for recurrent neural networks were introduced. Try to learn your last model (Elman + BTT) with the SimpleRnn unit replaced with a Gated Recurrent Unit (GRU). Report the model cross-entropy loss. Compare your results in terms of cross-entropy loss with two other approach(part 1 and 2). Use each model to generate 10 synthetic sentences of 15 words each. Discuss the quality of the sentences generated - do they look like proper English? Do they match the training set?
    Text generation from a given language model can be done using the following iterative process:
   1. Set sequence = \[first_word\], chosen randomly.
   2. Select a new word based on the sequence so far, add this word to the sequence, and repeat. At each iteration, select the word with maximum probability given the sequence so far. The trained language model outputs this probability. 

4. (15 points) The text describes how to extract a word representation from a trained RNN (Chapter 4). How we can evaluate the extracted word representation for your trained RNN? Compare the words representation extracted from each of the approaches using one of the existing methods.

5. (20 points) We are aiming to learn an RNN model that predicts document categories given its content (text classification). For this task, we will use the 20Newsgroupst dataset. The 20Newsgroupst contains messages from twenty newsgroups.  We selected four major categories (comp, politics, rec, and religion) comprising around 13k documents altogether. Your model should learn word representations to support the classification task. For solving this problem modify the __Elman network__ architecture such that the last layer is a softmax layer with just 4 output neurons (one for each category). 

    1. Download the 20Newsgroups dataset, and use the implemented code from the notebook to read in the dataset.
    2. Split the data into a training set (90 percent) and validation set (10 percent). Train the model on  20Newsgroups.
    3. Report your accuracy results on the validation set.

__NOTE__: Please use Jupyter Notebook. The notebook should include the final code, results and your answers. You should submit your Notebook in (.pdf or .html) and .ipynb format. (penalty 10 points) 

To reduce the parameters, you can merge all words that occur less often than a threshold into a special rare token (\__unk__).

__Instructions__:

The university policy on academic dishonesty and plagiarism (cheating) will be taken very seriously in this course. Everything submitted should be your own writing or coding. You must not let other students copy your work. Spelling and grammar count.

Your assignments will be marked based on correctness, originality (the implementations and ideas are from yourself), clarification and test performance.


[1] Tom´ as Mikolov, Martin Kara ˇ fiat, Luk´ ´ as Burget, Jan ˇ Cernock´ ˇ y,Sanjeev Khudanpur: Recurrent neural network based language model, In: Proc. INTERSPEECH 2010

[2] Tallec, Corentin, and Yann Ollivier. "Unbiasing truncated backpropagation through time." arXiv preprint arXiv:1705.08209 (2017).