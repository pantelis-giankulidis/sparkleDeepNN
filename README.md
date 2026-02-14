# Sparkle Deep Neural Network
A simple implementation of a deep neural network generating funny unique people's names. The model get's one or more characters from a name and tries to predict the next character in the sequence.

## The architecture
The model creates a vocabulary of 27 characters, one for each of the lowercase letters of the alphabet and one for the end or the beginning of the sequence. 
Each character, is represented as a one-hot encoding vector of size 27. The network, consists of the following layers:
* An embedding layer that takes n characters (the block size) and embeds them in a (*n* * *embedding_size*) vector. 
* Two linear Layers of size 100 each
* Batch normalization layers that follows each linear layer
* Tanh activation layers that follows each one of the batch normalization layer
* Output layer of size 27 that generates the logits.

## The implementation
The network has been created with the use of three custom components, based on pytorch, and has been train with the 32033 real people's names in the data/names.txt file.
There is track of the loss per epoch and many parameters that can be adjusted for better results!

## The example
The basic arameters that we used for training our model, as a basic working example, and exist in main.py (feel free to play with them) are the embedding size, which is set to 10, and the block size, which is 5, which
means that we take 5 characters tom predict the next. With that setting, the names that have been generated are:
ayfoah. \
memariona. \
jada. \
aeri. \
karim. \
safza. \
carola. \
ayafah. \ 
amadah. \
merye. \
malah. \
audri. \
quinnlee. \
dailey. \
talyn. \
zyaretsy. \
borie. \
mudhan. \
hadari. \
dilah.



