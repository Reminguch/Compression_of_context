- The working file if Unsloth.py

We download the unsloth Qwen3 model, and try to compress it with our custom class Qwen3Compressed
- This custom class inclued: the 
- custom wrapper
- the custom MLP class, which allows us to compress the tokens
- the custom attention kernel, computing the intermediate importance scores



- We then try to train the model with our custom loss function, which allows us to train the model with windowed supervision
- We then try to train the model with our custom loss function, which allows us to train the model with windowed supervision

At the current stage, we encounter the two problems:
 - inference is not working correctly
 - minor issue with the 5-D array returned by the custom attention kernel

 Lets dig into the Compress.py file: