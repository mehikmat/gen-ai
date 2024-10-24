REF: https://github.com/rasbt/LLM-workshop-2024

Input data preparation:

- Tokenization: it breaks the input text into individual tokens (in general words).
- Vocabulary: Each unique token is added to the vocabulary in alphabetical order which means assigning a unique ID to
  each token.
- These integers token IDs are used as input to LLM.
- Then the sliding window is used for selecting input and target ids.

Models like GPT, Gemma, Phi, Mistral, Llama etc. generate words sequentially and are based on the decoder
part of the original transformer architecture.

Token Embeddings:
Token IDs are simply indices representing words in a vocabulary, while embeddings are rich vector representations that
encode the semantic meaning of those words.
The transition from token IDs to embeddings usually happens in the embedding layer of a neural network, where each ID is
mapped to its corresponding vector.

Positional Embeddings: assigns positional embeddings to each token in the input sequence.

Layers and Attention heads.

Output Layers.

Softmax function.

LLM Architecture
-----------------
![llm_arch.png](llm_arch.png)