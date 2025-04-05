# Experiments-On-Transformers
Experimenting, exploring and fine-tuning transformers.


# Transformer Architecture Standard

1. Encoder: N layers (usually 6) with self-attention and feed-forward networks.
2. Decoder: N layers with self-attention, source-attention (to encoder), and feed-forward networks.
3. Attention: Mechanism to weigh word importance.
4. Forward Pass: Input → Encoder → Memory → Decoder → Output.

# Methods

Standard: Encoder-Decoder with multi-head attention. (Harvard)
Variants: BERT (encoder-only), GPT (decoder-only).
Customization: You can adjust N, hidden size, or attention heads, but the structure is usually fixed.

# Attention Mechanism
- How It Works: Attention calculates "scores" between words. For "Hello world", it checks how much "Hello" relates to "world" using their hidden states.
- Training: The model learns these relationships from data (e.g., "Hello" often precedes "world").
- Multi-Head Attention: Looks at multiple relationships at once (e.g., syntax, meaning).

# Papers

- https://nlp.seas.harvard.edu/annotated-transformer/#prelims
![Transformer Architecture](https://nlp.seas.harvard.edu/images/the_transformer_architecture.jpg)
- https://arxiv.org/pdf/1706.03762