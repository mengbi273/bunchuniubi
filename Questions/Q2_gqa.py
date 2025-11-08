import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == "__main__":
    # Example usage:
    # 12 total Q heads, but only 6 heads for K/V => 2 groups of queries
    batch_size, seq_len, d_model = 2, 1024, 768
    num_query_heads = 12
    num_kv_heads = 6 # Must divide num_query_heads

    gqa = GroupedQueryAttention(d_model, num_query_heads, num_kv_heads)
    print("GQA module initialized successfully.")

    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)

    out = gqa(query, key, value)
    print("Output shape:", out.shape)
    assert out.shape == (batch_size, seq_len, d_model)
    print("Test passed!")

