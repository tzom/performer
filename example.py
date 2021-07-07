import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf

NUM_HEADS=16
KEY_DIMS=512

from performer.networks.multi_head_attention import MultiHeadAttention

from performer.networks.linear_attention import Performer

layer = Performer(num_heads=NUM_HEADS, # Number of attention heads
                  key_dim=KEY_DIMS, # Size of each attention head for query and key
                  attention_method='linear', # attention method, 'linear' or 'quadratic'
                  supports=2, # only used in 'linear' attention, number of random features
                  attention_axes=None # axes over which the attention is applied.
                  )

mha = MultiHeadAttention(NUM_HEADS,KEY_DIMS)

query = tf.keras.Input(shape=[500, 64])
key = query

output_tensor = mha(query, key)
#output_tensor = layer([query, key])

print(output_tensor.shape)

