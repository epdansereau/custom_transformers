# For dataloader
from datasets import load_dataset
from transformers import AutoTokenizer, default_data_collator
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

# Creating the model
import torch
import torch.nn as nn
from torch.nn import functional as F

from transformers import set_seed
from transformers.activations import gelu_new
from torch.nn import CrossEntropyLoss

# Loading pretrained weights:
import os
from transformers import AutoModelForCausalLM # Used for downloading the weights
from datetime import datetime

# saving
import json

def log(message, output_dir):
    ''' Prints and saves string to a file'''
    os.makedirs(output_dir,exist_ok=True)
    log_path = os.path.join(output_dir,"log.txt")
    with open(log_path, 'a') as f:
        f.write('\n{}'.format(message))
    print(message)

def get_pretrained_tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")

class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
    """

    def __init__(self, output_dim, input_dim):
        super().__init__()
        self.output_dim = output_dim
        weight = torch.empty(input_dim, output_dim)
        nn.init.normal_(weight, std=0.02)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        ''' 
        The last dimension (the embeds) is resized
        It is important to note that different batches or tokens
        won't affect each other, only the embeds is resized
        '''
        # size_out is (batch_size, context_size, output_dim)
        size_out = x.size()[:-1] + (self.output_dim,)
        # self.bias is of shape (output_dim)
        # x is reshaped to be of size (batch_size*context_size, input_dim)
        # self.weight is of size (input_dim, output_dim)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        # x is of size (batch_size*context_size, output_dim)
        # Has to be reshaped into (batch_size, context_size, output_dim)
        x = x.view(*size_out)
        return x


'''
Something, that’s just so well explained in Jay Alammar’s post - also referenced above, is how the inputs are
passed through ATTENTION layer first and then on to FEEDFORWARD layer. The Feedforward network, is a normal neural
network that accepts the outputs from the ATTENTION layer (768), casts them to nx (768*4) dimension, adds an activation
function self.act (GELU), casts them back to d_model (768) and adds dropout (0.1).
source: https://amaarora.github.io/2020/02/18/annotatedGPT2.html
'''
class MLP(nn.Module):
    def __init__(self, inner_dim, n_embed, dropout):
        super().__init__()
        self.c_fc = Conv1D(inner_dim, n_embed)
        self.c_proj = Conv1D(n_embed, inner_dim)
        self.act = gelu_new
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)



class Attention(nn.Module):
    def __init__(self, n_embed, n_context, n_head, dropout):
        super().__init__()
        assert n_embed % n_head == 0
        self.register_buffer(
            "bias", torch.tril(torch.ones((n_context, n_context), dtype=torch.uint8)).view(1, 1, n_context, n_context)
        )
        ''' 
        self.bias
        tensor([[[[1, 0, 0,  ..., 0, 0, 0],
                [1, 1, 0,  ..., 0, 0, 0],
                [1, 1, 1,  ..., 0, 0, 0],
                ...,
                [1, 1, 1,  ..., 1, 0, 0],
                [1, 1, 1,  ..., 1, 1, 0],
                [1, 1, 1,  ..., 1, 1, 1]]]], dtype=torch.uint8)
        self.bias.shape
        torch.Size([1, 1, 1024, 1024])
        '''
        self.register_buffer("masked_bias", torch.tensor(-1e4)) 
        self.n_head = n_head
        self.split_size = n_embed
        self.c_attn = Conv1D(3 * n_embed, n_embed)
        self.c_proj = Conv1D(n_embed, n_embed)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def _attn(self, q, k, v):
        # q is of shape (batch_size, num_heads, context_size, head_embeds)
        # k is of shape (batch_size, num_heads, head_embeds, context_size)
        # By multiplying q and k, we get matrices of size (context_size, context_size),
        # where each word q_embeds and k_embeds are multiplied to see if
        # the model should pay attention or not.

        w = torch.matmul(q, k)
        # w is of shape (batch_size, num_heads, context_size, context_size)
        # scaling: weights are divided by the square root of head_embeds
        w = w / (float(v.size(-1)) ** 0.5)

        # Copying self.bias (in this case we could just do
        # mask = self.bias)
        nd, ns = w.size(-2), w.size(-1)
        mask = self.bias[:, :, ns - nd : ns, :ns]
        # The mask selects the first word, first two words, etc.
        
        # The triangular matrix allows to read words from left to right,
        # ignoring words to the right.
        w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))
        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        # Finally, we multiply the attention matrix of size (batch_size, num_heads,
        # context_size, context_size) with the value matrix of size
        # (batch_size, num_heads, context_size, head_embeds), to get a matrix
        # of size (batch_size, num_heads, context_size, head_embeds)
        # Because of the triangular attention matrix, the first row only has info
        # about the first word, the second row about the first two words, etc.
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, is_key = False):
        # x starts with shape (batch_size, context, embeds)
        # The shape returned will be (batch_size, num_heads,context, embeds/num_heads)
        # for query and value, and (batch_size, num_heads, embeds/num_heads, context)
        # for keys
        # x is splitted into a number of heads. The context_size stays the same, 
        # but embeds are split into the different heads
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if is_key:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(
        self,
        hidden_states,
        past_key_value,
    ):
        if past_key_value is not None:
            print("")

        # Cached keys and values:
        # During text generation, cached keys and values are used to speed up the process. 
        # On the first pass, q, k and v are created by a simple multiplication of the hidden states of shape
        # (batch_size, n_context, n_embd) with the c_attn weights of shape (n_embed, n_embed*3).
        # The k (..., n_ctx, n_embd) and q (..., n_embd, n_ctx) are multiplied together to obtain the
        # attention matrix of shape (..., n_ctx, n_ctx), which is then masked to ensure that words cannot pay
        # attention to the words before them.
        # The attention matrix is then multiplied with the v matrix (...,n_ctx, n_embd) to get the
        # predicted words (...,n_ctx, n_embd).
        # This process is repeated in the next blocks.
        # 
        # When a word is added, the attention matrix doesn't need to be  entirely recomputed, since 
        # the attention of the new word on the old words is masked. We only need to add the last row of the  
        # attention matrix, which is the new key multipled with the queries. The new k matrix can be obtained just
        # by adding the new k to the cached ks (same with the v matrix), since q, k and v don't depend on the
        # context of the word.
        # 

        # With cached k and v, we only need to compute q, k and v for the last word.
        # dimensions are then:
        # q (batch_size, n_head, 1, n_embd)
        # k (batch_size, n_head, n_embd, 1)
        # v (batch_size, n_head, 1, n_embd)
        query_key_value = self.c_attn(hidden_states)
        query, key, value = query_key_value.split(self.split_size, dim=2)

        query = self.split_heads(query)      #   Reshaping into
        key = self.split_heads(key, is_key=True)  #   multiple heads
        value = self.split_heads(value)      #

        # Using cached keys and values:
        if past_key_value is not None:
            # past_key (batch_size, n_head, n_embd, old_n_ctx)
            # past_value (batch_size, n_head, old_n_ctx, n_embd)
            past_key, past_value = past_key_value[0], past_key_value[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
            # with new_n_ctx = old_n_ctx + 1
            # key (batch_size, n_head, n_embd, new_n_ctx)
            # value (batch_size, n_head, new_n_ctx, n_embd)

        # Computing _attn with cache:
        # The last q (batch_size, n_head, 1, n_embd) is multiplied with K (batch_size, n_head, n_embd, new_n_ctx)
        # to get the last row of the attention matrix (batch_size, n_head, 1, new_n_ctx), which is then multiplied
        # with the V (batch_size, n_head, new_n_ctx, n_embd) to get predicted logits or hidden_state (batch_size, n_head, 1, n_embd)
        output = self._attn(query, key, value)
        output = self.merge_heads(output)   # Reshaping into a single head
        output = self.c_proj(output)  # Linear layer that keeps the same shape

        output = self.resid_dropout(output)

        return output, key, value

class Block(nn.Module):
    def __init__(self, n_context, n_embed, n_head, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embed, eps=1e-05)
        self.attn = Attention(n_embed, n_context, n_head, dropout)
        self.ln_2 = nn.LayerNorm(n_embed, eps=1e-05)
        self.mlp = MLP(4 * n_embed, n_embed, dropout)

    def forward(
        self,
        hidden_states,
        past_key_value = None
    ):
        # The hidden_states goes through a linear norm layer before attention.
        # https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        
        # The attention output will be of the same size than the hidden state,
        # but each row now contains information not only about one token, but
        # also of the tokens that came before (but not after).
        attn_output, present_key, present_value = self.attn(
            self.ln_1(hidden_states),
            past_key_value
        )
        # The attention output is added to the hidden state:
        hidden_states = attn_output + hidden_states
        
        # We then go trough a normal multilayer perceptron, which only contains
        # Conv1D layers, so the context doesn't change.
        feed_forward_hidden_states = self.mlp(self.ln_2(hidden_states))
        hidden_states = hidden_states + feed_forward_hidden_states

        return hidden_states, present_key, present_value


class GPT2Model(nn.Module):
    def __init__(self,
                vocab_size,
                n_embed,
                n_context,
                n_layer,
                n_head,
                dropout):
        super().__init__()

        self.wte = nn.Embedding(vocab_size, n_embed)

        self.wpe = nn.Embedding(n_context, n_embed)

        self.drop = nn.Dropout(dropout)

        self.h = nn.ModuleList([Block(n_context, n_embed, n_head, dropout) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embed, eps=1e-05)

        self.init_weights()

    def init_weights(self):
        """
        Initializes and prunes weights if needed.
        """
        # Initialize weights
        self.apply(self._init_weights)

        # I HAVE REMOVED THE WHOLE TYING WEIGHTS THING HERE, THOUGH IT SHOULD BE HANDLED LATER IN THE LM MODEL

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(
        self,
        input_ids,
        past_key_values,
        position_ids
    ):
        # Input_ids is a tensor of integers, each of them representing a word.
        # The tensor is of size (batch_size, context window (1024))
        batch_size, context_size = input_ids.size() # context_size is usually 1024

        if position_ids is None:
            # We create position_ids, which is a tensor of shape (1, context_size)
            # containing integers [0,1,2,3..., context_size] 
            position_ids = torch.arange(context_size, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, context_size)

        # Input_ids go through an embedding layer. (batch_size, context_size) -> (batch_size, context_size, embedding_size (768))
        inputs_embeds = self.wte(input_ids)

        # position ids go through an embedding layer, (1, context_size) -> (1, context_size, embedding_size (768))
        # the embeddings have been trained to represent position. They'll be the same for
        # each batch.
        position_embeds = self.wpe(position_ids)

        # hidden_states (batch size, context size, embed size) includes positional info
        # position_embeds are broadcasted for batch_size
        hidden_states = inputs_embeds + position_embeds

        hidden_states = self.drop(hidden_states)

        # output_shape (batch_size, context_size, embed_size)
        output_shape = (batch_size, context_size, hidden_states.size(-1),)

        presents = []
        if past_key_values is None:
            for i, block in enumerate(self.h):
                hidden_states, present_key, present_values = block(hidden_states)
                presents.append((present_key, present_values))
        else:
            for i, (block, past_key_value) in enumerate(zip(self.h, past_key_values)):
                hidden_states, present_key, present_values = block(hidden_states, past_key_value)
                presents.append((present_key, present_values))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)

        return hidden_states, presents

class GPT2LMHeadModel(nn.Module):
    def __init__(self,
                vocab_size,
                n_embed,
                n_context,
                n_layer,
                n_head,
                dropout):
        super().__init__()

        self.transformer = GPT2Model(vocab_size = vocab_size,
                                    n_embed = n_embed,
                                    n_context = n_context,
                                    n_layer = n_layer,
                                    n_head = n_head,
                                    dropout = dropout)
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)

        self.init_weights()

    def init_weights(self):
        """
        Initializes and prunes weights if needed.
        """
        # Initialize weights
        self.apply(self._init_weights)

        # TYING THE WEIGHTS
        # output_embeddings.weight = input_embeddings.weight
        self.lm_head.weight = self.transformer.wte.weight

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids,
        past_key_values,
        position_ids
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """

        hidden_states, past_key_values = self.transformer(input_ids,
                                                        past_key_values,
                                                        position_ids)
        # The lm_head is a linear layer that multiplies the hidden state (batch, ctx, embd)
        # with weights of size (vocab_size, embd). This doesn't affect context
        lm_logits = self.lm_head(hidden_states)
        # output of size (batch_size, ctx, vocab_size)
        return lm_logits, past_key_values

def download_weights(model_name, cache_path):
    ''' Downloads weights and returns path'''
    model_path = os.path.join(cache_path,model_name)
    weights_path = os.path.join(model_path,"pytorch_model.bin")
    if not(os.path.isdir(model_path) and os.path.isfile(weights_path)):
        os.makedirs(cache_path, exist_ok = True)
        _model = AutoModelForCausalLM.from_pretrained(model_name)
        _model.save_pretrained(os.path.join(cache_path,model_path))
        del _model
    return weights_path

def load_pretrained_model(model_name, model_config, device):
    model = GPT2LMHeadModel(**model_config)
    # Downloading the pretrained weights
    # See https://github.com/huggingface/transformers/issues/2422
    home = os.path.expanduser("~")
    cache_path = os.path.join(home, '.cache/transformers_local_model')
    weights_path = download_weights(model_name, cache_path)

    # The state dict can be loaded directly
    weights = torch.load(weights_path)
    model.load_state_dict(weights)

    return model.to(device)

def load_untrained_model(model_config, device):
    model = GPT2LMHeadModel(**model_config)
    return model.to(device)

def load_saved_model(model_path, device):
    config_path = os.path.join(model_path, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    model = GPT2LMHeadModel(**config)
    weights_path = os.path.join(model_path, 'pytorch_model.bin')
    weights = torch.load(weights_path)
    model.load_state_dict(weights)
    return model.to(device)

def expand_inputs_for_generation(
    input_ids,
    expand_size,
):
    expanded_return_idx = (
        torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
    )
    input_ids = input_ids.index_select(0, expanded_return_idx)
    return input_ids

def setMinLength(input_ids, min_length, eos_token_id):
    cur_len = input_ids.shape[-1]
    if cur_len < min_length:
        input_ids[:, eos_token_id] = -float("inf")
    return input_ids

def apply_temperature(input_ids, temperature):
    return input_ids / temperature

def topKLogits(input_ids, top_k, min_tokens_to_keep):
    filter_value = -float("Inf")
    top_k = min(max(top_k, min_tokens_to_keep), input_ids.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = input_ids < torch.topk(input_ids, top_k)[0][..., -1, None]
    input_ids = input_ids.masked_fill(indices_to_remove, filter_value)
    return input_ids

def topPLogits(input_ids, top_p, min_tokens_to_keep):
    filter_value = -float("Inf")
    sorted_logits, sorted_indices = torch.sort(input_ids, descending=True)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs > top_p
    if min_tokens_to_keep > 1:
        # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
        sorted_indices_to_remove[..., : min_tokens_to_keep - 1] = 0
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    input_ids = input_ids.masked_fill(indices_to_remove, filter_value)
    return input_ids

def init_sequence_length_for_generation(input_ids, max_length):
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    sequence_lengths = input_ids.new(input_ids.shape[0]).fill_(max_length)

    cur_len = input_ids.shape[-1]
    return sequence_lengths, unfinished_sequences, cur_len

def update_seq_length_for_generation(
    sequence_lengths,
    unfinished_sequences,
    cur_len,
    is_eos_in_next_token,
):
    # check if sentence is not finished yet
    is_sent_unfinished = unfinished_sequences.mul(is_eos_in_next_token.long()).bool()

    # update sentence length
    sequence_lengths = sequence_lengths.masked_fill(is_sent_unfinished, cur_len)
    unfinished_sequences = unfinished_sequences.mul((~is_eos_in_next_token).long())
    return sequence_lengths, unfinished_sequences

if __name__ == "__main__":
    # Settings
    # Dataset
    use_custom_dataset = False
    use_prepared_dataset = True
    # Custom dataset
    train_path = None
    validation_path = None
    train_val_split = 10
    # prepared dataset
    dataset_name = 'wikitext'
    dataset_config_name = 'wikitext-2-raw-v1'

    # Model 
    use_pretrained_model = True
    use_untrained_model = False
    use_saved_model = False
    #Pretrained model
    model_name = "gpt2"
    model_type = "gpt2"
    gpt2_configs = {"distilgpt2":{'n_embed':768, 'n_layer':6,'n_head':12},
        "gpt2":{'n_embed':768,'n_layer':12,'n_head':12},
        "gpt2-medium":{'n_embed':1024,'n_layer':24,'n_head':16},
        "gpt2-large":{'n_embed':1280,'n_layer':36,'n_head':20},
        "gpt2-xl":{'n_embed':1600,'n_layer':48,'n_head':25} 
    }
    model_config = gpt2_configs[model_type]
    model_config.update({'n_context':1024,'vocab_size':50257,'dropout':0.1})
    # Saved model
    saved_model_path = "gpt2_test_output/ep0step29"

    device = "cuda"
    seed = 42
    output_dir="gpt2_gen_output"

    prompt = "There were 42"

    if seed is not None:
        set_seed(seed) # A convenient function from the transformer library that sets all the seeds

    # Using the Tokenizer from hugging face
    tokenizer = get_pretrained_tokenizer()

    if use_pretrained_model:
        model = load_pretrained_model(model_name, model_config, device)
    elif use_untrained_model:
        model = load_untrained_model(model_config, device)
    elif use_saved_model:
        model = load_saved_model(saved_model_path, device)

    encoded_prompt = tokenizer.encode(prompt, return_tensors="pt")

    input_ids = encoded_prompt.to(device)
    if seed is not None:
        set_seed(seed) 
    max_len = 23
    min_len = 0
    eos_token_id = 50256
    pad_token_id = 50256
    num_beams = 1
    num_return_sequences = 1
    top_k = 0
    top_p = 0.9
    temperature = 1.0
    stop_token = None
    model.eval()
    cached_key_values = None
    # turning off gradients for induction
    torch.autograd.set_grad_enabled(False)
    # init sequence length tensors
    if num_return_sequences > 1:
        input_ids = expand_inputs_for_generation(input_ids,num_return_sequences)
    sequence_lengths, unfinished_sequences, cur_len = init_sequence_length_for_generation(input_ids, max_len)
    while cur_len < max_len:
        if cached_key_values is None:
            position_ids = None
            output, cached_key_values = model(input_ids, cached_key_values, position_ids)
        else:
            position_ids = torch.tensor([[cur_len - 1]]).to(device)
            output, cached_key_values = model(input_ids[:, -1].unsqueeze(-1), cached_key_values, position_ids)
        next_token_logits = output[:, -1, :]
        if min_len is not None and eos_token_id is not None and min_len > -1:
            next_token_logits = setMinLength(next_token_logits, min_len, eos_token_id)
        if temperature is not None and temperature != 1.0:
            next_token_logits = apply_temperature(next_token_logits, temperature)
        if top_k is not None and top_k != 0:
            next_token_logits = topKLogits(next_token_logits, top_k, min_tokens_to_keep=(2 if num_beams > 1 else 1))
        if top_p is not None and top_p < 1.0:
            next_token_logits = topPLogits(next_token_logits, top_p, min_tokens_to_keep=(2 if num_beams > 1 else 1))
        probs = F.softmax(next_token_logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        if eos_token_id is not None:
            next_tokens = next_tokens * unfinished_sequences + (pad_token_id) * (1 - unfinished_sequences)
        # add token and increase length by one
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        cur_len = cur_len + 1
        
        # update sequence length
        if eos_token_id is not None:
            sequence_lengths, unfinished_sequences = update_seq_length_for_generation(
                sequence_lengths, unfinished_sequences, cur_len, next_tokens == eos_token_id
            )

        # stop when there is a </s> in each sentence, or if we exceed the maximul length
        if unfinished_sequences.max() == 0:
            break

    if len(input_ids.shape) > 2:
        input_ids.squeeze_()

    generated_sequences = []

    for generated_sequence_idx, generated_sequence in enumerate(input_ids):
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

        # Remove all text after the stop token
        if stop_token is not None:
            text = text[: text.find(stop_token)]

        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        total_sequence = (
            prompt + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
        )

        generated_sequences.append(total_sequence)

    print(generated_sequences)
