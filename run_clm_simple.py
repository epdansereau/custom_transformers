# For dataloader
from datasets import load_dataset
from transformers import AutoTokenizer, default_data_collator
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

# Creating the model
import torch
import torch.nn as nn

from transformers import set_seed
from transformers.activations import gelu_new
from torch.nn import CrossEntropyLoss

# Loading pretrained weights:
import os
from transformers import AutoModelForCausalLM # Used for downloading the weights

# Training
from transformers.optimization import AdamW, get_scheduler
from transformers import SchedulerType

set_seed(42) # A convenient function from the transformer library that sets all the seeds

### CREATING THE DATALOADER ###

source = "insert_source_path_here"

def custom_dataset(source):
    return load_dataset('text',data_files={"train":source})


# We create a dataset that indexes all lines for a text document
datasets = custom_dataset(source)
# example
# dataset['train'][0]

# Using the Tokenizer from hugging face
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def tokenize_function(examples):
    return tokenizer(examples['text'])

# Tokenizing all the examples
# The "batched" arguement means that the map function processes by batches (more efficient)
num_proc = None
load_from_cache_file = True
tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    num_proc=num_proc,
    remove_columns=['text'],
    load_from_cache_file=load_from_cache_file,
)
# The tokenize function does two thing: saves the tokens in 'input_ids',
# and creates attention mask, which are 1 for every token in this case
# However, for this specific task, we don't need the attention_mask, since
# there's no padding
# example
# tokenized_datasets['train'][0]['input_ids']
# tokenized_datasets['train'][0]['attention_mask']

# 1024 tokens in each example
# We're putting the tokens in two identical
block_size = 1024
def group_texts(examples):
    # This function will process 1000 examples at a time
    examples = sum(examples, [])
    total_length = len(examples)
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    examples = [examples[i : i + block_size] for i in range(0, total_length, block_size)]
    result = {
        "input_ids": examples,
        "labels": examples.copy(),
    }
    return result
# Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
# for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
# to preprocess.
#
# To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
# https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=None,
    load_from_cache_file=False,
    input_columns = ['input_ids'],
    remove_columns=['attention_mask'],

)
# lm_datasets["train"][0]
# {'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, ...], 'input_ids': [220, 2, 43638, 18276, 2, 350, 2394, 5781, 4242, ...], 'labels': [220, 2, 43638, 18276, 2, 350, 2394, 5781, 4242, ...]}

# Making the dataloader
# source of data collator: https://github.com/huggingface/transformers/blob/master/src/transformers/data/data_collator.py

sampler = RandomSampler(lm_datasets['train'])
collate_fn = default_data_collator
batch_size = 2
dataloader = DataLoader(lm_datasets['train'], batch_size = batch_size, collate_fn=collate_fn, sampler = sampler)

class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    I AM PRETTY SURE THAT THIS SHOULD JUST BE A LINEAR LAYER, OR AT LEAST THAT TWO LINES CAN BE REMOVED.
    I SHOULD PROBABLY MAKE A PULL REQUEST
    """

    def __init__(self, output_dim, input_dim):
        super().__init__()
        self.output_dim = output_dim
        weight = torch.empty(input_dim, output_dim)
        nn.init.normal_(weight, std=0.02)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.output_dim,) # USELESS
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out) # USELESS
        return x


'''
Something, that’s just so well explained in Jay Alammar’s post - also referenced above, is how the inputs are
passed through ATTENTION layer first and then on to FEEDFORWARD layer. The Feedforward network, is a normal neural
network that accepts the outputs from the ATTENTION layer (768), casts them to nx (768*4) dimension, adds an activation
function self.act (GELU), casts them back to d_model (768) and adds dropout (0.1).
source: https://amaarora.github.io/2020/02/18/annotatedGPT2.html
'''
class MLP(nn.Module):
    def __init__(self, n_state = 3072, nx = 768, dropout = 0.1):
        super().__init__()
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu_new
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)



class Attention(nn.Module):
    def __init__(self, nx = 768, n_ctx = 1024, n_head = 12, dropout = 0.1):
        super().__init__()
        assert nx % n_head == 0
        self.register_buffer(
            "bias", torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx)
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4)) 
        self.n_head = n_head
        self.split_size = nx
        self.c_attn = Conv1D(3 * nx, nx)
        self.c_proj = Conv1D(nx, nx)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        # scaling:
        w = w / (float(v.size(-1)) ** 0.5)
        nd, ns = w.size(-2), w.size(-1)
        mask = self.bias[:, :, ns - nd : ns, :ns]
        w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))
        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, is_key = False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if is_key:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(
        self,
        hidden_states,
    ):
        # With a linear layer, make the inputs 3 times larger and split in 3
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        query = self.split_heads(query)      #   Reshaping into
        key = self.split_heads(key, is_key=True)  #   multiple heads
        value = self.split_heads(value)      #

        output = self._attn(query, key, value)

        output = self.merge_heads(output)   # Reshaping into a single head
        output = self.c_proj(output)  # Linear layer that keeps the same shape
        output = self.resid_dropout(output)

        return output

class Block(nn.Module):
    def __init__(self, n_ctx = 1024, hidden_size = 768):
        super().__init__()
        inner_dim = 4 * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=1e-05)
        self.attn = Attention(hidden_size, n_ctx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=1e-05)
        self.mlp = MLP(inner_dim)

    def forward(
        self,
        hidden_states,
    ):
        attn_output = self.attn(
            self.ln_1(hidden_states),
        )
        # residual connection
        hidden_states = attn_output + hidden_states
        feed_forward_hidden_states = self.mlp(self.ln_2(hidden_states))
        # residual connection
        hidden_states = hidden_states + feed_forward_hidden_states

        return hidden_states


class GPT2Model(nn.Module):
    def __init__(self,
                vocab_size = 50257,
                n_embd = 768,
                n_ctx = 1024,
                n_layer = 12,
                droupout = 0.1):
        super().__init__()

        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(n_ctx, n_embd)
        self.drop = nn.Dropout(droupout)
        self.h = nn.ModuleList([Block(n_ctx) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd, eps=1e-05)

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
    ):

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
        

        position_ids = torch.arange(input_shape[-1], dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        for i, block in enumerate(self.h):
            outputs = block(hidden_states)

            hidden_states = outputs

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)

        return hidden_states

class GPT2LMHeadModel(nn.Module):

    def __init__(self,
                vocab_size = 50257,
                n_embd = 768,
                n_ctx = 1024,
                n_layer = 12,
                droupout = 0.1):
        super().__init__()
        self.transformer = GPT2Model(vocab_size, n_embd, n_ctx, n_layer)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

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
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """

        hidden_states = self.transformer(input_ids)

        lm_logits = self.lm_head(hidden_states)

        return lm_logits

model = GPT2LMHeadModel()

# Downloading the pretrained weights
# See https://github.com/huggingface/transformers/issues/2422
home = os.path.expanduser("~")
cache_path = os.path.join(home, '.cache/transformers_local_model')
model_name = "gpt2"
def download_weights(model_name, cache_path):
    model_path = os.path.join(cache_path,model_name)
    weights_path = os.path.join(model_path,"pytorch_model.bin")
    if not(os.path.isdir(model_path) and os.path.isfile(weights_path)):
        os.makedirs(cache_path, exist_ok = True)
        _model = AutoModelForCausalLM.from_pretrained(model_name)
        _model.save_pretrained(os.path.join(cache_path,model_path))
        del _model
    return weights_path

weights_path = download_weights(model_name, cache_path)

# The state dict can be loaded directly
weights = torch.load(weights_path)
model.load_state_dict(weights)

# Doing training
epochs = 3
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_epsilon = 1e-08
learning_rate = 5e-05
warmup_steps = 0
lr_scheduler_type = "linear"
device = "cuda"
max_grad_norm = 1.0
seed = 42

model.to(device)

# Creating the optimizer and scheduler
max_steps = len(dataloader)*epochs
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

optimizer = AdamW(optimizer_grouped_parameters, betas = (adam_beta1, adam_beta2), eps = adam_epsilon, lr = learning_rate)
lr_scheduler = get_scheduler(
    name = SchedulerType(lr_scheduler_type),
    optimizer = optimizer,
    num_warmup_steps = warmup_steps,
    num_training_steps = max_steps,
)

### Training loop

model.zero_grad()
training_loss = 0
loss_func=CrossEntropyLoss()
print('done')
set_seed(seed)
for epoch in range(epochs):
    for step, inputs in enumerate(dataloader):
        model.train()
        input_ids = inputs['input_ids'].to(device)
        labels = inputs['labels'].to(device)
        logits = model(input_ids)
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss = loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        model.zero_grad()
