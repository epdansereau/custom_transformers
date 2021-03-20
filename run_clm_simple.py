# For dataloader
from datasets import load_dataset
from transformers import AutoTokenizer, default_data_collator
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

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
from datetime import datetime, timedelta
import math

# saving
import json
import warnings
from transformers.trainer_pt_utils import reissue_pt_warnings

def log(message, output_dir):
    ''' Prints and saves string to a file'''
    os.makedirs(output_dir,exist_ok=True)
    log_path = os.path.join(output_dir,"log.txt")
    with open(log_path, 'a') as f:
        f.write('\n{}'.format(message))
    print(message)

def get_pretrained_tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")

def custom_dataset(train_path = None, validation_path = None, validation_split_percentage = None):
    data_files = {}
    if train_path is not None:
        data_files['train'] = train_path
    if validation_path is not None:
        data_files['validation'] = validation_path
    datasets = load_dataset('text',data_files=data_files)
    if validation_split_percentage is not None:
        datasets["validation"] = load_dataset(
            'text',
            data_files=data_files,
            split="train[:{}%]".format(validation_split_percentage),
        )
        datasets["train"] = load_dataset(
            'text',
            data_files=data_files,
            split="train[{}%:]".format(validation_split_percentage),
        )
    return datasets


def prepared_dataset(name, config_name):
    return load_dataset(name, config_name)

def tokenize_function(examples):
    return tokenizer(examples['text'])

# 1024 tokens in each example
# We're putting the tokens in two identical
n_context = 1024
def group_texts(examples):
    # This function will process 1000 examples at a time
    examples = sum(examples, [])
    total_length = len(examples)
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // n_context) * n_context
    # Split by chunks of max_len.
    examples = [examples[i : i + n_context] for i in range(0, total_length, n_context)]
    result = {
        "input_ids": examples,
        "labels": examples.copy(),
    }
    return result
# Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
# for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
# to preprocess.

# Tokenizing all the examples
# The "batched" arguement means that the map function processes by batches (more efficient)
def get_dataloader(datasets, tokenizer, use_cache):
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=None,
        remove_columns=['text'],
        load_from_cache_file=use_cache,
    )
    # The tokenize function does two thing: saves the tokens in 'input_ids',
    # and creates attention mask, which are 1 for every token in this case
    # However, for this specific task, we don't need the attention_mask, since
    # there's no padding
    # example
    # tokenized_datasets['train'][0]['input_ids']
    # tokenized_datasets['train'][0]['attention_mask']

    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=None,
        load_from_cache_file=use_cache,
        input_columns = ['input_ids'],
        remove_columns=['attention_mask'],

    )
    # lm_datasets["train"][0]
    # {'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, ...], 'input_ids': [220, 2, 43638, 18276, 2, 350, 2394, 5781, 4242, ...], 'labels': [220, 2, 43638, 18276, 2, 350, 2394, 5781, 4242, ...]}
    # Making the dataloader
    # source of data collator: https://github.com/huggingface/transformers/blob/master/src/transformers/data/data_collator.py

    collate_fn = default_data_collator
    if 'train' in lm_datasets:
        train_sampler = RandomSampler(lm_datasets['train'])
        train_dataloader = DataLoader(lm_datasets['train'], batch_size = batch_size, collate_fn=collate_fn, sampler = train_sampler)
    else:
        train_dataloader = None
    
    if 'validation' in lm_datasets:
        eval_sampler = SequentialSampler(lm_datasets['validation'])
        eval_dataloader = DataLoader(lm_datasets['validation'], batch_size = batch_size, collate_fn=collate_fn, sampler = eval_sampler)
    else:
        eval_dataloader = None
    
    return train_dataloader, eval_dataloader

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
    ):
        # With a linear layer, make the inputs 3 times larger
        # The dimension goes from (batch_size, context_size, embed size)
        # to (batch_size, context_size, embed size * 3)
        # Only the last dimension is modified through the Conv1D layer,
        # meaning that tokens don't influence each other on that step
        query_key_value = self.c_attn(hidden_states)
        # query, key value all have shape (batch_size, context_size, embed size)
        query, key, value = query_key_value.split(self.split_size, dim=2)

        query = self.split_heads(query)      #   Reshaping into
        key = self.split_heads(key, is_key=True)  #   multiple heads
        value = self.split_heads(value)      #
        # The different head contains different embeds, and are able to pay attention
        # to different words.

        # the output of _attn is of shape (batch_size, num_heads, context_size, head_embeds)
        # The embeds for each token now contains information about the tokens that comes
        # before it, selected by self-attention, but not from tokens that come after 
        output = self._attn(query, key, value)
        # The output is then reshaped to be of size (batch_size, context_size, embeds(768))
        output = self.merge_heads(output)   # Reshaping into a single head
        # We then go through an additional Conv1D layer. This one doesn't change the size,
        # of the embeds, and we don't get info from outside of the allowed context.
        output = self.c_proj(output)  # Linear layer that keeps the same shape

        output = self.resid_dropout(output)

        return output

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
    ):
        # The hidden_states goes through a linear norm layer before attention.
        # https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        
        # The attention output will be of the same size than the hidden state,
        # but each row now contains information not only about one token, but
        # also of the tokens that came before (but not after).
        attn_output = self.attn(
            self.ln_1(hidden_states),
        )
        # The attention output is added to the hidden state:
        hidden_states = attn_output + hidden_states
        
        # We then go trough a normal multilayer perceptron, which only contains
        # Conv1D layers, so the context doesn't change.
        feed_forward_hidden_states = self.mlp(self.ln_2(hidden_states))
        hidden_states = hidden_states + feed_forward_hidden_states

        return hidden_states


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
    ):
        # Input_ids is a tensor of integers, each of them representing a word.
        # The tensor is of size (batch_size, context window (1024))
        batch_size, context_size = input_ids.size() # context_size is usually 1024

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

        for i, block in enumerate(self.h):
            hidden_states = block(hidden_states)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)

        return hidden_states

class GPT2LMHeadModel(nn.Module):
    def __init__(self,
                vocab_size,
                n_embed,
                n_context,
                n_layer,
                n_head,
                dropout):
        super().__init__()
        self.config = {'vocab_size':vocab_size,
                    'n_embed':n_embed,
                    'n_context':n_context,
                    'n_layer':n_layer,
                    'n_head':n_head,
                    'dropout':dropout}

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
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """

        hidden_states = self.transformer(input_ids)
        # The lm_head is a linear layer that multiplies the hidden state (batch, ctx, embd)
        # with weights of size (vocab_size, embd). This doesn't affect context
        lm_logits = self.lm_head(hidden_states)
        # output of size (batch_size, ctx, vocab_size)
        return lm_logits

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

def load_optimzer_and_scheduler(model,
                                dataloader,
                                epochs,
                                learning_rate,
                                adam_beta1,
                                adam_beta2,
                                adam_epsilon,
                                lr_scheduler_type,
                                warmup_steps,
                                ):
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
    return optimizer, lr_scheduler

def save_optimizer_scheduler(optimizer, scheduler, output_dir):
    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    with warnings.catch_warnings(record=True) as caught_warnings:
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    reissue_pt_warnings(caught_warnings)

def reload_optimizer_scheduler(optimizer, scheduler, output_dir, device):
    optimizer.load_state_dict(
        torch.load(os.path.join(output_dir, "optimizer.pt"), map_location=device)
    )
    with warnings.catch_warnings(record=True) as caught_warnings:
        scheduler.load_state_dict(torch.load(os.path.join(output_dir, "scheduler.pt")))
    reissue_pt_warnings(caught_warnings)
    return optimizer, scheduler

def format_time(time):
    return str(timedelta(seconds = round(time.total_seconds())))

def train(model,
        train_dataloader,
        eval_dataloader,
        optimizer,
        lr_scheduler,
        epochs,
        device,
        seed,
        max_grad_norm,
        output_dir,
        log_steps,
        save_each_x_steps,
        skip_to_epoch,
        skip_to_step,
        saved_rng_state,
        saved_epoch_rng_state):
    print("Starting training")
    training_start = datetime.now()
    model.zero_grad()
    training_loss = 0
    loss_func=CrossEntropyLoss()
    if seed is not None:
       set_seed(seed)
    steps_per_epoch = len(train_dataloader)
    time_start = datetime.now()
    for epoch in range(epochs):
        if skip_to_epoch is not None:
            if epoch < skip_to_epoch:
                continue
            else:
                skip_to_epoch = None
        if saved_epoch_rng_state is not None:
            torch.set_rng_state(saved_epoch_rng_state[0])
            torch.cuda.set_rng_state(saved_epoch_rng_state[1])
            saved_epoch_rng_state = None
        epoch_rng_state = (torch.get_rng_state(), torch.cuda.get_rng_state())
        time_start_epoch = datetime.now()
        for step, inputs in enumerate(train_dataloader):
            if skip_to_step is not None:
                if step < skip_to_step:
                    continue
                else:
                    skip_to_step = None
                    continue
            if saved_rng_state is not None:
                torch.set_rng_state(saved_rng_state[0])
                torch.cuda.set_rng_state(saved_rng_state[1])
                saved_rng_state = None
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
            training_loss += loss.item()
            if not (steps_per_epoch*epoch + step + 1)%log_steps:
                time_new = datetime.now()
                duration = time_new - time_start
                epoch_duration = time_new - time_start_epoch
                estimated_epoch = ((epoch_duration)/(step+1))*steps_per_epoch
                log_msg = f"Loss {training_loss/log_steps} Epoch {epoch + 1}/{epochs} Step {step+1}/{steps_per_epoch}"
                log_msg += f" [{format_time(epoch_duration)}<{format_time(estimated_epoch)},  {round(log_steps/duration.total_seconds(),3)}it/s]"
                time_start = time_new
                if save_each_x_steps and not((steps_per_epoch*epoch + step + 1)%save_each_x_steps):
                    val_loss, perplexity  = eval(model,eval_dataloader)
                    if val_loss is not None:
                        log_msg += f" Eval_loss {val_loss}"
                    save_model(model, output_dir, epoch, step, perplexity, optimizer, lr_scheduler, epoch_rng_state)
                    if epoch == 2:
                        print('done')
                log(log_msg,output_dir)
                training_loss = 0
        if not(save_each_x_steps):
            log_msg = f"Loss {training_loss/steps_per_epoch} Epoch {epoch + 1} [{format_time(datetime.now() - time_start_epoch)}]"
            val_loss, perplexity  = eval(model,eval_dataloader)
            if val_loss is not None:
                log_msg += f" Eval_loss {val_loss}"
            save_model(model, output_dir, epoch, step, perplexity, optimizer, lr_scheduler, epoch_rng_state)
    if save_each_x_steps:
        val_loss, perplexity  = eval(model,eval_dataloader)
        save_model(model, output_dir, epoch, step, perplexity, optimizer, lr_scheduler, epoch_rng_state)
    print("Training done in", format_time(datetime.now() - training_start))

def eval(model, dataloader):
    if dataloader is None:
        return None, None
    print("Starting evaluation")
    eval_start = datetime.now()
    model.eval()
    loss_func=CrossEntropyLoss()
    eval_loss = 0
    total_steps = len(dataloader)
    for inputs in dataloader:
        input_ids = inputs['input_ids'].to(device)
        labels = inputs['labels'].to(device)
        logits = model(input_ids)
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        eval_loss += loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).item()
    eval_loss = eval_loss/total_steps
    perplexity = round(math.exp(eval_loss),4)
    print(f"Evaluated {total_steps} batches in {format_time(datetime.now() - eval_start)}")
    return eval_loss, perplexity

def get_perplexity(loss):
    return math.exp(loss)

def save_model(model, output_dir, epoch, step = None, perplexity = None, optimizer = None, scheduler = None, epoch_rng_state = None):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = "ep{}".format(epoch)
    if step is not  None:
        checkpoint_dir += "step{}".format(step)
    if step is not  None:
        checkpoint_dir += "perp{}".format(perplexity)
    checkpoint_dir = os.path.join(output_dir,checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    state_dict = model.state_dict()
    output_model_file = os.path.join(checkpoint_dir, 'pytorch_model.bin')
    output_config_file = os.path.join(checkpoint_dir, 'config.json')
    training_state_file = os.path.join(checkpoint_dir, 'training_state.json')
    torch.save(state_dict, output_model_file)
    with open(output_config_file, 'w') as f:
        json.dump(model.config, f)
    training_state = {'epoch':epoch,'step':step}
    with open(training_state_file, 'w') as f:
        json.dump(training_state, f)
    save_optimizer_scheduler(optimizer, scheduler, checkpoint_dir)
    save_rng_state(checkpoint_dir, epoch_rng_state)
    log_msg = f'Model saved in {output_dir}: Epoch {epoch} Step {step}'
    if perplexity is not None:
        log_msg += f' Perpexity {perplexity}'
    log(log_msg, output_dir)

def save_rng_state(output_dir, epoch_rng_state):
    rng_state = torch.get_rng_state()
    rng_cuda_state = torch.cuda.get_rng_state()
    torch.save(rng_state, os.path.join(output_dir,'rng_state.pt'))
    torch.save(rng_cuda_state, os.path.join(output_dir,'rng_cuda_state.pt'))
    torch.save(epoch_rng_state[0], os.path.join(output_dir,'epoch_rng_state.pt'))
    torch.save(epoch_rng_state[1], os.path.join(output_dir,'epoch_rng_cuda_state.pt'))

def find_resume_step(saved_dir, epochs, steps_per_epoch):
    with open(os.path.join(saved_dir,'training_state.json')) as f:
        reloaded_state = json.load(f)
    skip_to_epoch = reloaded_state['epoch']
    skip_to_step = reloaded_state['step']
    if skip_to_epoch == epochs -1 and skip_to_step == steps_per_epoch - 1:
        raise Exception('Training is already completed. resume_training should be set to False')
    if skip_to_step == steps_per_epoch - 1:
        skip_to_step = None
        skip_to_epoch += 1
    msg = f'Resuming training from epoch {skip_to_epoch}'
    if skip_to_step is not None:
        msg += f', step {skip_to_step}'
    print(msg)
    return skip_to_epoch, skip_to_step

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
    
    # Dataloader
    batch_size = 2
    use_cache = True

    test_save = False
    if test_save:
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
        saved_model_path = None
        resume_training = True


        # Training settings
        do_train = True
        output_dir="test_save_output"
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
        log_steps = 50
        save_each_x_steps = 100 # Set to 0 to save at the end of each epoch (Has to be a multiple of log_steps)
        assert save_each_x_steps%log_steps == 0
    else:
        # Model 
        use_pretrained_model = False
        use_untrained_model = False
        use_saved_model = True
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
        saved_model_path = 'test_save_output/ep1step1040perp20.9532'
        resume_training = True


        # Training settings
        do_train = True
        output_dir="gpt2_test_output2"
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
        log_steps = 50
        save_each_x_steps = 100 # Set to 0 to save at the end of each epoch (Has to be a multiple of log_steps)
        assert save_each_x_steps%log_steps == 0

    if seed is not None:
        set_seed(seed) # A convenient function from the transformer library that sets all the seeds

    ### CREATING THE DATALOADER ###

    if use_custom_dataset:
        # We create a dataset that indexes all lines for a text document
        datasets = custom_dataset(train_path, validation_path, train_val_split)
    elif use_prepared_dataset:
        datasets = prepared_dataset(dataset_name, dataset_config_name)

    # Using the Tokenizer from hugging face
    tokenizer = get_pretrained_tokenizer()

    train_dataloader, eval_dataloader = get_dataloader(datasets, tokenizer, use_cache)

    if use_pretrained_model:
        model = load_pretrained_model(model_name, model_config, device)
    elif use_untrained_model:
        model = load_untrained_model(model_config, device)
    elif use_saved_model:
        model = load_saved_model(saved_model_path, device)
    optimizer, lr_scheduler = load_optimzer_and_scheduler(model,
                                train_dataloader,
                                epochs,
                                learning_rate,
                                adam_beta1,
                                adam_beta2,
                                adam_epsilon,
                                lr_scheduler_type,
                                warmup_steps,
                            )
    
    if use_saved_model and resume_training:
        optimizer, lr_scheduler = reload_optimizer_scheduler(optimizer, lr_scheduler, saved_model_path, device)
        skip_to_epoch, skip_to_step = find_resume_step(saved_model_path, epochs, len(train_dataloader))
        saved_rng_state = (torch.load(os.path.join(saved_model_path,'rng_state.pt')),
                        torch.load(os.path.join(saved_model_path,'rng_cuda_state.pt')))
        saved_epoch_rng_state = (torch.load(os.path.join(saved_model_path,'epoch_rng_state.pt')),
                        torch.load(os.path.join(saved_model_path,'epoch_rng_cuda_state.pt')))
    else:
        skip_to_epoch, skip_to_step, saved_rng_state, saved_epoch_rng_state = None, None, None, None

    if do_train:
        train(model,
            train_dataloader,
            eval_dataloader,
            optimizer,
            lr_scheduler,
            epochs,
            device,
            seed,
            max_grad_norm,
            output_dir,
            log_steps,
            save_each_x_steps,
            skip_to_epoch,
            skip_to_step,
            saved_rng_state,
            saved_epoch_rng_state)
    else:
        eval_loss, perplexity = eval(model, eval_dataloader)
        print("Eval loss:",eval_loss)
        print("Perplexity:",perplexity)


