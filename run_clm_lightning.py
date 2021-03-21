import os
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from transformers import AutoModelForCausalLM
from transformers.activations import gelu_new
from transformers.optimization import AdamW
from torch.nn import CrossEntropyLoss

from get_dataset import custom_dataset, prepared_dataset, get_pretrained_tokenizer, get_dataloader

import math
import json
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false" # This need to be set or the hugging face tokenizer will throw an error if num_workers is not 1

class Conv1D(nn.Module):
    def __init__(self, output_dim, input_dim):
        super().__init__()
        self.output_dim = output_dim
        weight = torch.empty(input_dim, output_dim)
        nn.init.normal_(weight, std=0.02)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.output_dim,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x

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
        self.register_buffer("masked_bias", torch.tensor(-1e4)) 
        self.n_head = n_head
        self.split_size = n_embed
        self.c_attn = Conv1D(3 * n_embed, n_embed)
        self.c_proj = Conv1D(n_embed, n_embed)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
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
        query_key_value = self.c_attn(hidden_states)
        query, key, value = query_key_value.split(self.split_size, dim=2)

        query = self.split_heads(query)      #   Reshaping into
        key = self.split_heads(key, is_key=True)  #   multiple heads
        value = self.split_heads(value)      #
        output = self._attn(query, key, value)
        output = self.merge_heads(output)   # Reshaping into a single head
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
        attn_output = self.attn(
            self.ln_1(hidden_states),
        )
        hidden_states = attn_output + hidden_states
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
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
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
        batch_size, context_size = input_ids.size() # context_size is usually 1024
        position_ids = torch.arange(context_size, dtype=torch.long).type_as(input_ids)
        position_ids = position_ids.unsqueeze(0).view(-1, context_size)
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        output_shape = (batch_size, context_size, hidden_states.size(-1),)
        for i, block in enumerate(self.h):
            hidden_states = block(hidden_states)
        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(*output_shape)
        return hidden_states

class GPT2LMHeadModel(pl.LightningModule):

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
        self.loss_func = CrossEntropyLoss()

    def init_weights(self):
        self.apply(self._init_weights)
        self.lm_head.weight = self.transformer.wte.weight

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        hidden_states = self.transformer(x)
        logits = self.lm_head(hidden_states)
        return logits

    def training_step(self, batch, _):
        input_ids = batch['input_ids']
        labels = batch['labels']
        hidden_states = self.transformer(input_ids)
        logits = self.lm_head(hidden_states)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = self.loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        self.log('train_loss', loss)
        return {'loss':loss}

    def validation_step(self, batch, _):
        input_ids = batch['input_ids']
        labels = batch['labels']
        hidden_states = self.transformer(input_ids)
        logits = self.lm_head(hidden_states)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = self.loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        perplexity = math.exp(loss)
        return {'loss':loss,'perplexity':perplexity}

    def configure_optimizers(self):
        adam_beta1 = settings['adam_beta1']
        adam_beta2 = settings['adam_beta2']
        adam_epsilon = settings['adam_epsilon']
        learning_rate = settings['learning_rate']
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, betas = (adam_beta1, adam_beta2), eps = adam_epsilon, lr = learning_rate)
        return optimizer

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

def get_config(model_type):
    gpt2_configs = {"distilgpt2":{'n_embed':768, 'n_layer':6,'n_head':12},
        "gpt2":{'n_embed':768,'n_layer':12,'n_head':12},
        "gpt2-medium":{'n_embed':1024,'n_layer':24,'n_head':16},
        "gpt2-large":{'n_embed':1280,'n_layer':36,'n_head':20},
        "gpt2-xl":{'n_embed':1600,'n_layer':48,'n_head':25} 
    }
    model_config = gpt2_configs[model_type]
    model_config.update({'n_context':1024,'vocab_size':50257,'dropout':0.1})
    return model_config

def load_pretrained_model(model_name, model_type):
    model = GPT2LMHeadModel(**get_config(model_type))
    home = os.path.expanduser("~")
    cache_path = os.path.join(home, '.cache/transformers_local_model')
    weights_path = download_weights(model_name, cache_path)
    weights = torch.load(weights_path)
    model.load_state_dict(weights)
    return model

def load_untrained_model(model_type):
    model = GPT2LMHeadModel(**get_config(model_type))
    return model

def load_saved_model(model_path):
    config_path = os.path.join(model_path, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    model = GPT2LMHeadModel(**config)
    weights_path = os.path.join(model_path, 'pytorch_model.bin')
    weights = torch.load(weights_path)
    model.load_state_dict(weights)
    return model

if __name__ == '__main__':
    
    settings = {
        'use_custom_dataset':False,
        'custom_dataset':{
            'train_path':None,
            'validation_path':None,
            'validation_split_percentage':10,
        },
        'use_prepared_dataset':True,
        'prepared_dataset':{
            'dataset_name':'wikitext',
            'dataset_config_name':'wikitext-2-raw-v1',
        },
        'use_cache':True,
        'num_workers':1,
        'adam_beta1':0.9,
        'adam_beta2':0.999,
        'adam_epsilon':1e-08,
        'learning_rate':5e-05,
        'batch_size':2,
        'use_pretrained_model':True,
        'pretrained_model':{
            'model_name':'gpt2',
            'model_type':'gpt2',
        },
        'use_untrained_model':False,
        'untrained_model':{
            'model_type':'gpt2',
        },
        'use_saved_model':False,
        'saved_model':{
            'saved_model_path':None,
        },
        'do_train':True,
        'train_settings':{
            'auto_lr_find':False,
            'gradient_clip_val':1.0,
            'fast_dev_run':False,
            'log_every_n_steps':50,
            'precision':32, # set to 16 for fp16
            'max_epochs':3,
            'val_check_interval':0, # Use float to check within a training epoch, use int to check every n steps (batches).
            'gpus':1
        },
        'test_settings':{
            'fast_dev_run':False,
            'precision':32, # set to 16 for fp16
        }
    }

    #### 
    ####

    if settings['use_custom_dataset']:
        datasets = custom_dataset(**settings['custom_dataset'])
    elif settings['use_prepared_dataset']:
        datasets = prepared_dataset(**settings['prepared_dataset'])

    # Using the Tokenizer from hugging face
    tokenizer = get_pretrained_tokenizer()

    train_dataloader, eval_dataloader = get_dataloader(datasets, settings['use_cache'], settings['batch_size'],settings['num_workers'])

    if settings['use_pretrained_model']:
        model = load_pretrained_model(**settings['pretrained_model'])
    elif settings['use_untrained_model']:
        model = load_untrained_model(**settings['untrained_model'])
    elif settings['use_saved_model']:
        model = load_saved_model(**settings['saved_model'])

if settings['do_train']:
    start_time = datetime.now()
    trainer = pl.Trainer(**settings['train_settings'])
    trainer.fit(model, train_dataloader)
    print('time', datetime.now()-start_time)
else:
    trainer = pl.Trainer(**settings['train_settings'])
    trainer.test(model, eval_dataloader)