import sacrebleu
#from transformers import T5ForConditionalGeneration
import sys
sys.path.append("../transformers/src/")
from transformers.modeling_t5 import T5ForConditionalGeneration, T5Config
from transformers.modeling_bart import BartForConditionalGeneration, BartConfig
from torch.utils.data import DataLoader
from data import create_adapted_tokenizer, MyDataset
import pytorch_lightning as pl
import config
import torch


def fix_accent_breaks(text, added_tokens):
    """
    A ideia é fazer a junção de letras com acento de volta para frases na validação.
    Isso serve para melhorar o BLEU.

    Args
        text: texto que terá acentuação corrigida

    Returns:
        Texto completo com acentuação corrigida

    """
    words = text.split(" ")
    out_words = []
    merge_pos = [idx for idx, dat in enumerate(words) if dat in added_tokens]
    for pos in sorted(merge_pos, reverse=True):
        if pos == 0:
            new_word = words[pos] + words[pos + 1]
            for i in range(2): words.pop
            words.pop(pos + 1)
            words.pop(pos)
            words.insert(pos, new_word)
        elif pos == len(words) - 1:
            new_word = words[pos - 1] + words[pos]
            words.pop(pos)
            words.pop(pos - 1)
            words.insert(pos - 1, new_word)
        else:
            new_word = words[pos - 1] + words[pos] + words[pos + 1]
            words.pop(pos + 1)
            words.pop(pos)
            words.pop(pos - 1)
            words.insert(pos - 1, new_word)

    return " ".join(words)


class MyModel(pl.LightningModule):

    def __init__(self, tokenizer, train_dataloader, val_dataloader,
                 test_dataloader, learning_rate, added_tokens, target_max_length=32):
        super(MyModel, self).__init__()

        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._test_dataloader = test_dataloader
        #model_name = config.get_model_name()
        self.is_ptt5 = True #config.get_ptt5_checker()
        #self.model = T5ForConditionalGeneration.from_pretrained(model_name)

        train_type = config.items['training']['type']
        model_type = config.items['model']['type']

        if 't5' in model_type:
            if train_type == 'fine-tune':
                self.model = T5ForConditionalGeneration.from_pretrained(model_type, return_dict=True)
            else:
                pad_token_id = tokenizer.pad_token_id
                model_spec = T5Config(
                  decoder_start_token_id=pad_token_id
                ).from_pretrained(model_type) # creating the model

                self.model = T5ForConditionalGeneration(model_spec)
        elif 'bart' in model_type:
            print(train_type)
            if train_type == 'fine-tune':
                self.model = BartForConditionalGeneration.from_pretrained(model_type, return_dict=True)
            else:
                model_spec = BartConfig().from_pretrained(model_type)
                self.model = BartForConditionalGeneration(model_spec)

        self.added_tokens = added_tokens
        self.training = False
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.target_max_length = target_max_length

    def forward(self, source_token_ids, source_mask, target_token_ids=None,
                target_mask=None, training=False):

        if training:
            #target_token_ids[target_token_ids == self.tokenizer.pad_token_id] = -100

            #loss = self.model(input_ids=source_token_ids, attention_mask=source_mask,
            #                  lm_labels=target_token_ids)
            loss = self.model(input_ids=source_token_ids, labels=target_token_ids)
            return loss[0]
        else:
            # gerador de tokens de saída
            predicted_token_ids = self.model.generate(input_ids=source_token_ids, max_length=self.target_max_length)
            return predicted_token_ids

    def training_step(self, batch, batch_nb):
        # batch
        source_token_ids, source_mask, target_token_ids, target_mask, _, _ = batch

        # fwd
        loss = self(
            source_token_ids, source_mask, target_token_ids, target_mask, training=True)

        # logs
        tensorboard_logs = {'train_loss': loss}
        progress_bar = {'gpu_usage': 100}
        return {'loss': loss, 'log': tensorboard_logs,
                'progress_bar': progress_bar}

    def validation_step(self, batch, batch_nb):
        source_token_ids, source_mask, target_token_ids, target_mask, source, refs = batch
        predict = self(source_token_ids, source_mask).permute(0, 1)
        #print(predict)
        if self.is_ptt5:
            sys = [self.tokenizer.decode(tokens) for tokens in predict]
            #sys = []
            #for tokens in predict:
            #    print(tokens)
            #    sys.append(self.tokenizer.decode(tokens))
        else:
            sys = [fix_accent_breaks(self.tokenizer.decode(tokens), self.added_tokens) for tokens in predict]

        progress_bar = {'gpu_usage': 100}
        return {'pred': sys, 'target': refs, 'progress_bar': progress_bar}

    def test_step(self, batch, batch_nb):
        source_token_ids, source_mask, target_token_ids, target_mask, source, refs = batch
        predict = self(source_token_ids, source_mask).permute(0, 1)

        if self.is_ptt5:
            sys = [self.tokenizer.decode(tokens) for tokens in predict]
        else:
            sys = [fix_accent_breaks(self.tokenizer.decode(tokens), self.added_tokens) for tokens in predict]

        progress_bar = {'gpu_usage': 100}
        return {'pred': sys, 'target': refs, 'progress_bar': progress_bar}

    def validation_epoch_end(self, outputs):
        trues = sum([list(x['target']) for x in outputs], [])
        preds = sum([list(x['pred']) for x in outputs], [])

        bleu = sacrebleu.corpus_bleu(trues, [preds]).score
        test_dict = {'val_bleu': bleu}

        return {'val_bleu': bleu, 'log': test_dict, 'progress_bar': test_dict}

    def test_epoch_end(self, outputs):
        trues = sum([list(x['target']) for x in outputs], [])
        preds = sum([list(x['pred']) for x in outputs], [])

        bleu = sacrebleu.corpus_bleu(trues, [preds]).score
        test_dict = {'test_bleu': bleu}

        return {'test_bleu': bleu, 'log': test_dict, 'progress_bar': test_dict}

    def configure_optimizers(self):
        return torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.learning_rate, eps=1e-05)

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader


def create_model(x_train, x_val, x_test):
    model_name = config.get_model_name()
    target_max_length = config.get_target_max_length()
    source_max_length = config.get_source_max_length()
    tokenizer, added_tokens = create_adapted_tokenizer(model_name)

    dataset_train = MyDataset(text_pairs=x_train,
                              tokenizer=tokenizer,
                              source_max_length=source_max_length,
                              target_max_length=target_max_length)

    dataset_val = MyDataset(text_pairs=x_val,
                            tokenizer=tokenizer,
                            source_max_length=source_max_length,
                            target_max_length=target_max_length)

    dataset_test = MyDataset(text_pairs=x_test,
                             tokenizer=tokenizer,
                             source_max_length=source_max_length,
                             target_max_length=target_max_length)

    train_dataloader = DataLoader(dataset_train, batch_size=config.get_batch_size(),
                                  shuffle=False, num_workers=4)

    val_dataloader = DataLoader(dataset_val, batch_size=config.get_batch_size(), shuffle=False,
                                num_workers=4)

    test_dataloader = DataLoader(dataset_test, batch_size=config.get_batch_size(),
                                 shuffle=False, num_workers=4)

    lr = config.get_learning_rate()

    model = MyModel(tokenizer=tokenizer,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    test_dataloader=test_dataloader,
                    learning_rate=lr,
                    added_tokens=added_tokens,
                    target_max_length=target_max_length)

    return model
