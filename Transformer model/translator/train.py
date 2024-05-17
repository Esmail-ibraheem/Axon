import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from model import TransformerModel
from dataset import BilingualDataset, casual_mask
from configuration import Get_configuration, Get_weights_file_path, latest_weights_file_path

from datasets import load_dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from pathlib import Path 

import warnings
from tqdm import tqdm
import os 

def greedy_search(model, source, source_mask, source_tokenizer, target_tokenizer, max_len, device):
    sos_idx = target_tokenizer.token_to_id('[SOS]')
    eos_idx = target_tokenizer.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)

    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = casual_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token (get the token with the maximum probabilty)
        prob = model.linear(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, source_tokenizer, target_tokenizer, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0
    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) 
            encoder_mask = batch["encoder_input_mask"].to(device) 

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_search(model, encoder_input, encoder_mask, source_tokenizer, target_tokenizer, max_len, device)

            source_text = batch["target_text"][0]
            target_text = batch["target_text"][0]
            model_out_text = target_tokenizer.decode(model_out.detach().cpu().numpy())

            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                break

def Get_All_Sentences(dataset, language):
    for lang in dataset:
        yield lang['translation'][language]

def Build_Tokenizer(configuration, dataset, language):
    tokenizer_path = Path(configuration['tokenizer_file'].format(language))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token= "[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency = 2)
        tokenizer.train_from_iterator(Get_All_Sentences(dataset, language), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer 

def Get_dataset(configuration):
    dataset_Raw = load_dataset(f"{configuration['datasource']}", f"{configuration['source_language']}-{configuration['target_language']}", split="train")

    source_tokenizer = Build_Tokenizer(configuration, dataset_Raw, configuration['source_language'])
    target_tokenizer = Build_Tokenizer(configuration, dataset_Raw, configuration['target_language'])

    train_dataset_Size = int(0.9 * len(dataset_Raw)) 
    validation_dataset_Size = len(dataset_Raw) - train_dataset_Size

    train_dataset_Raw, validation_dataset_Raw = random_split(dataset_Raw, [train_dataset_Size, validation_dataset_Size])

    train_dataset = BilingualDataset(train_dataset_Raw, source_tokenizer, target_tokenizer, configuration['source_language'], configuration['target_language'], configuration['sequence_length'])
    validation_dataset = BilingualDataset(validation_dataset_Raw, source_tokenizer, target_tokenizer, configuration['source_language'], configuration['target_language'], configuration['sequence_length'])

    maximum_source_sequence_length = 0 
    maximum_target_sequence_length = 0 

    for item in dataset_Raw:
        source_id = source_tokenizer.encode(item['translation'][configuration['source_language']]).ids
        target_id = target_tokenizer.encode(item['translation'][configuration['target_language']]).ids
        maximum_source_sequence_length = max(maximum_source_sequence_length, len(source_id))
        maximum_target_sequence_length = max(maximum_target_sequence_length, len(target_id))

    print(f"maximum_source_sequence_length : {maximum_source_sequence_length}")
    print(f"maximum_target_sequence_length: {maximum_target_sequence_length}")

    train_dataLoader = DataLoader(train_dataset, batch_size= configuration['batch_size'], shuffle=True)
    validation_dataLoader = DataLoader(validation_dataset, batch_size= 1, shuffle=True)
  
    return train_dataLoader, validation_dataLoader, source_tokenizer, target_tokenizer

def Get_model(configuration, source_vocab_size, target_vocab_size):
    model = TransformerModel(source_vocab_size, target_vocab_size, configuration['sequence_length'], configuration['sequence_length'], configuration['d_model'])
    return model 

def train_model(configuration):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)

    Path(f"{configuration['datasource']}_{configuration['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataLoader, validation_dataLoader, source_tokenizer, target_tokenizer = Get_dataset(configuration)
    model = Get_model(configuration, source_tokenizer.get_vocab_size(), target_tokenizer.get_vocab_size()).to(device)

    writer = SummaryWriter(configuration['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=configuration['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    preload = configuration['preload']
    model_filename = latest_weights_file_path(configuration) if preload == 'latest' else Get_weights_file_path(configuration, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=source_tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, configuration['num_epochs']):
        torch.cuda.empty_cache()
        batch_iterator = tqdm(train_dataLoader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            model.train()
            encoder_input = batch['encoder_input'].to(device) 
            decoder_input = batch['decoder_input'].to(device) 
            encoder_mask = batch['encoder_input_mask'].to(device) 
            decoder_mask = batch['encoder_input_mask'].to(device) 

            encoder_output = model.encode(encoder_input, encoder_mask) 
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) 
            proj_output = model.linear(decoder_output) 

            Target = batch['Target'].to(device) 

            loss = loss_fn(proj_output.view(-1, target_tokenizer.get_vocab_size()), Target.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # run_validation(model, validation_dataLoader, source_tokenizer, target_tokenizer, configuration['sequence_length'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

            global_step += 1


        model_filename = Get_weights_file_path(configuration, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)



if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    configuration = Get_configuration()
    train_model(configuration)