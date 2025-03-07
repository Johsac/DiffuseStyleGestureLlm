import glob
import os
import sys
import numpy as np
import torch
import argparse
from torch.utils import tensorboard
import torchvision
import torch.nn as nn
[sys.path.append(i) for i in ['.', '..','../', '../../process','../../process/WavLM']]
from process_TWH_bvh import wavlm_init, load_audio
from transformers import AutoTokenizer, AutoModelForCausalLM

# Deshabilitar advertencias de transformaciones beta de torchvision
torchvision.disable_beta_transforms_warning()

class AttentionAutoencoder(nn.Module):
    def __init__(self, input_dim=3072, compressed_dim=300, num_heads=8):
        super(AttentionAutoencoder, self).__init__()
        # Encoder
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.encoder_fc1 = nn.Linear(input_dim, 2048)
        self.encoder_fc2 = nn.Linear(2048, 1024)
        self.encoder_fc3 = nn.Linear(1024, 512)
        self.encoder_fc4 = nn.Linear(512, compressed_dim)
        
        # Decoder
        self.decoder_fc1 = nn.Linear(compressed_dim, 512)
        self.decoder_fc2 = nn.Linear(512, 1024)
        self.decoder_fc3 = nn.Linear(1024, 2048)
        self.decoder_fc4 = nn.Linear(2048, input_dim)
        
    def forward(self, x):
        x = x.unsqueeze(0)  # [1, seq_len, input_dim]
        attn_output, _ = self.attention(x, x, x)
        attn_output = attn_output.squeeze(0)  # [seq_len, input_dim]
        x = torch.relu(self.encoder_fc1(attn_output))
        x = torch.relu(self.encoder_fc2(x))
        x = torch.relu(self.encoder_fc3(x))
        compressed = self.encoder_fc4(x)
        
        x = torch.relu(self.decoder_fc1(compressed))
        x = torch.relu(self.decoder_fc2(x))
        x = torch.relu(self.decoder_fc3(x))
        reconstructed = self.decoder_fc4(x)
        
        return compressed, reconstructed

def load_tsv_unclipped(tsvfile):
    sentence = []
    with open(tsvfile, "r") as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip().split("\t")
            if len(line) == 3:
                start, end, raw_word = line
                start = float(start)
                end = float(end)
                sentence.append([start, end, raw_word])
    return sentence

def load_tsv(tsvpath, clip_length, tokenizer, model, device, encoder=None):
    sentence = load_tsv_unclipped(tsvpath)
    word_embeddings = []
    for _, _, raw_word in sentence:
        inputs = tokenizer(raw_word, return_tensors="pt", add_special_tokens=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            embedding = outputs.hidden_states[-1][0, -1, :]  # Último token del input
            word_embeddings.append(embedding)
    
    if len(word_embeddings) != len(sentence):
        print(f"Advertencia: mismatch entre palabras ({len(sentence)}) y embeddings ({len(word_embeddings)}) en {tsvpath}")
        min_len = min(len(word_embeddings), len(sentence))
        word_embeddings = word_embeddings[:min_len]
        sentence = sentence[:min_len]
    
    textfeatures = np.zeros([clip_length, 3074 if encoder is None else 302])  # 3072 + 2
    textfeatures[:, -1] = 1
    for wi, (start, end, raw_word) in enumerate(sentence):
        has_laughter = "#" in raw_word
        start_frame = int(start * 30)
        end_frame = int(end * 30)
        textfeatures[start_frame:end_frame, -1] = 0
        vector = word_embeddings[wi].cpu().numpy()
        if encoder is not None:
            reduced_vector, _ = encoder(torch.FloatTensor(vector).unsqueeze(0).to(device))
            reduced_vector = reduced_vector.squeeze(0).detach().cpu().numpy()
            textfeatures[start_frame:end_frame, :300] = reduced_vector
        else:
            textfeatures[start_frame:end_frame, :3072] = vector
        textfeatures[start_frame:end_frame, -2] = has_laughter
    return textfeatures

def train_autoencoder(encoder, train_files, val_files, device, args):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate)
    
    loss_file_path = os.path.join(args.checkpoint_path, args.loss_file)
    with open(loss_file_path, 'w') as f:
        f.write("Epoch,Train_Loss,Val_Loss\n")
    
    for epoch in range(args.num_epochs):
        encoder.train()
        train_loss = 0.0
        for file in train_files:
            data = np.load(file)
            text_embedding = torch.FloatTensor(data[:, :3072]).to(device)
            optimizer.zero_grad()
            _, reconstructed = encoder(text_embedding)
            loss = criterion(reconstructed, text_embedding)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_files)
        
        encoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for file in val_files:
                data = np.load(file)
                text_embedding = torch.FloatTensor(data[:, :3072]).to(device)
                _, reconstructed = encoder(text_embedding)
                loss = criterion(reconstructed, text_embedding)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_files)
        
        with open(loss_file_path, 'a') as f:
            f.write(f'{epoch+1},{avg_train_loss:.4f},{avg_val_loss:.4f}\n')
        
        print(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        # Guardar checkpoint cada 3 épocas o al final
        if (epoch + 1) % 3 == 0 or (epoch + 1) == args.num_epochs:
            checkpoint_path = os.path.join(args.checkpoint_path, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(encoder.state_dict(), checkpoint_path)
            print(f"Checkpoint saved in {checkpoint_path}")

def reduce_dimension_with_checkpoint(args, device, encoder):
    # Reducir dimensiones de los .npy preprocesados
    for split, npy_path, output_path in [
        ('train', args.train_npy_path, os.path.join(args.output_text_audio_path, 'reduction')),
        ('val', args.val_npy_path, os.path.join(args.val_npy_path, 'reduction'))
    ]:
        npy_files = glob.glob(os.path.join(npy_path, '*.npy'))
        os.makedirs(output_path, exist_ok=True)
        for npy_file in npy_files:
            data = np.load(npy_file)
            text_embedding = torch.FloatTensor(data[:, :3072]).to(device)
            with torch.no_grad():
                reduced_text, _ = encoder(text_embedding)
                reduced_text = reduced_text.cpu().numpy()
            # Mantener audio, risa y ausencias sin cambios
            reduced_data = np.concatenate((reduced_text, data[:, 3072:]), axis=1)
            reduced_filename = os.path.basename(npy_file)
            reduced_output_file = os.path.join(output_path, reduced_filename)
            np.save(reduced_output_file, reduced_data)
            print(f"Reduced embeddings saved in: {reduced_output_file} ({split})")

def main(args):
    print(f"Available GPU: {torch.cuda.is_available()}")
    device_name = 'cuda:0'
    device = torch.device(device_name)

    # Liberar memoria GPU antes de empezar
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.train_autoencoder:
        # Entrenar el AttentionAutoencoder
        encoder = AttentionAutoencoder().to(device)
        train_files = glob.glob(os.path.join(args.train_npy_path, '*.npy'))
        val_files = glob.glob(os.path.join(args.val_npy_path, '*.npy'))
        print(f"Found {len(train_files)} training files and {len(val_files)} validation files.\n")
        train_autoencoder(encoder, train_files, val_files, device, args)
    elif args.reduce_dimension:
        # Reducir dimensión con el modelo entrenado
        train_npy_files = glob.glob(os.path.join(args.train_npy_path, '*.npy'))
        val_npy_files = glob.glob(os.path.join(args.val_npy_path, '*.npy'))
        if train_npy_files and val_npy_files:
            # Usar .npy preprocesados directamente
            encoder = AttentionAutoencoder().to(device)
            checkpoint_file = os.path.join(args.checkpoint_path, args.checkpoint_file)
            encoder.load_state_dict(torch.load(checkpoint_file, map_location=device))
            encoder.eval()
            print(f"AttentionAutoencoder loaded from checkpoint: {checkpoint_file}")
            reduce_dimension_with_checkpoint(args, device, encoder)
        else:
            # Procesar desde .tsv y .wav con reducción
            wavlm_model, cfg = wavlm_init(args.wavlm_path, device)
            print(f"** The WavLM model was successfully imported from: {args.wavlm_path} **\n")
            tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path)
            model = AutoModelForCausalLM.from_pretrained(args.llm_model_path, device_map="auto").to(device)
            print(f"** The LLM model was successfully imported from: {args.llm_model_path} **\n")
            encoder = AttentionAutoencoder().to(device)
            checkpoint_file = os.path.join(args.checkpoint_path, args.checkpoint_file)
            encoder.load_state_dict(torch.load(checkpoint_file, map_location=device))
            encoder.eval()
            print(f"AttentionAutoencoder loaded from checkpoint: {checkpoint_file}")
            for split, wav_path, txt_path, output_path in [
                ('train', args.wav_path, args.txt_path, os.path.join(args.output_text_audio_path, 'reduction')),
                ('val', args.val_wav_path, args.val_txt_path, os.path.join(args.val_npy_path, 'reduction'))
            ]:
                wav_files = glob.glob(os.path.join(wav_path, '*.wav'))
                os.makedirs(output_path, exist_ok=True)
                for wav_file in wav_files:
                    combined_filename = os.path.basename(wav_file).replace('.wav', '_text_audio_reduced.npy')
                    text_audio_output_file = os.path.join(output_path, combined_filename)
                    if os.path.exists(text_audio_output_file):
                        print(f"File already processed: {combined_filename} ({split}), skipping.\n")
                        continue
                    wav = load_audio(wav_file, wavlm_model, cfg)
                    clip_len = wav.shape[0]
                    filename = os.path.basename(wav_file).replace('.wav', '.tsv')
                    txt_file = os.path.join(txt_path, filename)
                    if os.path.exists(txt_file):
                        tsv = load_tsv(txt_file, clip_len, tokenizer, model, device, encoder)
                        textaudio = np.concatenate((wav, tsv), axis=-1)
                        print(f"Dimension of embeddings (text + audio): {textaudio.shape}, file: {filename} ({split})")
                        np.save(text_audio_output_file, textaudio)
                        print(f"Embeddings saved in: {text_audio_output_file} ({split})\n")
    else:
        # Procesar datos WAV/TSV sin reducción
        wavlm_model, cfg = wavlm_init(args.wavlm_path, device)
        print(f"** The WavLM model was successfully imported from: {args.wavlm_path} **\n")
        tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path)
        model = AutoModelForCausalLM.from_pretrained(args.llm_model_path, device_map="auto").to(device)
        print(f"** The LLM model was successfully imported from: {args.llm_model_path} **\n")
        for split, wav_path, txt_path, output_path in [
            ('train', args.wav_path, args.txt_path, args.output_text_audio_path),
            ('val', args.val_wav_path, args.val_txt_path, args.val_npy_path)
        ]:
            wav_files = glob.glob(os.path.join(wav_path, '*.wav'))
            done_files = set(os.listdir(output_path))
            for wav_file in wav_files:
                combined_filename = os.path.basename(wav_file).replace('.wav', '_text_audio.npy')
                text_audio_output_file = os.path.join(output_path, combined_filename)
                if combined_filename in done_files:
                    print(f"File already processed: {combined_filename} ({split}), skipping.\n")
                    continue
                wav = load_audio(wav_file, wavlm_model, cfg)
                clip_len = wav.shape[0]
                filename = os.path.basename(wav_file).replace('.wav', '.tsv')
                txt_file = os.path.join(txt_path, filename)
                if os.path.exists(txt_file):
                    tsv = load_tsv(txt_file, clip_len, tokenizer, model, device)
                    textaudio = np.concatenate((wav, tsv), axis=-1)
                    print(f"Dimension of embeddings (text + audio): {textaudio.shape}, file: {filename} ({split})")
                    np.save(text_audio_output_file, textaudio)
                    print(f"Embeddings saved in: {text_audio_output_file} ({split})\n")

# def main(args):
#     print(f'Available GPU : {torch.cuda.is_available()}')
#     device_name = 'cuda:0'
#     device = torch.device(device_name) 

#     # Liberar memoria GPU antes de empezar
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()

#     # Si entrenamos el autoencoder, no cargamos LLM ni WavLM
#     if args.train_autoencoder:
#         encoder = AttentionAutoencoder().to(device)
#         train_files = glob.glob(os.path.join(args.train_npy_path, '*.npy'))
#         val_files = glob.glob(os.path.join(args.val_npy_path, '*.npy'))
#         print(f"Found {len(train_files)} training files and {len(val_files)} validation files. \n")
#         train_autoencoder(encoder, train_files, val_files, device, args)
#     else:
#         # Cargar modelos solo si vamos a procesar datos WAV/TSV
#         wavlm_model, cfg = wavlm_init(args.wavlm_path, device)
#         print(f'** The WavLM model was successfully imported from: {args.wavlm_path} ** \n')
#         tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path)
#         model = AutoModelForCausalLM.from_pretrained(args.llm_model_path, device_map="auto").to(device)
#         print(f'** The LLM model was successfully imported from: {args.llm_model_path} **\n') ########
        
#         for split, wav_path, txt_path, output_path in [
#             ('train', args.wav_path, args.txt_path, args.output_text_audio_path),
#             ('val', args.val_wav_path, args.val_txt_path, args.val_npy_path)
#         ]:
#             wav_files = glob.glob(os.path.join(wav_path, '*.wav'))
#             done_files = set(os.listdir(output_path))
#             for wav_file in wav_files:
#                 combined_filename = os.path.basename(wav_file).replace('.wav', '_text_audio.npy')
#                 text_audio_output_file = os.path.join(output_path, combined_filename)
#                 if combined_filename in done_files:
#                     print(f'File already processed: {combined_filename} ({split}), skipping.  \n')
#                     continue
#                 wav = load_audio(wav_file, wavlm_model, cfg)
#                 clip_len = wav.shape[0]
#                 filename = os.path.basename(wav_file).replace('.wav', '.tsv')
#                 txt_file = os.path.join(txt_path, filename)
#                 if os.path.exists(txt_file):
#                     tsv = load_tsv(txt_file, clip_len, tokenizer, model, device)
#                     textaudio = np.concatenate((wav, tsv), axis=-1)
#                     print(f'Dimension embeddings (texto + audio): {textaudio.shape}, file: {filename} ({split})')
#                     np.save(text_audio_output_file, textaudio)
#                     print(f'Embeddings saved in: {text_audio_output_file} ({split}) \n')


# def main(args):
#     print(f'Available GPU : {torch.cuda.is_available()}')
#     device_name = 'cuda:0'
#     device = torch.device(device_name)  

#     # Cargar modelo WavLM
#     wavlm_model, cfg = wavlm_init(args.wavlm_path, device)
#     print(f'** The WavLM model was successfully imported from: {args.wavlm_path} ** \n')

#     # Cargar modelo LLM y tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path)
#     model = AutoModelForCausalLM.from_pretrained(args.llm_model_path, device_map="auto")
#     print(f'** The LLM model was successfully imported from: {args.llm_model_path} **\n') ########

#     # Crear directorios de salida si no existen
#     for path in [args.output_text_audio_path, args.val_npy_path, args.checkpoint_path]:
#         os.makedirs(path, exist_ok=True)
#         print(f"Directory saved in: {path}")

#     # Inicializar encoder solo si se va a entrenar
#     encoder = None
#     if args.train_autoencoder:
#         encoder = AttentionAutoencoder().to(device)
#         # Cargar archivos .npy para entrenamiento y validación
#         train_files = glob.glob(os.path.join(args.train_npy_path, '*.npy'))
#         val_files = glob.glob(os.path.join(args.val_npy_path, '*.npy'))
#         print(f"Found {len(train_files)} training files and {len(val_files)} validation files. \n")
#         # Entrenar el autoencoder
#         train_autoencoder(encoder, train_files, val_files, device, args)
#     else:
#         # Procesar archivos WAV y TSV para generar .npy
#         for split, wav_path, txt_path, output_path in [
#             ('train', args.wav_path, args.txt_path, args.output_text_audio_path),
#             ('val', args.val_wav_path, args.val_txt_path, args.val_npy_path)
#         ]:
#             wav_files = glob.glob(os.path.join(wav_path, '*.wav'))
#             done_files = set(os.listdir(output_path))
#             for wav_file in wav_files:
#                 combined_filename = os.path.basename(wav_file).replace('.wav', '_text_audio.npy')
#                 text_audio_output_file = os.path.join(output_path, combined_filename)
#                 if combined_filename in done_files:
#                     print(f'File already processed: {combined_filename} ({split}), skipping.  \n')
#                     continue
#                 wav = load_audio(wav_file, wavlm_model, cfg)
#                 clip_len = wav.shape[0]
#                 filename = os.path.basename(wav_file).replace('.wav', '.tsv')
#                 txt_file = os.path.join(txt_path, filename)
#                 if os.path.exists(txt_file):
#                     tsv = load_tsv(txt_file, clip_len, tokenizer, model, device, encoder)
#                     textaudio = np.concatenate((wav, tsv), axis=-1)
#                     print(f'Dimension embeddings (texto + audio): {textaudio.shape}, file: {filename} ({split})')
#                     np.save(text_audio_output_file, textaudio)
#                     print(f'Embeddings saved in: {text_audio_output_file} ({split}) \n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Procesar embeddings y entrenar AttentionAutoencoder.')
    parser.add_argument('--wavlm_path', type=str, default='../../process/WavLM/WavLM-Large.pt', 
                        help='Ruta al modelo WavLM.')
    parser.add_argument('--llm_model_path', type=str, default='../download_llm/llama-3.2-3b-instruct', 
                        help='Ruta al modelo LLM.')
    parser.add_argument('--wav_path', type=str, default='../../data/trn/main-agent/wav/', 
                        help='Ruta a los WAV de entrenamiento.')
    parser.add_argument('--txt_path', type=str, default='../../data/trn/main-agent/tsv/', 
                        help='Ruta a los TSV de entrenamiento.')
    parser.add_argument('--val_wav_path', type=str, default='../../data/val/main-agent/wav/', 
                        help='Ruta a los WAV de validación.')
    parser.add_argument('--val_txt_path', type=str, default='../../data/val/main-agent/tsv/', 
                        help='Ruta a los TSV de validación.')
    parser.add_argument('--output_text_audio_path', type=str, 
                        default='../../data/trn/main-agent/text_audio/', 
                        help='Directorio de salida para archivos .npy de entrenamiento.')
    parser.add_argument('--val_npy_path', type=str, 
                        default='../../data/val/main-agent/text_audio/', 
                        help='Directorio de salida para archivos .npy de validación.')
    parser.add_argument('--checkpoint_path', type=str, default='./training/checkpoints/', 
                        help='Directorio para guardar los checkpoints del autoencoder.')
    parser.add_argument('--checkpoint_file', type=str, default='checkpoint_epoch_10.pth', 
                    help='Archivo de checkpoint a usar para la reducción de dimensiones.')
    parser.add_argument('--train_npy_path', type=str, default='../../data/trn/main-agent/text_audio/', 
                        help='Ruta a los archivos .npy de entrenamiento para el autoencoder.')
    parser.add_argument('--train_autoencoder', action='store_true', 
                        help='Entrenar el autoencoder en lugar de procesar datos.')
    parser.add_argument('--reduce_dimension', action='store_true', 
                        help='Reducir la dimensión de los embeddings de texto usando el AttentionAutoencoder entrenado.')
    parser.add_argument('--num_epochs', type=int, default=30, 
                        help='Número de épocas para entrenamiento.')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='Tasa de aprendizaje para el autoencoder.')
    parser.add_argument('--loss_file', type=str, default='training_loss.txt', 
                        help='Archivo para guardar los losses de entrenamiento.')
    args = parser.parse_args()
    main(args)

