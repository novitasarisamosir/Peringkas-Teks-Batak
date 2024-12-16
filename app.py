import os
import requests
from flask import Flask, render_template, url_for, request as req
from requests.exceptions import RequestException, Timeout
import time
import textwrap
import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments, TrainerCallback
from datasets import Dataset, concatenate_datasets
import evaluate
import torch
from tqdm import tqdm
from bs4 import BeautifulSoup
import logging
import numpy as np
from collections import Counter
import csv
from io import StringIO
import re
from sklearn.model_selection import KFold
import json
from datetime import datetime

# Konfigurasi logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Inisialisasi aplikasi flask
app = Flask(__name__)

# Konfigurasi sistem
MAX_RETRIES = 3
RETRY_DELAY = 2
CHUNK_SIZE = 1000
MAX_TOTAL_LENGTH = 10000
TIMEOUT = 60

# Lokasi dataset
MODEL_PATH = "fine_tuned_bart"
KALIMAT_DATASET_PATH = "./data/kalimat.csv"
CAMPURAN_DATASET_PATH = "./data/campuran.csv"
DATA_DATASET_PATH = "./data/data.csv"
COMBINED_MODEL_FLAG = "combined_model_flag.txt"

# Konfigurasi training
MAX_SAMPLES_PER_DATASET = 1500
NUM_EPOCHS = 15
K_FOLDS = 5

# Menambahkan path penyimpanan data
DATA_SAVE_DIR = "kfold_data"
if not os.path.exists(DATA_SAVE_DIR):
    os.makedirs(DATA_SAVE_DIR)

rouge_metric = evaluate.load("rouge")

# Deklarasi Variabel Global Model
model = None
tokenizer = None
device = None

# Fungsi logging ROUGE scores
def log_detailed_rouge_scores(rouge_scores, source=""):
    """
    Fungsi untuk logging detail ROUGE scores termasuk unigram, bigram, LCS, dan scores.
    Args:
        rouge_scores: Dictionary hasil perhitungan ROUGE
        source: String untuk mengidentifikasi sumber input ("URL" atau "Text Input")
    """
    logger.info(f"\nROUGE Analysis Results for {source}:")
    
    # Log N-gram details
    for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
        logger.info(f"\n{rouge_type.upper()}:")
        if rouge_type == 'rouge1':
            logger.info(f"    Jumlah unigram yang sama: {sum(rouge_scores[rouge_type]['common_unigrams'].values())}")
            logger.info(f"    Total unigram di ringkasan sistem: {sum(rouge_scores[rouge_type]['summary_unigrams'].values())}")
            logger.info(f"    Total unigram di ringkasan manual: {sum(rouge_scores[rouge_type]['reference_unigrams'].values())}")
            logger.info("    Detail unigram yang sama:")
            for unigram, count in rouge_scores[rouge_type]['common_unigrams'].items():
                logger.info(f"      '{unigram}': {count}")
            logger.info("    Detail unigram di ringkasan sistem:")
            for unigram, count in rouge_scores[rouge_type]['summary_unigrams'].items():
                logger.info(f"      '{unigram}': {count}")

            # Log ROUGE-1 scores
            logger.info(f"\n    ROUGE-1 Scores:")
            logger.info(f"      Precision: {rouge_scores[rouge_type]['p']:.4f}")
            logger.info(f"      Recall: {rouge_scores[rouge_type]['r']:.4f}")
            logger.info(f"      F1-score: {rouge_scores[rouge_type]['f']:.4f}")
            
        elif rouge_type == 'rouge2':
            logger.info(f"    Jumlah bigram yang sama: {sum(rouge_scores[rouge_type]['common_bigrams'].values())}")
            logger.info(f"    Total bigram di ringkasan sistem: {sum(rouge_scores[rouge_type]['summary_bigrams'].values())}")
            logger.info(f"    Total bigram di ringkasan manual: {sum(rouge_scores[rouge_type]['reference_bigrams'].values())}")
            logger.info("    Detail bigram yang sama:")
            for bigram, count in rouge_scores[rouge_type]['common_bigrams'].items():
                logger.info(f"      '{bigram}': {count}")
            logger.info("    Detail bigram di ringkasan sistem:")
            for bigram, count in rouge_scores[rouge_type]['summary_bigrams'].items():
                logger.info(f"      '{bigram}': {count}")

            # Log ROUGE-2 scores
            logger.info(f"\n    ROUGE-2 Scores:")
            logger.info(f"      Precision: {rouge_scores[rouge_type]['p']:.4f}")
            logger.info(f"      Recall: {rouge_scores[rouge_type]['r']:.4f}")
            logger.info(f"      F1-score: {rouge_scores[rouge_type]['f']:.4f}")
            
        elif rouge_type == 'rougeL':
            logger.info(f"    Panjang LCS: {rouge_scores[rouge_type]['lcs_length']}")
            logger.info(f"    Panjang ringkasan sistem: {len(rouge_scores[rouge_type]['summary_words'])}")
            logger.info(f"    Panjang teks asli: {len(rouge_scores[rouge_type]['reference_words'])}")

            # Log ROUGE-L scores
            logger.info(f"\n    ROUGE-L Scores:")
            logger.info(f"      Precision: {rouge_scores[rouge_type]['p']:.4f}")
            logger.info(f"      Recall: {rouge_scores[rouge_type]['r']:.4f}")
            logger.info(f"      F1-score: {rouge_scores[rouge_type]['f']:.4f}")
    
    # Log skor ringkasan keseluruhan
    logger.info("\nOverall ROUGE Scores Summary:")
    logger.info(f"ROUGE-1: Precision={rouge_scores['rouge1']['p']:.4f}, Recall={rouge_scores['rouge1']['r']:.4f}, F1={rouge_scores['rouge1']['f']:.4f}")
    logger.info(f"ROUGE-2: Precision={rouge_scores['rouge2']['p']:.4f}, Recall={rouge_scores['rouge2']['r']:.4f}, F1={rouge_scores['rouge2']['f']:.4f}")
    logger.info(f"ROUGE-L: Precision={rouge_scores['rougeL']['p']:.4f}, Recall={rouge_scores['rougeL']['r']:.4f}, F1={rouge_scores['rougeL']['f']:.4f}")

# Fungsi pembersihan teks:
def clean_text(text):
    # Menghapus karakter non-ASCII
    text = ''.join([i if ord(i) < 128 else ' ' for i in str(text)])
    
    # Menghapus tanda petik dua, tanda petik satu, underscore, tanda tanya, titik dua, dan tanda seru
    text = text.replace('"', '').replace("'", '').replace('_', '').replace('?', '').replace(':', '').replace('!', '')
    
    # Menghapus karakter khusus kecuali koma dan titik
    text = re.sub(r'[^a-zA-Z0-9\s.,-]', '', text)
    
    # Menghapus whitespace berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Fungsi case folding dan kapitalisasi:
def apply_case_folding_and_capitalize(text):
    # Pembersihan awal
    text = clean_text(text)
    
    # Ubah semua teks menjadi huruf kecil
    text = text.lower()
    
    # Pisahkan teks menjadi kalimat hanya menggunakan titik sebagai pembatas karena tanda baca lainnya dihilangkan
    sentences = re.split(r'([.]\s*)', text)
    
    # Gunakan huruf kapital pada huruf pertama setiap kalimat dan gabung kembali
    processed_text = ''
    for i in range(0, len(sentences)-1, 2):
        sentence = sentences[i]
        delimiter = sentences[i+1] if i+1 < len(sentences) else ''
        
        if sentence:  # Hanya memproses kalimat yang tidak kosong
            sentence = sentence.strip()
            if sentence:  # Jika kalimat masih belum kosong setelah stripping
                sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
        
        processed_text += sentence + delimiter
    
   # Tangani kalimat terakhir jika ada dan tidak diikuti pembatas
    if len(sentences) % 2 == 1 and sentences[-1]:
        last_sentence = sentences[-1].strip()
        if last_sentence:
            last_sentence = last_sentence[0].upper() + last_sentence[1:] if len(last_sentence) > 1 else last_sentence.upper()
        processed_text += last_sentence
    
    return processed_text

# Fungsi pemuatan model dan tokenizer
def load_model_and_tokenizer():
    global model, tokenizer, device
    try:
        if os.path.exists(os.path.join(MODEL_PATH, "config.json")):
            model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)
            tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)
        else:
            logger.warning("Model fine-tuned tidak ditemukan. Menggunakan model default.")
            model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
            tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    except Exception as e:
        logger.error(f"Terjadi kesalahan saat memuat model: {str(e)}")
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

# Fungsi pemuatan dan pemrosesan dataset
def load_and_process_dataset(file_path, dataset_type='summarization'):
    logger.info(f"Memuat dataset dari {file_path}...")
    try:
        # Baca dataset
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # Cek kolom yang diperlukan berdasarkan tipe dataset
        if dataset_type == 'summarization':
            # Untuk kalimat.csv, campuran.csv, dan data.csv
            required_columns = ['content', 'summary']
            
            # Tambahkan penanganan untuk kemungkinan kolom berbeda
            if not all(col in df.columns for col in required_columns):
                # Coba temukan kolom yang mirip
                content_cols = [col for col in df.columns if 'content' in col.lower() or 'text' in col.lower()]
                summary_cols = [col for col in df.columns if 'summary' in col.lower() or 'ringkasan' in col.lower()]
                
                if not content_cols or not summary_cols:
                    raise ValueError(f"Dataset {file_path} tidak memiliki kolom 'content' dan 'summary' yang diperlukan")
                
                # Gunakan kolom pertama yang cocok
                content_col = content_cols[0]
                summary_col = summary_cols[0]
                
                logger.info(f"Menggunakan kolom: {content_col} untuk konten, {summary_col} untuk ringkasan")
            else:
                content_col = 'content'
                summary_col = 'summary'
            
            # Konversi nilai non-string menjadi string dan handle NaN
            df[content_col] = df[content_col].fillna('').astype(str)
            df[summary_col] = df[summary_col].fillna('').astype(str)
            
            # Hapus baris yang kosong setelah konversi
            df = df[df[content_col].str.strip() != '']
            df = df[df[summary_col].str.strip() != '']
            
            processed_data = {
                'text': df[content_col].apply(lambda x: apply_case_folding_and_capitalize(str(x)) if pd.notna(x) else ''),
                'summary': df[summary_col].apply(lambda x: apply_case_folding_and_capitalize(str(x)) if pd.notna(x) else '')
            }
        
        # Buat DataFrame dan bersihkan
        processed_df = pd.DataFrame(processed_data)
        
        # Hapus baris yang teks atau ringkasannya kosong setelah diproses
        processed_df = processed_df[
            (processed_df['text'].str.strip() != '') & 
            (processed_df['summary'].str.strip() != '')
        ]
        
        # Log dataset statistics
        logger.info(f"Dataset {file_path}:")
        logger.info(f"  Jumlah baris sebelum cleaning: {len(df)}")
        logger.info(f"  Jumlah baris setelah cleaning: {len(processed_df)}")
        
        if len(processed_df) == 0:
            raise ValueError(f"Tidak ada data valid dalam dataset {file_path} setelah cleaning")
        
        # Sampel jika diperlukan
        if len(processed_df) > MAX_SAMPLES_PER_DATASET:
            processed_df = processed_df.sample(n=MAX_SAMPLES_PER_DATASET, random_state=42)
        
        return processed_df
    
    except Exception as e:
        logger.error(f"Terjadi kesalahan saat memuat dataset {file_path}: {str(e)}")
        logger.error(f"Detail error: ", exc_info=True)
        raise

# Class callback untuk training
class CustomCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            logger.info(f"Epoch {state.epoch:.2f} - Step {state.global_step} - Loss: {logs.get('loss', 'N/A')}")
            
# Fungsi perhitungan pkor ROUGE:
def calculate_rouge_scores(summary, reference):
    def count_ngrams(text, n):
        words = text.split()
        return Counter([' '.join(words[i:i+n]) for i in range(len(words)-n+1)])
    
# Fungsi LCS untuk Perhitungan ROUGE-L
    def lcs(X, Y):
        m, n = len(X), len(Y)
        L = [[0 for i in range(n+1)] for j in range(m+1)]
        for i in range(m+1):
            for j in range(n+1):
                if i == 0 or j == 0:
                    L[i][j] = 0
                elif X[i-1] == Y[j-1]:
                    L[i][j] = L[i-1][j-1] + 1
                else:
                    L[i][j] = max(L[i-1][j], L[i][j-1])
        return L[m][n]

    summary_words = summary.split()
    reference_words = reference.split()

    # ROUGE-1
    summary_unigrams = count_ngrams(summary, 1)
    reference_unigrams = count_ngrams(reference, 1)
    common_unigrams = summary_unigrams & reference_unigrams
    rouge1_recall = sum(common_unigrams.values()) / len(reference_words) if reference_words else 0
    rouge1_precision = sum(common_unigrams.values()) / len(summary_words) if summary_words else 0
    rouge1_f1 = 2 * (rouge1_precision * rouge1_recall) / (rouge1_precision + rouge1_recall) if (rouge1_precision + rouge1_recall) > 0 else 0
    # ROUGE-2
    summary_bigrams = count_ngrams(summary, 2)
    reference_bigrams = count_ngrams(reference, 2)
    common_bigrams = summary_bigrams & reference_bigrams
    rouge2_recall = sum(common_bigrams.values()) / max(1, len(reference_words) - 1)
    rouge2_precision = sum(common_bigrams.values()) / max(1, len(summary_words) - 1)
    rouge2_f1 = 2 * (rouge2_precision * rouge2_recall) / (rouge2_precision + rouge2_recall) if (rouge2_precision + rouge2_recall) > 0 else 0

    # ROUGE-L
    lcs_length = lcs(summary_words, reference_words)
    rougeL_recall = lcs_length / len(reference_words) if reference_words else 0
    rougeL_precision = lcs_length / len(summary_words) if summary_words else 0
    rougeL_f1 = 2 * (rougeL_precision * rougeL_recall) / (rougeL_precision + rougeL_recall) if (rougeL_precision + rougeL_recall) > 0 else 0

    return {
        'rouge1': {
            'f': rouge1_f1,
            'p': rouge1_precision,
            'r': rouge1_recall,
            'common_unigrams': common_unigrams,
            'summary_unigrams': summary_unigrams,
            'reference_unigrams': reference_unigrams
        },
        'rouge2': {
            'f': rouge2_f1,
            'p': rouge2_precision,
            'r': rouge2_recall,
            'common_bigrams': common_bigrams,
            'summary_bigrams': summary_bigrams,
            'reference_bigrams': reference_bigrams
        },
        'rougeL': {
            'f': rougeL_f1,
            'p': rougeL_precision,
            'r': rougeL_recall,
            'lcs_length': lcs_length,
            'summary_words': summary_words,
            'reference_words': reference_words
        }
    }

# Fungsi fine-tuning model BART
def fine_tune_bart(dataset):
    global model, tokenizer
    logger.info("Memulai proses fine-tuning dengan 5-fold cross validation...")
    
    # Initsialisasi lists untuk menyimpan  ROUGE scores pada setiap fold
    all_rouge_scores = []
    
    # Buat K-Fold cross validator
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    # Ubah dataset menjadi pandas DataFrame agar splitting lebih mudah
    df = dataset if isinstance(dataset, pd.DataFrame) else pd.DataFrame(dataset)
    
    # Buat direktori untuk menyimpan data fold
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fold_dir = os.path.join(DATA_SAVE_DIR, f"kfold_data_{timestamp}")
    os.makedirs(fold_dir)
    
    # Lakukan K-Fold cross validation
    for fold, (train_idx, test_idx) in enumerate(kf.split(df), 1):
        logger.info(f"\nProcessing Fold {fold}/{K_FOLDS}")
        
        # Split data
        train_data = df.iloc[train_idx]
        test_data = df.iloc[test_idx]
        
        logger.info(f"Training set size: {len(train_data)} samples")
        logger.info(f"Test set size: {len(test_data)} samples")
        
        # Simpan data fold
        train_path = os.path.join(fold_dir, f"train_fold_{fold}.csv")
        test_path = os.path.join(fold_dir, f"test_fold_{fold}.csv")
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        logger.info(f"Saved fold {fold} data to {fold_dir}")
        
        # Konversi menjadi format dataset
        train_dataset = Dataset.from_pandas(train_data)
        test_dataset = Dataset.from_pandas(test_data)
        
        # Inisialisasi instance model baru untuk setiap fold
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        model.to(device)
        
        # Fungsi preprocessing data:
        def preprocess_function(examples):
            inputs = tokenizer(examples["text"], max_length=256, truncation=True, padding="max_length")
            targets = tokenizer(examples["summary"], max_length=64, truncation=True, padding="max_length")
            inputs["labels"] = targets["input_ids"]
            return inputs
        
        logger.info(f"Preprocessing fold {fold} dataset...")
        tokenized_train = train_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc=f"Tokenizing fold {fold} training data",
        )
        
        tokenized_test = test_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=test_dataset.column_names,
            desc=f"Tokenizing fold {fold} test data",
        )
        
        training_args = TrainingArguments(
            output_dir=f"./results/fold_{fold}",
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'./logs/fold_{fold}',
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            eval_strategy="steps",
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=8,
            remove_unused_columns=False,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            callbacks=[CustomCallback()],
        )
        
        logger.info(f"Training fold {fold}...")
        trainer.train()
        
        # Evaluasi pada test set
        logger.info(f"Evaluating fold {fold}...")
        fold_rouge_scores = []
        
        for idx in range(len(test_data)):
            test_text = test_data.iloc[idx]['text']
            reference_summary = test_data.iloc[idx]['summary']
            
            
            # Hasilkan ringkasan menggunakan model fold saat ini
            inputs = tokenizer.encode(test_text, return_tensors="pt", max_length=512, truncation=True)
            inputs = inputs.to(device)
            
            summary_ids = model.generate(
                inputs,
                max_length=150,
                min_length=40,
                num_beams=4,
                length_penalty=2.0,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
            
            generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            generated_summary = apply_case_folding_and_capitalize(generated_summary)
            
            # Menghitung ROUGE scores
            rouge_scores = calculate_rouge_scores(generated_summary, reference_summary)
            fold_rouge_scores.append(rouge_scores)
        
        # Mengitung rata-rata ROUGE scores untuk fold 
        fold_avg_scores = {
            'fold': fold,
            'rouge1': {
                'f': np.mean([s['rouge1']['f'] for s in fold_rouge_scores]),
                'p': np.mean([s['rouge1']['p'] for s in fold_rouge_scores]),
                'r': np.mean([s['rouge1']['r'] for s in fold_rouge_scores])
            },
            'rouge2': {
                'f': np.mean([s['rouge2']['f'] for s in fold_rouge_scores]),
                'p': np.mean([s['rouge2']['p'] for s in fold_rouge_scores]),
                'r': np.mean([s['rouge2']['r'] for s in fold_rouge_scores])
            },
            'rougeL': {
                'f': np.mean([s['rougeL']['f'] for s in fold_rouge_scores]),
                'p': np.mean([s['rougeL']['p'] for s in fold_rouge_scores]),
                'r': np.mean([s['rougeL']['r'] for s in fold_rouge_scores])
            }
        }
        
        # Simpan individual sample scores untuk fold 
        sample_scores_path = os.path.join(fold_dir, f'sample_rouge_scores_fold_{fold}.json')
        with open(sample_scores_path, 'w') as f:
            json.dump(fold_rouge_scores, f, indent=4)
        
        all_rouge_scores.append(fold_avg_scores)

        # Simpan rata-rata ROUGE scores untuk fold
        with open(os.path.join(fold_dir, f'rouge_scores_fold_{fold}.json'), 'w') as f:
            json.dump(fold_avg_scores, f, indent=4)
        
        logger.info(f"\nROUGE Scores for Fold {fold}:")
        logger.info(f"ROUGE-1 F1: {fold_avg_scores['rouge1']['f']:.4f}")
        logger.info(f"ROUGE-2 F1: {fold_avg_scores['rouge2']['f']:.4f}")
        logger.info(f"ROUGE-L F1: {fold_avg_scores['rougeL']['f']:.4f}")
        
        # Simpan model untuk fold
        fold_model_path = os.path.join(fold_dir, f'model_fold_{fold}')
        model.save_pretrained(fold_model_path)
        tokenizer.save_pretrained(fold_model_path)

    # Hitung dan simpan skor rata-rata keseluruhan
    overall_avg_scores = {
        'rouge1': {
            'f': np.mean([fold['rouge1']['f'] for fold in all_rouge_scores]),
            'p': np.mean([fold['rouge1']['p'] for fold in all_rouge_scores]),
            'r': np.mean([fold['rouge1']['r'] for fold in all_rouge_scores])
        },
        'rouge2': {
            'f': np.mean([fold['rouge2']['f'] for fold in all_rouge_scores]),
            'p': np.mean([fold['rouge2']['p'] for fold in all_rouge_scores]),
            'r': np.mean([fold['rouge2']['r'] for fold in all_rouge_scores])
        },
        'rougeL': {
            'f': np.mean([fold['rougeL']['f'] for fold in all_rouge_scores]),
            'p': np.mean([fold['rougeL']['p'] for fold in all_rouge_scores]),
            'r': np.mean([fold['rougeL']['r'] for fold in all_rouge_scores])
        }
    }
    
    # Simpan semua skor termasuk lipatan individu dan rata-rata keseluruhan
    final_scores = {
        'fold_scores': all_rouge_scores,
        'overall_average': overall_avg_scores
    }
    
    with open(os.path.join(fold_dir, 'all_rouge_scores.json'), 'w') as f:
        json.dump(final_scores, f, indent=4)
    
    logger.info("\nOverall Average ROUGE Scores across all folds:")
    logger.info(f"ROUGE-1 F1: {overall_avg_scores['rouge1']['f']:.4f}")
    logger.info(f"ROUGE-2 F1: {overall_avg_scores['rouge2']['f']:.4f}")
    logger.info(f"ROUGE-L F1: {overall_avg_scores['rougeL']['f']:.4f}")
    
    # Save the final model
    logger.info("Simpan final model...")
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    logger.info("Fine-tuning Selesai.")

# Fungsi untuk membuat ringkasan teks
def summarize_text(input_text, max_summary_length=150, min_summary_length=50, log_scores=True):
    global model, tokenizer, device
    
    if model is None or tokenizer is None:
        load_model_and_tokenizer()
    
    logger.info(f"Memulai summarization teks dengan panjang {len(input_text)} karakter")
    
   # Terapkan fold dan kapitalisasi sebelum tokenisasi
    processed_text = apply_case_folding_and_capitalize(input_text)
    
    inputs = tokenizer.encode(processed_text, return_tensors="pt", max_length=512, truncation=True)
    inputs = inputs.to(device)
    
    summary_ids = model.generate(
        inputs,
        max_length=max_summary_length,
        min_length=min_summary_length,
        num_beams=4,
        length_penalty=2.0,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    processed_summary = apply_case_folding_and_capitalize(summary)
    
    # Hitung ROUGE scores
    rouge_scores = calculate_rouge_scores(processed_summary, processed_text)
    
    # Catat skor ROUGE hanya jika diminta dan ini adalah ringkasan akhir
    if log_scores:
        log_detailed_rouge_scores(rouge_scores, "Final Summary")
    
    return processed_summary, rouge_scores

# Muat model dan tokenizer saat aplikasi dimulai
load_model_and_tokenizer()

if not os.path.exists(COMBINED_MODEL_FLAG):
    logger.info("Model belum di-fine-tune dengan dataset. Memulai proses fine-tuning...")
    try:
        # Memuat semua datasets]
        datasets = []
        
        # Memuat summarization datasets
        summarization_paths = [
            KALIMAT_DATASET_PATH,
            CAMPURAN_DATASET_PATH,
            DATA_DATASET_PATH
        ]
        
        for path in summarization_paths:
            dataset = load_and_process_dataset(path, 'summarization')
            datasets.append(dataset)
        
        # Gabungkan semua dataset
        combined_df = pd.concat(datasets, ignore_index=True)
        
        # Fine-tune model dengan  K-fold cross validation
        fine_tune_bart(combined_df)
        
        with open(COMBINED_MODEL_FLAG, 'w') as f:
            f.write("Model sudah di-fine-tune dengan semua dataset")
    except Exception as e:
        logger.error(f"Terjadi kesalahan saat fine-tuning: {str(e)}")
else:
    logger.info("Menggunakan model yang sudah di-fine-tune dengan semua dataset.")

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('index.html', result="Terjadi kesalahan internal server. Silakan coba lagi nanti."), 500

@app.route("/", methods=["GET", "POST"])
def Index():
    logger.info("Akses ke halaman utama")
    return render_template("index.html")

@app.route("/bantuan")
def bantuan():
    logger.info("Akses ke halaman bantuan")
    return render_template("bantuan.html")

@app.route("/tentang")
def tentang():
    logger.info("Akses ke halaman tentang")
    return render_template("tentang.html")

@app.route("/Summarize", methods=["GET", "POST"])
def Summarize():
    if req.method == "POST":
        logger.info("Menerima permintaan summarization")
        input_text = req.form["data"]
        
        # Terapkan case folding dan kapitalisasi pada input text
        input_text = apply_case_folding_and_capitalize(input_text)
        
        if len(input_text) > MAX_TOTAL_LENGTH:
            return render_template("index.html", result=f"Teks terlalu panjang. Maksimum {MAX_TOTAL_LENGTH} karakter.")
        
        max_summary_length = int(req.form["maxL"])
        min_summary_length = max_summary_length // 4

        logger.info(f"Panjang maksimum ringkasan: {max_summary_length}, panjang minimum: {min_summary_length}")
        
        chunks = textwrap.wrap(input_text, CHUNK_SIZE, break_long_words=False, replace_whitespace=False)
        summaries = []
        all_rouge_scores = []
        
        # Memproses chunks
        for chunk in chunks:
            summary, rouge_scores = summarize_text(
                input_text=chunk,
                max_summary_length=max_summary_length // len(chunks),
                min_summary_length=min_summary_length // len(chunks),
                log_scores=False  # Don't log ROUGE scores for intermediate chunks
            )
            summaries.append(summary)
            all_rouge_scores.append(rouge_scores)
        
        final_summary = " ".join(summaries)
        
        if len(final_summary) > max_summary_length:
            logger.info("Meringkas hasil gabungan")
            final_summary, final_rouge_scores = summarize_text(
                input_text=final_summary,
                max_summary_length=max_summary_length,
                min_summary_length=min_summary_length,
                log_scores=True  # Log ROUGE scores untuk final summary
            )
        else:
            # Menghitung rata-rata ROUGE scores jika multiple chunks
            final_rouge_scores = all_rouge_scores[0] if len(all_rouge_scores) == 1 else {
                'rouge1': {
                    'f': np.mean([s['rouge1']['f'] for s in all_rouge_scores]),
                    'p': np.mean([s['rouge1']['p'] for s in all_rouge_scores]),
                    'r': np.mean([s['rouge1']['r'] for s in all_rouge_scores])
                },
                'rouge2': {
                    'f': np.mean([s['rouge2']['f'] for s in all_rouge_scores]),
                    'p': np.mean([s['rouge2']['p'] for s in all_rouge_scores]),
                    'r': np.mean([s['rouge2']['r'] for s in all_rouge_scores])
                },
                'rougeL': {
                    'f': np.mean([s['rougeL']['f'] for s in all_rouge_scores]),
                    'p': np.mean([s['rougeL']['p'] for s in all_rouge_scores]),
                    'r': np.mean([s['rougeL']['r'] for s in all_rouge_scores])
                }
            }
            # Log the final averaged ROUGE scores
            log_detailed_rouge_scores(final_rouge_scores, "Final Summary")
        
        logger.info(f"Ringkasan selesai. Panjang ringkasan: {len(final_summary)} karakter")
        return render_template("index.html", 
                             result=final_summary,
                             rouge_scores=final_rouge_scores,
                             original_text=input_text)
    else:
        return render_template("index.html")

@app.route("/url", methods=["GET", "POST"])
def url_summarize():
    if req.method == "POST":
        logger.info("Menerima permintaan summarization dari URL")
        url = req.form["url"]
        try:
            logger.info(f"Mengambil konten dari URL: {url}")
            response = requests.get(url, timeout=TIMEOUT)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            input_text = ' '.join([p.get_text() for p in paragraphs])
            
            # Terapkan case folding and kapitalisasi pada teks melalui input URL
            input_text = apply_case_folding_and_capitalize(input_text)
            
            logger.info(f"Panjang teks dari URL: {len(input_text)} karakter")
            if len(input_text) > MAX_TOTAL_LENGTH:
                return render_template("url.html", result=f"Konten terlalu panjang. Maksimum {MAX_TOTAL_LENGTH} karakter.")
            
            max_summary_length = int(req.form["maxL"])
            min_summary_length = max_summary_length // 4
            logger.info(f"Panjang maksimum ringkasan: {max_summary_length}, panjang minimum: {min_summary_length}")

            summary, rouge_scores = summarize_text(
                input_text=input_text,
                max_summary_length=max_summary_length,
                min_summary_length=min_summary_length,
                log_scores=True  # Log ROUGE scores for URL summary
            )
            
            logger.info(f"Ringkasan selesai. Panjang ringkasan: {len(summary)} karakter")
            return render_template("url.html", 
                                result=summary,
                                rouge_scores=rouge_scores,
                                original_text=input_text)
        except Exception as e:
            return render_template("url.html", result=f"Terjadi kesalahan: {str(e)}")
    else:
        return render_template("url.html")

if __name__ == "__main__":
    logger.info("Memulai aplikasi Flask")
    app.run(debug=True)