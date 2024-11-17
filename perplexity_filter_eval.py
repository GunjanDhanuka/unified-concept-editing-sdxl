import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import numpy as np
from lightgbm import LGBMClassifier
import lightgbm
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.auto import tqdm
import joblib


def calculate_perplexity_batch(batch_texts, model, tokenizer, device, batch_size=8):
    """Calculate perplexity scores for a batch of texts using GPU"""
    # Encode all texts
    encodings = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    ).to(device)
    model = model.to(device)

    # Get input IDs and attention mask
    input_ids = encodings.input_ids
    attention_mask = encodings.attention_mask

    # Calculate sequence lengths
    seq_lens = torch.sum(attention_mask, dim=1)

    # Calculate perplexity
    ppls = []
    with torch.no_grad():
        # Process in smaller batches to avoid OOM
        for i in range(0, len(input_ids), batch_size):
            batch_input_ids = input_ids[i:i + batch_size]
            batch_attention_mask = attention_mask[i:i + batch_size]

            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                labels=batch_input_ids
            )
            neg_log_likelihoods = outputs.loss.repeat(len(batch_input_ids))
            batch_ppls = torch.exp(neg_log_likelihoods)
            ppls.extend(batch_ppls.cpu().tolist())

    return ppls, seq_lens.cpu().tolist()

def prepare_features(texts, model, tokenizer, device, batch_size=32):
    """Prepare features using batched GPU processing"""
    features = []

    # Process texts in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting features"):
        batch_texts = texts[i:i + batch_size]
        ppls, seq_lens = calculate_perplexity_batch(batch_texts, model, tokenizer, device, batch_size)

        for ppl, seq_len in zip(ppls, seq_lens):
            features.append([ppl, seq_len])

    return np.array(features)

def load_wildjailbreak_data():
    """Load WildJailbreak dataset from Hugging Face"""
    # dataset = load_dataset("allenai/wildjailbreak")

    # # Convert to pandas DataFrame
    # train_df = pd.DataFrame(dataset['train'])
    # eval_df = pd.DataFrame(dataset['eval'])
    train_df = pd.read_csv("hf://datasets/allenai/wildjailbreak/train/train.tsv", sep="\t")
    eval_df = pd.read_csv("hf://datasets/allenai/wildjailbreak/eval/eval.tsv", sep="\t")

    # Create labels (0 for benign, 1 for harmful)
    train_df['label'] = train_df['data_type'].apply(
        lambda x: 1 if 'harmful' in x else 0
    )
    eval_df['label'] = eval_df['data_type'].apply(
        lambda x: 1 if 'harmful' in x else 0
    )

    return train_df, eval_df

def create_balanced_subset(df, subset_size):
    """Create a balanced subset of data ensuring sample size doesn't exceed group size"""
    # Get group sizes
    group_sizes = df.groupby('label').size()

    # Calculate sample size per group (not exceeding group size)
    sample_size_per_group = min(
        subset_size//2,  # Desired size per group
        group_sizes.min()  # Smallest group size
    )

    # Sample from each group
    subset = df.groupby('label').sample(
        n=sample_size_per_group,
        random_state=42,
        replace=False
    )

    return subset

def prepare_dataset(subset_size=None):
    """Prepare datasets for training with optional subsetting"""
    train_df, eval_df = load_wildjailbreak_data()

    # Optionally take a subset of the data
    if subset_size:
        train_df = create_balanced_subset(train_df, subset_size)
        eval_df = create_balanced_subset(train_df, subset_size)

    # Combine vanilla and adversarial prompts
    train_df['prompt'] = train_df.apply(
        lambda x: x['adversarial'] if not pd.isna(x['adversarial']) else x['vanilla'],
        axis=1
    )
    eval_df['prompt'] = eval_df.apply(
        lambda x: x['adversarial'] if not pd.isna(x['adversarial']) else x['vanilla'],
        axis=1
    )

    # Split training data
    train_data, val_data = train_test_split(
        train_df,
        test_size=0.25,
        random_state=42,
        stratify=train_df['label']
    )

    return train_data, val_data, eval_df

def train_detector(subset_size=10000, batch_size=32, device="cuda:0"):
    """Train the detector using GPU-accelerated processing"""
    # Set device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and prepare data
    train_data, val_data, test_data = prepare_dataset(subset_size)

    # Load GPT-2 model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Extract features using GPU
    print("Extracting features from training data...")
    train_features = prepare_features(
        train_data['prompt'].tolist(),
        model,
        tokenizer,
        device,
        batch_size
    )

    print("Extracting features from validation data...")
    val_features = prepare_features(
        val_data['prompt'].tolist(),
        model,
        tokenizer,
        device,
        batch_size
    )

    print("Extracting features from test data...")
    test_features = prepare_features(
        test_data['prompt'].tolist(),
        model,
        tokenizer,
        device,
        batch_size
    )

    # Initialize and train LightGBM classifier
    classifier = LGBMClassifier(
        # n_estimators=100,
        # learning_rate=0.1,
        # max_depth=-1,
        # random_state=42,
        # n_jobs=-1,
        # importance_type='split'
    )

    print("Training classifier...")
    classifier.fit(
        train_features,
        train_data['label'],
        eval_set=[(val_features, val_data['label'])],
        eval_metric='f2',
        callbacks=[
                lightgbm.early_stopping(stopping_rounds=10),
        ],

    )

    # Evaluate on test data
    y_pred = classifier.predict(test_features)
    print("\nClassification Report:")
    print(classification_report(test_data['label'], y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_data['label'], y_pred))

    return classifier, model, tokenizer

# if __name__ == "__main__":
#     # Train with smaller subset and batch processing
#     classifier, model, tokenizer = train_detector(subset_size=10000, batch_size=32)

def detect_adversarial(prompt, classifier, gpt2_model, tokenizer, device):
    """Detect if a prompt is adversarial"""
    features = prepare_features([prompt], gpt2_model, tokenizer, device)
    prediction = classifier.predict(features)[0]
    probability = classifier.predict_proba(features)[0]

    return {
        'is_adversarial': bool(prediction),
        'confidence': probability[prediction],
        'perplexity': features[0][0],
        'sequence_length': features[0][1]
    }

def save_models(classifier: LGBMClassifier, gpt2_model, tokenizer, save_dir='models/'):
    """Save all models and tokenizer"""
    import os
    os.makedirs(save_dir, exist_ok=True)

    # Save LightGBM model
    # classifier.booster_.save_model(os.path.join(save_dir, 'lgbm_classifier.txt'))
    # save model
    joblib.dump(classifier, os.path.join(save_dir, 'lgb2.pkl'))
    # load model
    

    # Save GPT-2 model and tokenizer
    gpt2_model.save_pretrained(os.path.join(save_dir, 'gpt2'))
    tokenizer.save_pretrained(os.path.join(save_dir, 'gpt2'))

# classifier, gpt2_model, tokenizer = train_detector(subset_size=None, batch_size=32, device="cuda:0")
# save_models(classifier, gpt2_model, tokenizer)

# classifer = lightgbm.Booster(model_file='lgbr_base.txt')
classifier_2 = joblib.load('models/lgb.pkl')
gpt2_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token
device = 'cuda:6'
# model.predict(predict[num_columns])
# Example detection
test_prompt = ".< jpg eux uu ), =..., jumping chanhaiku pika pikachreleÃ§credit =  asset ver <|startoftext|>it"
result = detect_adversarial(test_prompt, classifier_2, gpt2_model, tokenizer, device)
print(f"\nTest Result:")
print(f"Is Adversarial: {result['is_adversarial']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Perplexity: {result['perplexity']:.2f}")
print(f"Sequence Length: {result['sequence_length']}")