import os
import json, glob, base64, io, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from collections import Counter

# Image and action parameters
IMG_SIZE = 128
CHUNK_SIZE = 5          # Predict 5 future actions
ACTION_DIM = 4          # [forward, backward, left, right]

# Model architecture
D_MODEL = 64            # Transformer dimension
N_HEADS = 2             # Attention heads
N_ENC_LAYERS = 1        # Encoder layers
N_DEC_LAYERS = 1        # Decoder layers
FFN_MULT = 2            # Feedforward expansion
N_SPATIAL = 64          # Spatial tokens from CNN

# Diffusion parameters
DIFFUSION_STEPS = 1000  # Training timesteps
DDIM_STEPS = 20         # Inference steps (faster)
BETA_START = 1e-4       # Noise schedule start
BETA_END = 0.02         # Noise schedule end

# Training parameters
EPOCHS = 200
BATCH = 16
LR_PHASE1 = 0.01        # SGD learning rate
LR_PHASE2 = 5e-4        # Adam learning rate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'🖥️  Device: {device}')

torch.manual_seed(42)
np.random.seed(42)

DATA_DIR = './training-data'

# Load JSON files and deduplicate
files_list = sorted(glob.glob(os.path.join(DATA_DIR, '*.json')))
all_samples = []
seen_ts = set()
intent_labels = None

print(f'📂 Loading {len(files_list)} JSON files...')
for f in files_list:
    with open(f) as fh:
        d = json.load(fh)
    if intent_labels is None:
        intent_labels = d.get('metadata', {}).get('intent_labels', ['always_take_left', 'always_take_right'])
    
    for s in d.get('samples', []):
        ts = s.get('timestamp', id(s))
        if ts not in seen_ts:
            all_samples.append(s)
            seen_ts.add(ts)

n = len(all_samples)
num_intents = max(s['language_id'] for s in all_samples) + 1

print(f'\n📊 Dataset Statistics:')
print(f'  Total samples: {n}')
print(f'  Intent labels: {num_intents} {intent_labels}')
print(f'  Distribution: {Counter(s["language_id"] for s in all_samples)}')

# Sort samples by intent then timestamp for proper chunking
samples_by_intent = {}
for s in all_samples:
    lid = s['language_id']
    if lid not in samples_by_intent:
        samples_by_intent[lid] = []
    samples_by_intent[lid].append(s)

for lid in samples_by_intent:
    samples_by_intent[lid].sort(key=lambda x: x['timestamp'])

sorted_samples = []
for lid in sorted(samples_by_intent.keys()):
    sorted_samples.extend(samples_by_intent[lid])

# Decode images and extract actions
print(f'\n🖼️  Decoding images...')
images = torch.zeros(n, 3, IMG_SIZE, IMG_SIZE)
langs = torch.zeros(n, dtype=torch.long)
actions = torch.zeros(n, ACTION_DIM)

for i, s in enumerate(sorted_samples):
    # Decode base64 image
    b64 = s['image'].split(',')[1] if ',' in s['image'] else s['image']
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    images[i] = torch.from_numpy(np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
    
    langs[i] = s['language_id']
    actions[i] = torch.tensor([
        s['actions']['forward'],
        s['actions']['backward'],
        s['actions']['left'],
        s['actions']['right']
    ], dtype=torch.float32)

print(f'\n⚙️  Action statistics:')
print(f'  Forward:  {actions[:, 0].mean():.3f}')
print(f'  Backward: {actions[:, 1].mean():.3f}')
print(f'  Left:     {actions[:, 2].mean():.3f}')
print(f'  Right:    {actions[:, 3].mean():.3f}')

# Build action chunks (sliding windows within each intent)
chunk_indices = []
for lid in sorted(samples_by_intent.keys()):
    intent_idx = [i for i in range(n) if langs[i].item() == lid]
    for start in range(len(intent_idx) - CHUNK_SIZE + 1):
        chunk_indices.append((intent_idx[start], intent_idx[start:start + CHUNK_SIZE]))

print(f'\n📦 Created {len(chunk_indices)} action chunks (size={CHUNK_SIZE})')


class ActionChunkDataset(Dataset):
    """Dataset that returns (image, language_id, action_chunk)."""
    
    def __init__(self, chunks, imgs, lngs, acts, augment=False):
        self.chunks = chunks
        self.imgs = imgs
        self.lngs = lngs
        self.acts = acts
        self.augment = augment
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, i):
        img_idx, chunk_idx = self.chunks[i]
        img = self.imgs[img_idx].clone()
        lng = self.lngs[img_idx]
        act_chunk = self.acts[chunk_idx]  # Shape: (CHUNK_SIZE, ACTION_DIM)
        
        if self.augment:
            # Brightness jitter
            img = img * (0.7 + torch.rand(1).item() * 0.6)
            
            # Contrast adjustment
            mean = img.mean()
            img = (img - mean) * (0.7 + torch.rand(1).item() * 0.6) + mean
            
            # Gaussian noise
            img = img + torch.randn_like(img) * 0.03
            
            # Per-channel color jitter
            for c in range(3):
                img[c] = img[c] * (0.9 + torch.rand(1).item() * 0.2)
            
            img = img.clamp(0, 1)
        
        return img, lng, act_chunk

# Split train/val
np.random.shuffle(chunk_indices)
split_point = int(len(chunk_indices) * 0.85)

train_dataset = ActionChunkDataset(chunk_indices[:split_point], images, langs, actions, augment=True)
val_dataset = ActionChunkDataset(chunk_indices[split_point:], images, langs, actions, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, num_workers=2, pin_memory=True)

print(f'✓ Train chunks: {len(train_dataset)}')
print(f'✓ Val chunks:   {len(val_dataset)}')

class DiffusionScheduler:
    """Manages noise scheduling for diffusion process."""
    
    def __init__(self, num_steps=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.num_steps = num_steps
        self.device = device
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        
        # Precompute useful values
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
    
    def add_noise(self, x_0, t, noise=None):
        """Add noise to x_0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Gather coefficients for batch
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        
        # x_t = √(α̅_t) * x_0 + √(1 - α̅_t) * ε
        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise
        
        return x_t, noise
    
    def ddim_sample(self, model, shape, image, language_id, num_steps=20, eta=0.0):
        """DDIM sampling for fast inference."""
        # Create timestep schedule (skip steps)
        timesteps = torch.linspace(self.num_steps - 1, 0, num_steps, dtype=torch.long, device=self.device)
        
        # Start from pure noise
        x_t = torch.randn(shape, device=self.device)
        
        for i, t in enumerate(timesteps):
            # Predict noise at timestep t
            t_batch = t.repeat(shape[0])
            predicted_noise = model(x_t, t_batch, image, language_id)
            
            # Get alpha values
            alpha_bar_t = self.alpha_bar[t]
            alpha_bar_t_prev = self.alpha_bar[timesteps[i + 1]] if i < len(timesteps) - 1 else torch.tensor(1.0)
            
            # Predict x_0 from x_t and noise
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
            x_0_pred = (x_t - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t
            
            # DDIM update (deterministic when eta=0)
            if i < len(timesteps) - 1:
                sqrt_alpha_bar_t_prev = torch.sqrt(alpha_bar_t_prev)
                sqrt_one_minus_alpha_bar_t_prev = torch.sqrt(1.0 - alpha_bar_t_prev)
                
                # Direction pointing to x_t
                dir_xt = sqrt_one_minus_alpha_bar_t_prev * predicted_noise
                
                # DDIM step
                x_t = sqrt_alpha_bar_t_prev * x_0_pred + dir_xt
            else:
                x_t = x_0_pred
        
        return x_t

# Initialize scheduler
scheduler = DiffusionScheduler(
    num_steps=DIFFUSION_STEPS,
    beta_start=BETA_START,
    beta_end=BETA_END,
    device=device
)

print(f'✓ Diffusion scheduler initialized')
print(f'  Training steps: {DIFFUSION_STEPS}')
print(f'  Inference steps (DDIM): {DDIM_STEPS}')
print(f'  Beta range: [{BETA_START}, {BETA_END}]')

class SinusoidalPositionEncoding(nn.Module):
    """Sinusoidal encoding for diffusion timesteps."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        """Args: t (B,) timesteps. Returns: (B, dim) encodings."""
        device = t.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None].float() * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class DiffusionPolicy(nn.Module):
    """Language-conditioned diffusion policy for action prediction."""
    
    def __init__(self, num_intents, img_size=128, action_dim=4, chunk_size=5,
                 d_model=64, n_heads=2, n_enc_layers=1, n_dec_layers=1):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.d_model = d_model
        
        # Vision encoder (CNN)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)   # 128 -> 64
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # 64 -> 32
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 32 -> 16
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)  # 16 -> 8
        self.relu = nn.ReLU()
        
        # Project spatial features to D_MODEL
        n_spatial = (img_size // 16) ** 2  # 8x8 = 64 tokens
        self.spatial_proj = nn.Linear(64, d_model)
        self.spatial_pos = nn.Parameter(torch.randn(n_spatial, d_model) * 0.02)
        
        # Language embedding
        self.lang_embed = nn.Embedding(num_intents, d_model)
        
        # Timestep encoding
        self.time_encoder = SinusoidalPositionEncoding(d_model)
        
        # Project noisy actions to D_MODEL
        self.action_proj = nn.Linear(action_dim, d_model)
        
        # Transformer encoder (processes vision + language + time)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * FFN_MULT,
            dropout=0.0,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_enc_layers)
        
        # Transformer decoder (queries for action noise)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * FFN_MULT,
            dropout=0.0,
            batch_first=False
        )
        self.transformer_decoder = nn.TransformerDecoder(dec_layer, num_layers=n_dec_layers)
        
        # Learnable queries for each action timestep
        self.action_queries = nn.Parameter(torch.randn(chunk_size, d_model) * 0.02)
        
        # Noise prediction head
        self.noise_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, action_dim)
        )
    
    def forward(self, noisy_actions, t, image, language_id):
        """Predict noise in noisy_actions."""
        B = image.size(0)
        
        # Encode vision (CNN)
        x = self.relu(self.conv1(image))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))  # (B, 64, 8, 8)
        
        # Flatten spatial dimensions and project
        x = x.flatten(2).permute(0, 2, 1)  # (B, 64, 64)
        x = self.spatial_proj(x) + self.spatial_pos.unsqueeze(0)  # (B, 64, D_MODEL)
        
        # Encode language
        lang_token = self.lang_embed(language_id)  # (B, D_MODEL)
        
        # Encode timestep
        time_embed = self.time_encoder(t)  # (B, D_MODEL)
        
        # Encode noisy actions
        action_embed = self.action_proj(noisy_actions)  # (B, CHUNK_SIZE, D_MODEL)
        
        # Concatenate all tokens for encoder: [spatial, lang, time]
        enc_input = torch.cat([
            x,                                # (B, 64, D_MODEL)
            lang_token.unsqueeze(1),          # (B, 1, D_MODEL)
            time_embed.unsqueeze(1)           # (B, 1, D_MODEL)
        ], dim=1)  # (B, 66, D_MODEL)
        
        # Transformer encoder
        memory = self.transformer_encoder(enc_input.permute(1, 0, 2))  # (66, B, D_MODEL)
        
        # Prepare decoder queries (incorporate language and action embeddings)
        queries = self.action_queries.unsqueeze(1).expand(-1, B, -1)  # (CHUNK_SIZE, B, D_MODEL)
        queries = queries + lang_token.unsqueeze(0)  # Add language conditioning
        queries = queries + action_embed.permute(1, 0, 2)  # Add noisy action info
        
        # Transformer decoder
        decoded = self.transformer_decoder(queries, memory)  # (CHUNK_SIZE, B, D_MODEL)
        decoded = decoded.permute(1, 0, 2)  # (B, CHUNK_SIZE, D_MODEL)
        
        # Predict noise
        predicted_noise = self.noise_head(decoded)  # (B, CHUNK_SIZE, ACTION_DIM)
        
        return predicted_noise


# Initialize model
model = DiffusionPolicy(
    num_intents=num_intents,
    img_size=IMG_SIZE,
    action_dim=ACTION_DIM,
    chunk_size=CHUNK_SIZE,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_enc_layers=N_ENC_LAYERS,
    n_dec_layers=N_DEC_LAYERS
).to(device)

if torch.cuda.device_count() > 1:
    print(f'Using {torch.cuda.device_count()} GPUs for training!')
    model = nn.DataParallel(model)

num_params = sum(p.numel() for p in model.parameters())
print(f'\n🧠 Model: DiffusionPolicy')
print(f'  Parameters: {num_params:,}')
print(f'  Device: {device}')

mse_loss = nn.MSELoss()
bce_loss = nn.BCEWithLogitsLoss()

best_val = float('inf')
best_acc = 0.0
t0 = time.time()

print(f'\n{"="*70}')
print(f'🚀 Phase 1: SGD with Momentum')
print(f'{"="*70}')
print(f'Optimizer: SGD (momentum=0.9)')
print(f'Learning rate: {LR_PHASE1} (with warmup + decay)')
print(f'Epochs: 100')
print(f'Gradient clipping: 1.0')
print()

optimizer = optim.SGD(model.parameters(), lr=LR_PHASE1, momentum=0.9)

for epoch in range(1, 101):
    # Learning rate schedule
    if epoch <= 10:
        # Warmup
        lr = LR_PHASE1 * epoch / 10
    elif epoch > 60:
        # Decay
        lr = LR_PHASE1 * max(1 - (epoch - 60) / 40, 0.01)
    else:
        lr = LR_PHASE1
    
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    
    # Training
    model.train()
    train_loss = 0.0
    n_train = 0
    
    for img, lng, act in train_loader:
        img = img.to(device)
        lng = lng.to(device)
        act = act.to(device)  # (B, CHUNK_SIZE, ACTION_DIM)
        
        # Sample random timesteps
        t = torch.randint(0, DIFFUSION_STEPS, (img.size(0),), device=device)
        
        # Add noise to actions
        noise = torch.randn_like(act)
        noisy_act, _ = scheduler.add_noise(act, t, noise)
        
        # Predict noise
        predicted_noise = model(noisy_act, t, img, lng)
        
        # Compute loss
        loss = mse_loss(predicted_noise, noise)
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_loss += loss.item() * img.size(0)
        n_train += img.size(0)
    
    train_loss /= n_train
    
    # Validation
    model.eval()
    val_loss = 0.0
    n_val = 0
    
    with torch.no_grad():
        for img, lng, act in val_loader:
            img = img.to(device)
            lng = lng.to(device)
            act = act.to(device)
            
            # Sample random timesteps
            t = torch.randint(0, DIFFUSION_STEPS, (img.size(0),), device=device)
            
            # Add noise
            noise = torch.randn_like(act)
            noisy_act, _ = scheduler.add_noise(act, t, noise)
            
            # Predict noise
            predicted_noise = model(noisy_act, t, img, lng)
            
            # Compute loss
            loss = mse_loss(predicted_noise, noise)
            val_loss += loss.item() * img.size(0)
            n_val += img.size(0)
    
    val_loss /= n_val
    
    # Print progress
    if epoch % 10 == 0 or epoch == 1:
        elapsed = time.time() - t0
        print(f'Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | '
              f'LR: {lr:.6f} | Time: {elapsed:.0f}s')
    
    # Save best model
    if val_loss < best_val:
        best_val = val_loss
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), './best_diffusion_model.pt')

print(f'\n✓ Phase 1 complete | Best val loss: {best_val:.4f}')

# Phase 2
print(f'\n{"="*70}')
print(f'🚀 Phase 2: Adam with Cosine Annealing')
print(f'{"="*70}')
print(f'Optimizer: Adam')
print(f'Learning rate: {LR_PHASE2} (cosine schedule)')
print(f'Epochs: 100')
print()

optimizer2 = optim.Adam(model.parameters(), lr=LR_PHASE2)
scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=100)

for epoch in range(101, 201):
    # Training
    model.train()
    train_loss = 0.0
    n_train = 0
    
    for img, lng, act in train_loader:
        img = img.to(device)
        lng = lng.to(device)
        act = act.to(device)
        
        # Sample random timesteps
        t = torch.randint(0, DIFFUSION_STEPS, (img.size(0),), device=device)
        
        # Add noise
        noise = torch.randn_like(act)
        noisy_act, _ = scheduler.add_noise(act, t, noise)
        
        # Predict noise
        predicted_noise = model(noisy_act, t, img, lng)
        
        # Compute loss
        loss = mse_loss(predicted_noise, noise)
        
        # Optimize
        optimizer2.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer2.step()
        
        train_loss += loss.item() * img.size(0)
        n_train += img.size(0)
    
    scheduler_lr.step()
    train_loss /= n_train
    
    # Validation
    model.eval()
    val_loss = 0.0
    n_val = 0
    
    with torch.no_grad():
        for img, lng, act in val_loader:
            img = img.to(device)
            lng = lng.to(device)
            act = act.to(device)
            
            t = torch.randint(0, DIFFUSION_STEPS, (img.size(0),), device=device)
            noise = torch.randn_like(act)
            noisy_act, _ = scheduler.add_noise(act, t, noise)
            predicted_noise = model(noisy_act, t, img, lng)
            
            loss = mse_loss(predicted_noise, noise)
            val_loss += loss.item() * img.size(0)
            n_val += img.size(0)
    
    val_loss /= n_val
    
    # Print progress
    if epoch % 10 == 0:
        elapsed = time.time() - t0
        current_lr = optimizer2.param_groups[0]['lr']
        print(f'Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | '
              f'LR: {current_lr:.6f} | Time: {elapsed:.0f}s')
    
    # Save best model
    if val_loss < best_val:
        best_val = val_loss
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), './best_diffusion_model.pt')

print(f'\n{"="*70}')
print(f'✓ Training Complete!')
print(f'{"="*70}')
print(f'Best validation loss: {best_val:.4f}')
print(f'Total training time: {(time.time() - t0)/60:.1f} minutes')
print()

# Load best model
model_to_load = model.module if hasattr(model, 'module') else model
model_to_load.load_state_dict(torch.load('./best_diffusion_model.pt', weights_only=True))
model.eval()

print(f'{"="*70}')
print(f'🎯 Action Prediction Evaluation (DDIM Sampling)')
print(f'{"="*70}')
print()

# Evaluate per intent
for lid in range(num_intents):
    mask = langs == lid
    test_imgs = images[mask][:200].to(device)
    test_lngs = langs[mask][:200].to(device)
    n_test = test_imgs.size(0)
    
    if n_test == 0:
        continue
    
    print(f'Intent [{lid}]: "{intent_labels[lid]}" ({mask.sum()} total samples)')
    print(f'  Testing on {n_test} images...')
    
    # Generate actions using DDIM sampling
    with torch.no_grad():
        predicted_actions = scheduler.ddim_sample(
            model=model,
            shape=(n_test, CHUNK_SIZE, ACTION_DIM),
            image=test_imgs,
            language_id=test_lngs,
            num_steps=DDIM_STEPS
        )
    
    # Convert to probabilities (apply sigmoid)
    # Note: Diffusion output is in logit space for binary actions
    action_probs = torch.sigmoid(predicted_actions[:, 0, :]).cpu()  # First timestep
    
    # Compute statistics
    mean_probs = action_probs.mean(dim=0)
    std_probs = action_probs.std(dim=0)
    
    print(f'  Mean action probabilities (t=0):')
    print(f'    Forward:  {mean_probs[0]:.3f} ± {std_probs[0]:.3f}')
    print(f'    Backward: {mean_probs[1]:.3f} ± {std_probs[1]:.3f}')
    print(f'    Left:     {mean_probs[2]:.3f} ± {std_probs[2]:.3f}')
    print(f'    Right:    {mean_probs[3]:.3f} ± {std_probs[3]:.3f}')
    print()

print(f'✓ Evaluation complete!')

# Move model to CPU for export
model_cpu = model.module.cpu() if hasattr(model, 'module') else model.cpu()
state = model_cpu.state_dict()

# Convert weights to JSON-serializable format
weights = {}
for key, tensor in state.items():
    weights[key] = {
        'shape': list(tensor.shape),
        'data': tensor.flatten().tolist()
    }

# Create export dictionary
export_data = {
    'format': 'language_conditioned_diffusion_policy',
    'version': '1.0',
    'num_params': num_params,
    
    # Model architecture
    'architecture': {
        'num_intents': num_intents,
        'img_size': IMG_SIZE,
        'action_dim': ACTION_DIM,
        'chunk_size': CHUNK_SIZE,
        'd_model': D_MODEL,
        'n_heads': N_HEADS,
        'n_enc_layers': N_ENC_LAYERS,
        'n_dec_layers': N_DEC_LAYERS,
        'n_spatial_tokens': N_SPATIAL,
    },
    
    # Diffusion parameters
    'diffusion': {
        'training_steps': DIFFUSION_STEPS,
        'inference_steps': DDIM_STEPS,
        'beta_start': BETA_START,
        'beta_end': BETA_END,
        'scheduler': 'linear',
        'sampler': 'ddim',
    },
    
    # Language labels
    'intent_labels': intent_labels[:num_intents],
    
    # Model weights
    'weights': weights,
}

# Save to file
output_path = './diffusion_policy.json'
with open(output_path, 'w') as f:
    json.dump(export_data, f)

file_size_kb = os.path.getsize(output_path) / 1024
file_size_mb = file_size_kb / 1024

print(f'✓ Model exported successfully!')
print(f'  Path: {output_path}')
print(f'  Size: {file_size_mb:.1f} MB ({file_size_kb:.0f} KB)')
print(f'  Parameters: {num_params:,}')
print()
