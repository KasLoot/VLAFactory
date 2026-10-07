# Guide to Tain a VAE

After cloning and cd to this repository.

## Stage 1: Download and prepare the training and validation datasets.

```bash
mkdir -p /workspace/data/dataset/poloclub/diffusiondb/images/train
mkdir -p /workspace/data/dataset/poloclub/diffusiondb/images/val

hf download poloclub/diffusiondb \
  metadata.parquet \
  --repo-type dataset \
  --local-dir /workspace/data/dataset/poloclub/diffusiondb

hf download poloclub/diffusiondb \
  images/part-{000001..000050}.zip \
  --repo-type dataset \
  --local-dir /workspace/data/dataset/poloclub/diffusiondb

mv /workspace/data/dataset/poloclub/diffusiondb/images/part-{000001..000050}.zip \
   /workspace/data/dataset/poloclub/diffusiondb/images/train/



hf download poloclub/diffusiondb \
  images/part-{000051..000060}.zip \
  --repo-type dataset \
  --local-dir /workspace/data/dataset/poloclub/diffusiondb

mv /workspace/data/dataset/poloclub/diffusiondb/images/part-{000051..000060}.zip \
   /workspace/data/dataset/poloclub/diffusiondb/images/val/



cd /workspace/data/dataset/poloclub/diffusiondb/images/train

for f in *.zip; do
  dir="${f%.zip}"
  mkdir -p "$dir"
  if unzip "$f" -d "$dir"; then
    rm "$f"
  fi
done


cd /workspace/data/dataset/poloclub/diffusiondb/images/val

for f in *.zip; do
  dir="${f%.zip}"
  mkdir -p "$dir"
  if unzip "$f" -d "$dir"; then
    rm "$f"
  fi
done

```


## Stage 2: Setup Environment

```bash
uv sync
```

## Stage 3: Pre-process dataset

run the `diffusion/preprocess_vae_data.py` code. Make sure to change the folder names.

## Stage 4: Start Training

```bash
python -m diffusion.train_vae
```