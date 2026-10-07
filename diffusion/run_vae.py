import torch
import PIL.Image as Image
import os

from diffusion.vae import VAE

from torchvision.transforms.functional import pil_to_tensor


def run_vae(image_path, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running VAE on device: {device}")

    vae = VAE().to(device)
    vae.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False)["model_state_dict"])
    vae.eval()

    image = Image.open(image_path).convert("RGB")


    image_tensor = pil_to_tensor(image).unsqueeze(0).float().to(device)

    with torch.no_grad():
        output = vae(image_tensor)

    return image, output


def show_image(original_img, model_output):
    original_img = original_img.convert("RGB")

    recon_img_tensor = model_output.reconstruction

    # clamp the reconstructed image tensor to [-1, 1]
    recon_img_tensor = torch.clamp(recon_img_tensor, min=-1, max=1)
    reconstructed_img = ((recon_img_tensor + 1) / 2).squeeze(0).permute(1, 2, 0).cpu().numpy()
    reconstructed_img = (reconstructed_img * 255).astype("uint8")
    reconstructed_img = Image.fromarray(reconstructed_img, mode="RGB")

    target_height = max(original_img.height, reconstructed_img.height)
    combined = Image.new("RGB", (original_img.width + reconstructed_img.width, target_height), "white")
    combined.paste(original_img, (0, 0))
    combined.paste(reconstructed_img, (original_img.width, 0))
    combined.show()


def main():
    image_path = "image/butterfly.jpg"
    checkpoint_path = "checkpoints/vae/8ce10dc70c69/best.pt"
    original_img, output = run_vae(image_path, checkpoint_path)
    show_image(original_img, output)

if __name__ == "__main__":
    main()
