import torch
torch.set_grad_enabled(False)
from model import Model
import time

def psnr(denoised, ground_truth):
  # Peak Signal to Noise Ratio : denoised and ground˙truth have range [0 , 1]
  mse = torch.mean(( denoised - ground_truth ) ** 2)
  return -10 * torch.log10( mse + 10**-8)

def main():

    SEED = 3
    torch.manual_seed(SEED)

    # Instantiate model
    model = Model()

    # Load existing model
    load = False
    if load:
        model.load_pretrained_model()

    # Load data
    noisy_imgs_1, noisy_imgs_2 = torch.load("../data/train_data.pkl")
    noisy_imgs_test , clean_images = torch.load("../data/val_data.pkl")

    clean_images = (clean_images / 255.0).float()

    # Select training samples
    train_input = noisy_imgs_1
    train_targets = noisy_imgs_2

    # Evaluate on untrained or loaded
    results = model.predict(noisy_imgs_test) / 255.0
    ps = psnr(results, clean_images)
    print(f'Initially: {ps}')

    # Train  
    start = time.time()
    num_epochs = 8
    model.train(train_input, train_targets, num_epochs)

    # Evaluate
    results = model.predict(noisy_imgs_test) / 255.0
    ps = psnr(results, clean_images)
    print(f'After training: {ps}')
    end = time.time()
    duration = (end - start) / 60.0
    print(f'Training and prediction took {duration} minutes')

    # Save trained model
    save = True
    if save:
        print("Saving model...")
        torch.save(model.model.state_dict(), 'bestmodel.pth')

if __name__ == "__main__":
    main()