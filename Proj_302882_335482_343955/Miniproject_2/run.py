import torch
torch.set_grad_enabled(False)
from model import Model
import time
import pickle

# def psnr(denoised, ground_truth):
#   # Peak Signal to Noise Ratio : denoised and ground˙truth have range [0 , 1]
#   mse = torch.mean(( denoised - ground_truth ) ** 2)
#   return -10 * torch.log10( mse + 10**-8)

def compute_psnr(x, y, max_range=1.0):
    assert x.shape == y.shape and x.ndim == 4
    return 20 * torch.log10(torch.tensor(max_range)) - 10 * torch.log10(((x-y) ** 2).mean((1,2,3))).mean()

def main():

    # SEED = 3
    # torch.manual_seed(SEED)
    # no need, seed is set inside Model

    # Instantiate model
    model = Model()

    # Load existing model
    load = False
    if load:
        model.load_pretrained_model()

    # Load data
    noisy_imgs_1, noisy_imgs_2 = torch.load("../../data/train_data.pkl")
    noisy_imgs_test , clean_images = torch.load("../../data/val_data.pkl")

    clean_images = (clean_images / 255.0).float()

    # Select training samples
    train_input = noisy_imgs_1
    train_targets = noisy_imgs_2

    # Evaluate on untrained or loaded
    results = model.predict(noisy_imgs_test) / 255.0
    ps = compute_psnr(results.cpu(), clean_images)
    print(f'Initially: {ps}')

    # Train  
    start = time.time()
    num_epochs = 8
    model.train(train_input, train_targets, num_epochs)

    # Evaluate
    results = model.predict(noisy_imgs_test) / 255.0
    ps = compute_psnr(results.cpu(), clean_images)
    print(f'After training: {ps}')
    end = time.time()
    duration = (end - start) / 60.0
    print(f'Training and prediction took {duration} minutes')

    # Save trained model
    save = False
    if save:
        print("Saving model...")
        with open('bestmodel_pickle.pth','wb') as outfile:
            pickle.dump(model.model.state_dict(), outfile)
        torch.save(model.model.state_dict(), 'bestmodel_torch.pth')

def convert_torch_to_pickle():
    print("Loading origi dict")
    best_model_state_dict = torch.load("bestmodel.pth")
    print("Saving origi dict with pickle")
    with open('bestmodel_pickle.pth','wb') as outfile:
            pickle.dump(best_model_state_dict, outfile)
    print("Loading origi dict with pickle")
    with open('bestmodel_pickle.pth','rb') as dict:
        best_model_state_dict_from_pickle = pickle.load(dict)
    
    origi_vals = best_model_state_dict.values()
    new_vals = best_model_state_dict_from_pickle.values()
    print("Comparing origi dict with pickle dict")
    for (el1, el2) in list(zip(origi_vals, new_vals)):
        torch.allclose(el1,el2)
    print("Success!")

if __name__ == "__main__":
    main()