import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pickle
import numpy as np
import tqdm

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation
from IPython.display import HTML

class ConvLinearAutoencoder(nn.Module):
    def __init__(self, latent_space):
        super(ConvLinearAutoencoder, self).__init__()
        self.latent_space = latent_space
        self.Encoder = nn.Sequential(
            nn.Unflatten(1, (3, 32, 32)),
            nn.Conv2d(3, 8, kernel_size=3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )
        self.CNN, self.CNN_flatten = self._get_conv_output((3072,), self.Encoder)
        self.Encoder.append(nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.CNN_flatten, latent_space),
            nn.Tanh()
        ))
        
        self.Decoder = nn.Sequential(
            nn.Linear(latent_space, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 3072),
            nn.Tanh()
        )
        
    def _get_conv_output(self, shape, layers):
        bs = 1
        dummy_x = torch.empty(bs, *shape)
        x = layers(dummy_x)
        CNN = x.size()
        CNN_flatten = x.flatten(1).size(1)
        return CNN, CNN_flatten
    
    def forward(self, x):
        return self.Decoder(self.Encoder(x))

def unpickle(file):
    import pickle
    with open(file, 'rb') as f:
        obj = pickle.load(f, encoding='latin1')
    return obj

def load_dataset(path, class_label):
    datadict = unpickle(path)
    data = datadict['data']
    labels = datadict['labels']
    
    dataset = []
    for image, label in zip(data, labels):
        if label == class_label:
            image = np.asarray(image, dtype=np.float32)
            image = (image - 127.5) / 127.5
            dataset.append(image)
    return dataset

train_data = load_dataset('cifar-10-batches-py/data_batch_1', 2)

model = torch.load('conv_linear.pkl', map_location=torch.device('cpu'))
model.eval()
encoder = model.Encoder
decoder = model.Decoder

def interactive_modification(encoder, decoder, images):
    core_data = []
    
    def on_slider_update(ax_mod, decoder, feature, val):
        nonlocal core_data
        core_data[feature] = val
        image = decoder(torch.from_numpy(core_data).unsqueeze(0)).detach().squeeze(0).numpy()
        ax_mod.set_array(image.reshape(3, 32, 32).transpose([1, 2, 0]) / 2 + 0.5)

    def on_button_click(ax_in, ax_out, ax_mod, encoder, decoder, images):
        nonlocal core_data
        image = images[np.random.randint(len(images))]
        core_data = encoder(torch.from_numpy(image).unsqueeze(0)).detach().squeeze(0).numpy()
        output = decoder(torch.from_numpy(core_data).unsqueeze(0)).detach().squeeze(0).numpy()
        ax_in.set_array(image.reshape(3, 32, 32).transpose([1, 2, 0]) / 2 + 0.5)
        ax_out.set_array(output.reshape(3, 32, 32).transpose([1, 2, 0]) / 2 + 0.5)
        ax_mod.set_array(output.reshape(3, 32, 32).transpose([1, 2, 0]) / 2 + 0.5)
    
    plt.rcParams['figure.dpi'] = 100
    fig, axes = plt.subplots(2, 2)
    fig.canvas.header_visible = False
    fig.tight_layout()
    
    axes[0, 0].set(visible=False)
    
    axes[0, 1].set(title='Вход', aspect='equal')
    ax_in = axes[0, 1].imshow([[0]])#, interpolation='bicubic')
        
    axes[1, 0].set(title='Мод. выход', aspect='equal')
    ax_mod = axes[1, 0].imshow([[0]])#, interpolation='bicubic')
    
    axes[1, 1].set(title='Выход', aspect='equal')
    ax_out = axes[1, 1].imshow([[0]])#, interpolation='bicubic')
    
    for ax in [i for j in axes for i in j]:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    features = np.random.randint(0, 96, 3)
    
    axs1 = fig.add_axes([0.85, 0.25, 0.0255, 0.65])
    slid1 = Slider(ax=axs1, label='#1', valinit=0, valmin=-1.0, valmax=1.0, orientation='vertical')
    slid1.on_changed(lambda val: on_slider_update(ax_mod, decoder, features[0], val))
    
    axs2 = fig.add_axes([0.9, 0.25, 0.0255, 0.65])
    slid2 = Slider(ax=axs2, label='#2', valinit=0, valmin=-1.0, valmax=1.0, orientation='vertical')
    slid2.on_changed(lambda val: on_slider_update(ax_mod, decoder, features[1], val))
    
    axs3 = fig.add_axes([0.95, 0.25, 0.0255, 0.65])
    slid3 = Slider(ax=axs3, label='#3', valinit=0, valmin=-1.0, valmax=1.0, orientation='vertical')
    slid3.on_changed(lambda val: on_slider_update(ax_mod, decoder, features[2], val))
    
    axs4 = fig.add_axes([0.8625, 0.1, 0.1, 0.075])
    btn = Button(axs4, 'Случ.')
    btn.on_clicked(lambda event: on_button_click(ax_in, ax_out, ax_mod, encoder, decoder, images))
    on_button_click(ax_in, ax_out, ax_mod, encoder, decoder, images)
    
    plt.subplots_adjust(right=0.8)
    plt.show()

interactive_modification(encoder, decoder, train_data)