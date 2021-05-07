import io
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import streamlit as st
from generator import ResidualBlock, UpsampleBLock, Generator
from critic import Critic


@st.cache()
def load_model(path, z_dim, im_chan, device):
    gen = Generator(z_dim=z_dim, im_chan=im_chan).to(device)
    # crit = Critic(im_chan=im_chan).to(device)
    # try:
    checkpoint = torch.load(path, map_location=torch.device(device))
    # except Exception as e:
    #     print(e)
    gen.load_state_dict(checkpoint)

    # print('Checkpoint Loaded')
    return gen


def load_images(image_list_stream = io.BytesIO()):
    image_list = []
    for image in image_list_stream:
        image.seek(0)
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) /255.0
        image = image_resize(image, height=64)
        image_list.append(image)

    transform = transforms.Compose([
            transforms.ToTensor()
        ])

    for img in image_list:
        yield transform(img).unsqueeze(0).float()
    

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation = inter)

    return resized