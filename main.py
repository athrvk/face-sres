import torchvision.transforms as transforms
from torchvision.utils import make_grid
from utils import image_resize, load_model, load_images 
import streamlit as st
from generator import ResidualBlock, UpsampleBLock, Generator
from critic import Critic


def file_selector(folder_path='.'):
    uploadedFiles = st.file_uploader("Select Input Image", type= ['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    return uploadedFiles

def predict(gen, images):
    multiple = True if (len(images)>1) else False
    st.write('## Original Image' + ('s' if multiple else ''))
    img_list = [i.permute(0,2,3,1).squeeze().numpy() for i in load_images(images)]
    for idx, col in enumerate(st.beta_columns(len(images))):
        col.image(img_list[idx], clamp=True)

    st.write('## Upscaled Image' + ('s' if multiple else ''))
    with st.spinner("Drawing the perfect upscaled images.."):
        for image in load_images(images):
            output = gen(image)
            st.image(output.detach().permute(0,2,3,1).squeeze().numpy(), clamp=True)


if __name__ == '__main__':
    gen = load_model('checkpoint_gen.pth', 3, 3, 'cpu')
    st.title("Single Face Image Super Resolution!")

    images = file_selector()

    if st.button('Upscale'):
        if not images:
            st.error("Please Select a file!") 
        predict(gen, images)


