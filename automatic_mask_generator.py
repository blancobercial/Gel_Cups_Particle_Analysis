import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from streamlit_image_coordinates import streamlit_image_coordinates
import streamlit as st
from PIL import Image



# Define global constants
MAX_WIDTH = 700



# Define helpful functions
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=20):
    pos_points = coords[labels==1]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='.', s=marker_size, edgecolor='white', linewidth=0.2)




# Get SAM
sam_checkpoint = 'sam_vit_h_4b8939.pth'
model_type = 'vit_h'
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    min_mask_region_area=200
)
predictor = SamPredictor(sam)



# Get uploaded files from user
scale = st.file_uploader('Upload Scale Image')
image = st.file_uploader('Upload Particle Image')



# Runs when scale image is uploaded
if scale:
    scale_np = np.asarray(bytearray(scale.read()), dtype=np.uint8)
    scale_np = cv2.imdecode(scale_np, 1)
    predictor.set_image(scale_np)
    
    scale_factor = scale_np.shape[1] / MAX_WIDTH # how many times larger scale_np is than the image shown for each dimension
    clicked_point = streamlit_image_coordinates(Image.open(scale.name), height=scale_np.shape[0] // scale_factor, width=MAX_WIDTH)
    if clicked_point:
        input_point = np.array([[clicked_point['x'], clicked_point['y']]]) * scale_factor
        input_point = input_point.astype(int)
        input_label = np.array([1])
        mask, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        areas = np.sum(mask, axis=(1,2))
        indx = np.argmin(areas)
        mask = mask[indx]
        mask = np.squeeze(mask) # mask.shape: (1,x,y) --> (x,y)
        mask = mask.astype(int)
        # mask is a bool np array with same shape as img an entry is True if that ppx is in the mask and False if it is not

        fig, ax = plt.subplots()
        ax.imshow(scale_np)
        show_mask(mask, ax)
        show_points(input_point, input_label, ax)
        ax.axis('off')
        st.pyplot(fig)



        # Get pixels per millimeter
        pixels_per_unit = np.sum(mask, axis=1)
        pixels_per_unit = pixels_per_unit[pixels_per_unit > 0]
        pixels_per_unit = np.mean(pixels_per_unit)



# Runs when image is uploaded
if image:
    image_np = np.asarray(bytearray(image.read()), dtype=np.uint8)
    image_np = cv2.imdecode(image_np, 1)
    predictor.set_image(image_np)
    
    scale_factor = image_np.shape[1] / MAX_WIDTH # how many times larger scale_np is than the image shown for each dimension
    clicked_point = streamlit_image_coordinates(Image.open(image.name), height=image_np.shape[0] // scale_factor, width=MAX_WIDTH)
    if clicked_point:
        input_point = np.array([[clicked_point['x'], clicked_point['y']]]) * scale_factor
        input_point = input_point.astype(int)
        input_label = np.array([1])
        mask, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        areas = np.sum(mask, axis=(1,2))
        indx = np.argmin(areas)
        mask = mask[indx]
        mask = np.squeeze(mask) # mask.shape: (1,x,y) --> (x,y)
        mask = mask.astype(int)
        # mask is a bool np array with same shape as img an entry is True if that ppx is in the mask and False if it is not

        fig, ax = plt.subplots()
        ax.imshow(image_np)
        show_mask(mask, ax)
        show_points(input_point, input_label, ax)
        ax.axis('off')
        st.pyplot(fig)



        # Get the area in square millimeters
        st.write(f'Area: {np.sum(mask) / pixels_per_unit ** 2} mm^2')