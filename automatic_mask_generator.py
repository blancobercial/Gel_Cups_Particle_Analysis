import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor



# Global variables
IMG_NAME = 'Nov21_150B_L1_F1'
EXT = '.tiff'
IMG_PATH = IMG_NAME + EXT
SCALE_PATH = IMG_NAME + '_Scale' + EXT



# Define helper functions
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=150):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='.', s=marker_size, edgecolor='white', linewidth=0.5)
    #ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='.', s=marker_size, edgecolor='white', linewidth=0.5)

clicked_pt = (0, 0)
def get_mouse_click(event, x, y, flags, params):
    global clicked_pt
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pt = x, y



# Read all images
img = cv2.imread(IMG_PATH)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#scale = cv2.imread(SCALE_PATH, cv2.IMREAD_GRAYSCALE)
scale = cv2.imread(SCALE_PATH)
#scale = cv2.cvtColor(scale, cv2.COLOR_BGR2RGB)



# Get SAM
sam_checkpoint = 'sam_vit_h_4b8939.pth'
model_type = 'vit_h'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    #points_per_side=32,
    #pred_iou_thresh=0.86,
    #stability_score_thresh=0.92,
    #crop_n_layers=1,
    #crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)



# Generate masks for image
img_masks = mask_generator.generate(img)
'''
img_masks[i] = {
    'segmentation': bool np array with same shape as img an entry is True if that ppx is in the mask and False if it is not
    'area': area of mask in pixels (equivalent to summing over entire array (T is 1 and F is 0)),
    'bbox': boundary box of the mask in (x, y, width, height) format,
    'predicted_iou': the model's own prediction for the quality of the mask,
    'point_coords': the sampled input point that generated this mask,
    'stability_score': an additional measure of mask quality,
    'crop_box': the crop of the img used to generate this mask in (x, y, width, height) format
}
'''



# Get mask that corresponds to center pixel in scale image
predictor = SamPredictor(sam)
predictor.set_image(scale)
cv2.imshow('Scale img', scale)
cv2.setMouseCallback('Scale img', get_mouse_click)
cv2.waitKey(0)
clicked_pt = np.array([[*clicked_pt]])
in_label = np.array([1])



# Generate mask for scale
scale_mask, scores, logits = predictor.predict(
    point_coords=clicked_pt,
    point_labels=in_label,
    multimask_output=True,
)
areas = np.sum(scale_mask, axis=(0,2))
indx = np.argmin(areas)
scale_mask = scale_mask[indx]
scale_mask = np.squeeze(scale_mask) # scale_mask.shape: (1,x,y) --> (x,y)
# scale_mask is a bool np array with same shape as img an entry is True if that ppx is in the mask and False if it is not
plt.figure()
plt.imshow(scale)
show_mask(scale_mask, plt.gca())
show_points(clicked_pt, in_label, plt.gca())
plt.axis('off')
plt.show()



# Get pixels per millimeter
ppx_per_mm = np.sum(scale_mask, axis=1)
ppx_per_mm = ppx_per_mm[ppx_per_mm > 0]
ppx_per_mm = np.mean(ppx_per_mm)
print(f'Pixels per millimeter: {ppx_per_mm}')



# Get area in square milimeters
for mask in img_masks:
    area_ppx = mask.pop('area')
    mask['area_ppx'] = area_ppx
    mask['area_mm^2'] = area_ppx / ppx_per_mm ** 2