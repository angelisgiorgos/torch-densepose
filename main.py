import torch
import PIL.Image
import PIL.ImageDraw
import numpy as np
# from densepose import DensePose
from densepose import densepose
import cv2
from torch.utils.mobile_optimizer import optimize_for_mobile


def pad_to_square(image):
    """
    Pad image to square shape.
    """
    height, width = image.shape[:2]

    if width < height:
        border_width = (height - width) // 2
        image = cv2.copyMakeBorder(image, 0, 0, border_width, border_width,
                                   cv2.BORDER_CONSTANT, value=0)
    else:
        border_width = (width - height) // 2
        image = cv2.copyMakeBorder(image, border_width, border_width, 0, 0,
                                   cv2.BORDER_CONSTANT, value=0)

    return image
VERSION = 'v0.0.2'

# model = DensePose()
model = densepose()
# if pretrained:
base_url = f'https://github.com/Licht-T/torch-densepose/releases/download/{VERSION}'
# state_dict = torch.hub.load_state_dict_from_url(
#     f'{base_url}/densepose_pretrained_msra-resnet101.pth',
#     progress=True,
#     file_name=f'densepose_pretrained_msra-resnet101_{VERSION}.pth'
#     )
filename = f'densepose_pretrained_msra-resnet101_{VERSION}.pth'
state_dict = torch.load(f'densepose_pretrained_msra-resnet101_{VERSION}.pth')

key_mapping = {
    'rpn.head.conv.weight': 'rpn.rpn.head.conv.0.0.weight',
    'rpn.head.conv.bias': 'rpn.rpn.head.conv.0.0.bias',
    'rpn.head.cls_logits.weight': 'rpn.rpn.head.cls_logits.weight',
    'rpn.head.cls_logits.bias': 'rpn.rpn.head.cls_logits.bias',
    'rpn.head.bbox_pred.weight': 'rpn.rpn.head.bbox_pred.weight',
    'rpn.head.bbox_pred.bias': 'rpn.rpn.head.bbox_pred.bias',
    'roi_heads.box_head.fc6.weight':'roi_heads.roi_heads.box_head.fc6.weight',
    'roi_heads.box_head.fc6.bias': "roi_heads.roi_heads.box_head.fc6.bias",
    "roi_heads.box_head.fc7.weight": "roi_heads.roi_heads.box_head.fc7.weight",
    "roi_heads.box_head.fc7.bias": "roi_heads.roi_heads.box_head.fc7.bias",
    "roi_heads.box_predictor.cls_score.weight": "roi_heads.roi_heads.box_predictor.cls_score.weight",
    "roi_heads.box_predictor.cls_score.bias": "roi_heads.roi_heads.box_predictor.cls_score.bias",
    "roi_heads.box_predictor.bbox_pred.weight":"roi_heads.roi_heads.box_predictor.bbox_pred.weight",
    "roi_heads.box_predictor.bbox_pred.bias": "roi_heads.roi_heads.box_predictor.bbox_pred.bias"

    }

new_state_dict = {}
for key, value in state_dict.items():
    if key in key_mapping:
        new_key = key_mapping[key]
    else:
        new_key = key
    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict)


device = torch.device('cpu')
model.to(device)
model.eval()

scripting = True
onnx_export = False

image_path = 'D:/github-repos/certh_straps/demo/0006.png'

input_image = cv2.imread(image_path)
input_image = pad_to_square(input_image)
input_image = cv2.resize(input_image, (256, 256),
                         interpolation=cv2.INTER_LINEAR)
img_array = np.array(input_image, dtype=np.float32).transpose((2, 0, 1))
img_tensor = torch.from_numpy(img_array).unsqueeze(0)

results = model(img_tensor.to(device))


boxes = results[0]['boxes'].to('cpu')
scores = results[0]['scores'].to('cpu')
coarse_segs = results[0]['coarse_segs'].to('cpu')
fine_segs = results[0]['fine_segs'].to('cpu')

inp_img = PIL.Image.fromarray(input_image)
draw = PIL.ImageDraw.Draw(inp_img)
for box in boxes:
    draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)

inp_img.save('./data/out_2_box.jpg')

seg_img_array = np.zeros(img_array.shape[1:], dtype=np.uint8)
for coarse_seg, fine_seg in zip(coarse_segs, fine_segs):
    coarse_seg = coarse_seg.numpy().astype(np.uint8)
    fine_seg = fine_seg.numpy().astype(np.uint8)
    seg = 10 * fine_seg * coarse_seg

    cond = seg_img_array == 0
    seg_img_array[cond] = seg_img_array[cond] + seg[cond]

print(np.unique(seg_img_array))
seg_img = PIL.Image.fromarray(seg_img_array)
seg_img.save('./data/out_2_seg.jpg')

if scripting:
    scripted_model = torch.jit.script(model)
    scripted_model.save('./densepose_scripted.pt')

    # Export lite interpreter version model (compatible with lite interpreter)
    scripted_model._save_for_lite_interpreter('./densepose_scripted.ptl')

    optimized_scripted_model = optimize_for_mobile(scripted_model)
    # using optimized lite interpreter model makes inference about 60% faster than the non-optimized
    # lite interpreter model, which is about 6% faster than the non-optimized full jit model
    optimized_scripted_model._save_for_lite_interpreter('./densepose_scripted_optimized.ptl')

if onnx_export:
    input_names = ['input_image']
    output_names = ['boxes', 'labels', 'scores', 'coarse_segs', 'fine_segs', 'us', 'vs']

    dyn_axes = {
        'boxes': {0: 'batch_size', 1: 'bbox'}, "labels": {0: 'batch_size'},
        "scores": {0: 'batch_size'}, 'coarse_segs': {0: 'batch_size', 1: 'width', 2: 'height'},
        'fine_segs': {0: 'batch_size', 1: 'width', 2: 'height'},
        'us': {0: 'batch_size', 1: 'channels', 2: 'width', 3: 'height'},
        'vs': {0: 'batch_size', 1: 'channels', 2: 'width', 3: 'height'}}

    torch.onnx.export(model, (img_tensor.to(device),), 'densepose_3.onnx', opset_version=11,
                      input_names=input_names,
                      output_names=output_names,
                      do_constant_folding=True,
                      dynamic_axes=dyn_axes,
                      export_params=True)
