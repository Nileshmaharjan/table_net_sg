import cv2
import numpy as np
import camelot

predicted_mask_image = cv2.imread('predicted_mask.png')
original_image = cv2.imread('Foam_2nd(0).png')

original_width, original_height = original_image.shape[1], original_image.shape[0]
predicted_width, predicted_height = predicted_mask_image.shape[1], predicted_mask_image.shape[0]

dim = (original_width, original_height)

x_scale = original_width / predicted_width
y_scale = original_height / predicted_height

# [272, 32, 544, 112], [752, 144, 1008, 448], [784, 704, 1008, 1008]
bbox = np.array([[272, 32, 544, 112]])
bbox[:, 0] = bbox[:, 0] * x_scale
bbox[:, 1] = bbox[:, 1] * y_scale
bbox[:, 2] = bbox[:, 2] * x_scale
bbox[:, 3] = bbox[:, 3] * y_scale

converted_bbox = bbox

for i in converted_bbox:
    table_area_region = i.tolist()
    table_area_region_mod = ','.join(map(str,table_area_region))
    tables = camelot.read_pdf('Foam_2nd(0).pdf', table_areas=[table_area_region_mod])

    cell_array = []
    for single_table in tables:
        cells = single_table.cells
        for cell in cells:
            for c in cell:
                cell_json = {
                    'x1': c.x1,
                    'y1': c.y1,
                    'x2': c.x2,
                    'y2': c.y2
                }
                cell_array.append(cell_json)

    print("cell_array", cell_array)