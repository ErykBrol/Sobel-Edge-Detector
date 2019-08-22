from sobel_operator import *

if __name__ == '__main__':
    img = Image.open('cat.png')
    img_as_array = np.array(img, dtype=float)

    # Apply sobel edge detection
    edges = sobel_edge_detect(img_as_array)

    output_img = Image.fromarray(edges.astype(np.uint8), 'L')
    output_img.save('cat.png')
