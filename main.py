from sobel_operator import *

if __name__ == '__main__':
    img = Image.open('C:\\Users\\Eryk\\Desktop\\cat.jpg')
    img_as_array = np.array(img, dtype=float)

    ret = sobel_gradient(img_as_array)
    ret *= 255.0 / np.max(ret)

    output_img = Image.fromarray(gaussed.astype(np.uint8), 'L')
    output_img.save('C:\\Users\\Eryk\\Desktop\\cat-processed.png')
