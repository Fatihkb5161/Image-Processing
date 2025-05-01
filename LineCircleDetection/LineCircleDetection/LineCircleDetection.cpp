#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class Image {
public:
    int w, h, c;
    unsigned char* data;

    Image(int width, int height, int color) {
        w = width; h = height; c = color;
        data = new unsigned char[h * w * c];
    }

    ~Image() {
        delete[] data;
    }
};


Image ShowGrayScaleImage(const Image& input_image) {
    Image gray(input_image.w, input_image.h, 1);

    for (int i = 0; i < input_image.h; i++) {
        for (int j = 0; j < input_image.w; j++) {
            int idx = (i * input_image.w + j) * input_image.c;
            unsigned char r = input_image.data[idx];
            unsigned char g = input_image.data[idx + 1];
            unsigned char b = input_image.data[idx + 2];
            gray.data[i * input_image.w + j] = 0.3 * r + 0.59 * g + 0.11 * b;
        }
    }
    return gray;
}

Image ConvertToBinary(const Image& gray, int threshold = 128) {
    Image binary(gray.w, gray.h, 1);

    for (int i = 0; i < gray.h; i++) {
        for (int j = 0; j < gray.w; j++) {
            int idx = i * gray.w + j;
            binary.data[idx] = (gray.data[idx] >= threshold) ? 255 : 0;
        }
    }
    return binary;
}

Mat ConvertToMat(const Image& img) {
    Mat output(img.h, img.w, CV_8UC1);
    memcpy(output.data, img.data, img.w * img.h);
    return output;
}

Image ConvertToImage(Mat img) {
    Image image(img.cols, img.rows, img.channels());

    int k = 0;
    for (int i = 0; i < image.h; i++) {
        for (int j = 0; j < image.w; j++) {
            Vec3b pixel = img.at<Vec3b>(i, j);
            image.data[k++] = pixel[2];
            image.data[k++] = pixel[1];
            image.data[k++] = pixel[0];
        }
    }
    return image;
}

int main()
{
    string image_path = "C:/Users/mefat/OneDrive/Masaüstü/ImageProcessing0.1/LineCircleDetection/image1.jpg";
    Mat image = imread(image_path, IMREAD_COLOR);
    if (image.empty()) {
        cerr << "Görüntü Yüklenemedi!" << endl;
        return -1;
    }
    // OpenCV Mat -> Kendi Image sınıfımıza aktarım ve BGR to RGB
    Image new_image = ConvertToImage(image);
    Image gray_scale_img = ShowGrayScaleImage(new_image);
    Image binary_img = ConvertToBinary(gray_scale_img);

    imshow("Original Image", image);
    imshow("Gray Scaled Image", ConvertToMat(gray_scale_img));
    imshow("Binary Image", ConvertToMat(binary_img));
    waitKey(0);

}
