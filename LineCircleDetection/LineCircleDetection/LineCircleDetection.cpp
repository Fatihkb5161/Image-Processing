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


Image ConvertToGrayScale(const Image& input_image) {
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
    if (img.channels() == 3) {
        for (int i = 0; i < image.h; i++) {
            for (int j = 0; j < image.w; j++) {
                Vec3b pixel = img.at<Vec3b>(i, j);
                image.data[k++] = pixel[2];
                image.data[k++] = pixel[1];
                image.data[k++] = pixel[0];
            }
        }
    }
    else if (img.channels() == 1) {
        for (int i = 0; i < image.h; i++) {
            for (int j = 0; j < image.w; j++) {
                uchar pixel = img.at<uchar>(i, j);
                image.data[k++] = pixel;
            }
        }
    }
    
    return image;
}

Image ComputeGradient(const Image& gray_image) {
    Image gradient(gray_image.w, gray_image.h, 1);

    int sobel_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int sobel_y[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}

    };

    for (int y = 1; y < gray_image.h - 1; y++) {
        for (int x = 1; x < gray_image.w - 1; x++) {
            int gx = 0, gy = 0;
            
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    int pixel = gray_image.data[(y + i) * gray_image.w + (x + j)];
                    gx += pixel * sobel_x[i + 1][j + 1];
                    gy += pixel * sobel_y[i + 1][j + 1];
                }
            }

            int magnitude = sqrt(gx * gx + gy * gy);
            magnitude = min(255, magnitude); // aşırı değerleri kırp

            gradient.data[y * gray_image.w + x] = magnitude;
        }
    }
    
    return gradient;
}

Image NonMaximumSupression(const Image& gradient_image) {
    Image result(gradient_image.w, gradient_image.h, 1);

    int sobel_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int sobel_y[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}

    };

    for (int y = 1; y < gradient_image.h - 1; y++) {
        for (int x = 1; x < gradient_image.w - 1; x++) {
            int gx = 0, gy = 0;

            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    int pixel = gradient_image.data[(y + i) * gradient_image.w + (x + j)];
                    gx += pixel * sobel_x[i + 1][j + 1];
                    gy += pixel * sobel_y[i + 1][j + 1];
                }
            }

            float angle = atan2(gy, gx) * 180.0 / CV_PI;
            if (angle < 0) angle += 180;

            int current = gradient_image.data[y * gradient_image.w + x];
            int neighbour1 = 0, neighbour2 = 0;

            // Komşuları belirle

            if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
                neighbour1 = gradient_image.data[y * gradient_image.w + (x - 1)];
                neighbour2 = gradient_image.data[y * gradient_image.w + (x + 1)];
            } 
            else if (angle >= 22.5 && angle < 67.5) {
                neighbour1 = gradient_image.data[(y - 1) * gradient_image.w + (x + 1)];
                neighbour2 = gradient_image.data[(y + 1) * gradient_image.w + (x - 1)];
            }
            else if (angle >= 67.5 && angle < 112.5) {
                neighbour1 = gradient_image.data[(y - 1) * gradient_image.w + x];
                neighbour2 = gradient_image.data[(y + 1) * gradient_image.w + x];
            }
            else if (angle >= 112.5 && angle < 157.5) {
                neighbour1 = gradient_image.data[(y - 1) * gradient_image.w + (x - 1)];
                neighbour2 = gradient_image.data[(y + 1) * gradient_image.w + (x + 1)];
            }

            if (current >= neighbour1 && current >= neighbour2) {
                result.data[y * gradient_image.w + x] = current;
            }
            else {
                result.data[y * gradient_image.w + x] = 0;
            }
        }

    }
    return result;
}

int main()
{
    string image_path = "C:/Users/mefat/OneDrive/Masaüstü/ImageProcessing0.1/LineCircleDetection/image5.jpg";
    Mat image = imread(image_path, IMREAD_COLOR);
    if (image.empty()) {
        cerr << "Görüntü Yüklenemedi!" << endl;
        return -1;
    }
    // OpenCV Mat -> Kendi Image sınıfımıza aktarım ve BGR to RGB
    Image new_image = ConvertToImage(image);
    Image gray_scale_img = ConvertToGrayScale(new_image);
    Image binary_img = ConvertToBinary(gray_scale_img);
    Image gradient_img = ComputeGradient(gray_scale_img);
    Image NMS_img = NonMaximumSupression(gradient_img);

    imshow("Original Image", image);
    imshow("Gray Scaled Image", ConvertToMat(gray_scale_img));
    imshow("Gradient Computed Image", ConvertToMat(gradient_img));
    imshow("NMS Image", ConvertToMat(NMS_img));
    waitKey(0);

}
