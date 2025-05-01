#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


Mat ShowGrayScaleImage(Mat img) {
    Mat image = img;
    if (image.empty()) {
        cerr<<"Resim açılamadı! Dosya yolunu kontrol et."<<endl;
        return Mat();
    }

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

    auto RGBToInt = [](Image& im) {
        unsigned char* row1 = new unsigned char[im.h * im.w];
        for (int row = 0; row < im.h; row++) {
            for (int col = 0; col < im.w; col++) {
                row1[row * im.w + col] =
                    0.3 * im.data[row * im.w * im.c + col * im.c + 2] +
                    0.59 * im.data[row * im.w * im.c + col * im.c + 1] +
                    0.11 * im.data[row * im.w * im.c + col * im.c];
            }
        }
        return row1;
        };

    Image image_info(image.cols, image.rows, image.channels());

    int counter = 0;
    for (int i = 0; i < image_info.h; i++) {
        for (int j = 0; j < image_info.w; j++) {
            Vec3b intensity = image.at<Vec3b>(i, j);
            image_info.data[counter++] = intensity[0]; // B
            image_info.data[counter++] = intensity[1]; // G
            image_info.data[counter++] = intensity[2]; // R
        }
    }

    unsigned char* gray_data = RGBToInt(image_info);
    Mat gray_image(image_info.h, image_info.w, CV_8UC1);
    memcpy(gray_image.data, gray_data, image_info.h * image_info.w);
    return gray_image;

}




int main()
{
    string image_path = "C:/Users/mefat/OneDrive/Masaüstü/ImageProcessing0.1/LineCircleDetection/image2.jpg";
    Mat image = imread(image_path, IMREAD_COLOR);

    Mat gray_scaled_image = ShowGrayScaleImage(image);
    
    imshow("Original Picture", image);
    imshow("Gray-Scale Picture", gray_scaled_image);
    waitKey(0);

}
