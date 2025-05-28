#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <map>

using namespace cv;
using namespace std;

class Image {
public:
    int w, h, c;
    unsigned char* data;
    vector<pair<int, int>> gradyan;

    Image(int width, int height, int color) {
        w = width; h = height; c = color;
        data = new unsigned char[h * w * c];
    }

    ~Image() = default;
};

Mat   ConvertToMat(const Image& img) {
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

Image ComputeGradient(const Image& gray_image) {
    Image gradient(gray_image.w, gray_image.h, 1);

    gradient.gradyan = std::vector<std::pair<int, int>>(gray_image.w * gray_image.h, { 0, 0 });
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

            gradient.gradyan[y * gray_image.w + x] = { gx, gy };

            int magnitude = sqrt(gx * gx + gy * gy);
            magnitude = min(255, magnitude); // aşırı değerleri kırp


            gradient.data[y * gray_image.w + x] = magnitude;
        }
    }

    return gradient;
}

Image NonMaximumSupression(const Image& gradient_image) {
    Image result(gradient_image.w, gradient_image.h, 1);
    for (int y = 1; y < gradient_image.h - 1; y++) {
        for (int x = 1; x < gradient_image.w - 1; x++) {
            int gx = gradient_image.gradyan[y * gradient_image.w + x].first; // X yönündeki gradient
            int gy = gradient_image.gradyan[y * gradient_image.w + x].second; // Y yönündeki gradient

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

Image HysteresisThreshold(const Image& nms_img, int low_thresh = 20, int high_thresh = 70) {
    Image result(nms_img.w, nms_img.h, 1);

    for (int y = 1; y < nms_img.h - 1; y++) {
        for (int x = 1; x < nms_img.w - 1; x++) {
            int idx = y * nms_img.w + x;
            int val = nms_img.data[idx];

            if (val >= high_thresh) {
                result.data[idx] = 255; // güçlü kenar
            }
            else if (val >= low_thresh) {
                // 8 komşusuna bak, biri güçlü kenar mı?
                bool connected_to_strong = false;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if (dy == 0 && dx == 0) continue;
                        int n_y = y + dy;
                        int n_x = x + dx;
                        if (n_y >= 0 && n_y < nms_img.h && n_x >= 0 && n_x < nms_img.w) {
                            int n_idx = n_y * nms_img.w + n_x;
                            if (nms_img.data[n_idx] >= high_thresh) {
                                connected_to_strong = true;
                            }
                        }
                    }
                }
                result.data[idx] = connected_to_strong ? 255 : 0;
            }
            else {
                result.data[idx] = 0;
            }
        }
    }

    // Kenar piksellerini sıfırla
    for (int x = 0; x < nms_img.w; x++) {
        result.data[x] = 0;
        result.data[(nms_img.h - 1) * nms_img.w + x] = 0;
    }
    for (int y = 0; y < nms_img.h; y++) {
        result.data[y * nms_img.w] = 0;
        result.data[y * nms_img.w + nms_img.w - 1] = 0;
    }

    return result;
}

vector<Vec2f> LineDetection(string path_name) {
    Mat image = imread(path_name, IMREAD_COLOR);
    if (image.empty()) {
        cerr << "Görüntü Yüklenemedi!" << endl;
    }

    Image new_image = ConvertToImage(image);
    Image gray_scale_img = ConvertToGrayScale(new_image);
    Image gradient_img = ComputeGradient(gray_scale_img);
    Image NMS_img = NonMaximumSupression(gradient_img);
    Image hysteresis_img = HysteresisThreshold(NMS_img);


    // Hough Line Detection
    vector<Vec2f> lines;
    Mat img = ConvertToMat(hysteresis_img);
    HoughLines(img, lines, 1, CV_PI / 180, 120); // 132
    for (size_t i = 0; i < lines.size(); i++) {
        float rho = lines[i][0];
        float theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        line(image, pt1, pt2, Scalar(0, 0, 255), 2);
    }

    //imshow("Image", image);
    //waitKey(0);
    return lines;
}


Point2f computeIntersection(Vec2f line1, Vec2f line2) {
    float rho1 = line1[0], theta1 = line1[1];
    float rho2 = line2[0], theta2 = line2[1];

    float sin1 = sin(theta1), cos1 = cos(theta1);
    float sin2 = sin(theta2), cos2 = cos(theta2);

    float det = cos1 * sin2 - sin1 * cos2;
    if (fabs(det) < 1e-10) // çizgiler paralelse
        return Point2f(-1, -1);

    float x = (sin2 * rho1 - sin1 * rho2) / det;
    float y = (-cos2 * rho1 + cos1 * rho2) / det;

    return Point2f(x, y);
}


vector<Point2f> findPoint(vector<Vec2f> lines, Mat image) {
    vector<Point2f> intersections;
    for (size_t i = 0; i < lines.size(); i++) {
        for (size_t j = i + 1; j < lines.size(); j++) {
            Point2f pt = computeIntersection(lines[i], lines[j]);
            if (pt.x >= 0 && pt.y >= 0 && pt.x < image.cols && pt.y < image.rows) {
                intersections.push_back(pt);
            }
        }
    }

    Point2f topLeft, topRight, bottomLeft, bottomRight;
    float minSum = FLT_MAX, maxSum = -FLT_MAX;
    float minDiff = FLT_MAX, maxDiff = -FLT_MAX;

    for (auto& pt : intersections) {
        float sum = pt.x + pt.y;
        float diff = pt.y - pt.x;

        if (sum < minSum) { minSum = sum; topLeft = pt; }
        if (sum > maxSum) { maxSum = sum; bottomRight = pt; }
        if (diff < minDiff) { minDiff = diff; topRight = pt; }
        if (diff > maxDiff) { maxDiff = diff; bottomLeft = pt; }
    }

    vector<Point2f> corners = { topLeft, topRight, bottomRight, bottomLeft };
    for (const auto& corner : corners) {
        cout << "Corner: " << corner << endl;
    }
    return corners;

}

int main()
{
    string image_path1 = "../1.jpg"; // Orijinal resim
    string image_path2 = "../2.jpg"; // Farklı açıdan çekilmiş resim

    
    Mat image1 = imread(image_path1, IMREAD_COLOR);
    Mat image2 = imread(image_path2, IMREAD_COLOR);


    vector<Vec2f> lines1 = LineDetection(image_path1);
    vector<Vec2f> lines2 = LineDetection(image_path2);
    
    vector<Point2f> corners = findPoint(lines1, image1);
    vector<Point2f> dst_corners = findPoint(lines2, image2);

    vector<pair<int, int>> a = { {0, 255}, {255, 0}, {255, 255} };

    // Köşegenleri görmek için
    for (int i = 0; i < 3; i++) {
        line(image1, corners[i], corners[i], Scalar(0, a[i].first, a[i].second), 5);
    }

    for (int i = 0; i < 3; i++) {
        line(image2, dst_corners[i], dst_corners[i], Scalar(0, a[i].first, a[i].second), 5);
    }

    Mat H = findHomography(dst_corners, corners);
    Mat aligned;
    warpPerspective(image2, aligned, H, image1.size()); // orijinal boyuta göre hizala

    imshow("Orijinal Resim", image1);
    imshow("Perspektif Resim ", image2);
    imshow("Aligned Resim", aligned);
    waitKey(0);
}

