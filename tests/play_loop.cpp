#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    std::string path = "records/test_change.mp4";
    if (argc > 1) path = argv[1];

    cv::VideoCapture cap(path);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << path << std::endl;
        return 1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 30;
    int delay = static_cast<int>(1000.0 / fps);

    std::cout << "Playing " << path << " at ~" << fps << " fps (delay=" << delay << " ms)" << std::endl;

    cv::Mat frame;
    while (true) {
        // Read until the end
        if (!cap.read(frame) || frame.empty()) {
            // restart
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }

        cv::imshow("loop", frame);
        int key = cv::waitKey(delay);
        if (key == 'q' || key == 27) break; // q or ESC
        // allow faster stepping with space
        if (key == ' ') cv::waitKey(0);
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
