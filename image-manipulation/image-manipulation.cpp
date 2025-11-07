// OpenCV image processing demo with interactive parameter controls
// Demonstrates flipping, grayscale conversion, blurring, edge detection,
// and includes interactive windows with trackbars for experimentation

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>

// Utility: Display image with optional scaling for large images
static void safeImShow(const std::string& winName, const cv::Mat& img, int maxSide = 1000)
{
    cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);
    if (img.empty()) {
        cv::imshow(winName, img);
        return;
    }

    const int h = img.rows, w = img.cols;
    const int longest = std::max(h, w);
    if (maxSide > 0 && longest > maxSide) {
        const double scale = static_cast<double>(maxSide) / static_cast<double>(longest);
        cv::Mat scaled;
        cv::resize(img, scaled, cv::Size(), scale, scale, cv::INTER_AREA);
        cv::imshow(winName, scaled);
    } else {
        cv::imshow(winName, img);
    }
}

// Utility: Show and position window on screen
static void showAndPlace(const std::string& winName, const cv::Mat& img, int x, int y, int maxSide = 1000)
{
    safeImShow(winName, img, maxSide);
    cv::moveWindow(winName, x, y);
}

// Convert slider values to usable parameters
static inline double sliderToSigma(int v) { return static_cast<double>(v) / 10.0; }
static inline int sliderToOddKernel(int v) { return 2 * v + 1; }

// Context for interactive smoothing window
struct SmoothingUIContext {
    cv::Mat gray;
    std::string winName;
    std::string trackName;
    int sigmaInit = 20;
};

// Callback for smoothing trackbar
static void onSmoothingChange(int /*pos*/, void* userdata)
{
    auto* ctx = reinterpret_cast<SmoothingUIContext*>(userdata);
    if (!ctx) return;

    int slider = cv::getTrackbarPos(ctx->trackName, ctx->winName);
    double sigma = sliderToSigma(slider);

    cv::Mat blurred, edges;
    cv::GaussianBlur(ctx->gray, blurred, cv::Size(0, 0), sigma, sigma);
    cv::Canny(blurred, edges, 20, 60);

    safeImShow(ctx->winName, edges);
}

// Context for edge detection lab window
struct EdgeLabContext {
    cv::Mat gray;
    std::string winName;
    std::string tkK   = "Blur kernel";
    std::string tkSig = "Sigma x10";
    std::string tkT1  = "Canny threshold 1";
    std::string tkT2  = "Canny threshold 2";
    int initK   = 3;
    int initSig = 20;
    int initT1  = 20;
    int initT2  = 60;
};

// Callback for edge lab trackbars
static void onEdgeLabChange(int /*pos*/, void* userdata)
{
    auto* ctx = reinterpret_cast<EdgeLabContext*>(userdata);
    if (!ctx) return;

    int kSlider     = cv::getTrackbarPos(ctx->tkK,   ctx->winName);
    int sigmaSlider = cv::getTrackbarPos(ctx->tkSig, ctx->winName);
    int thr1Slider  = cv::getTrackbarPos(ctx->tkT1,  ctx->winName);
    int thr2Slider  = cv::getTrackbarPos(ctx->tkT2,  ctx->winName);

    int ksize = sliderToOddKernel(kSlider);
    double sigma = sliderToSigma(sigmaSlider);
    double th1 = static_cast<double>(thr1Slider);
    double th2 = static_cast<double>(thr2Slider);

    cv::Mat blurred, edges;
    cv::GaussianBlur(ctx->gray, blurred, cv::Size(ksize, ksize), sigma, sigma);
    cv::Canny(blurred, edges, th1, th2);

    safeImShow(ctx->winName, edges);
}

int main()
{
    // Load input image
    const std::string inputPath = "flower.jpg";
    cv::Mat input = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (input.empty()) {
        std::cerr << "Error: Could not load '" << inputPath << "'\n";
        return 1;
    }

    // Window layout parameters
    const int CELL_W  = 460;
    const int CELL_H  = 340;
    const int START_X = 40;
    const int START_Y = 40;
    const int MAXSIDE = 420;

    // Fixed processing pipeline - each step displayed in its own window
    showAndPlace("01 Original", input, START_X + 0*CELL_W, START_Y + 0*CELL_H, MAXSIDE);

    cv::Mat flippedVert;
    cv::flip(input, flippedVert, 0);
    showAndPlace("02 Flip Vertical", flippedVert, START_X + 1*CELL_W, START_Y + 0*CELL_H, MAXSIDE);

    cv::Mat flippedHoriz;
    cv::flip(flippedVert, flippedHoriz, 1);
    showAndPlace("03 Flip Horizontal", flippedHoriz, START_X + 2*CELL_W, START_Y + 0*CELL_H, MAXSIDE);

    cv::Mat rotated180;
    cv::flip(input, rotated180, -1);
    showAndPlace("04 Rotate 180", rotated180, START_X + 0*CELL_W, START_Y + 1*CELL_H, MAXSIDE);

    cv::Mat gray;
    cv::cvtColor(flippedHoriz, gray, cv::COLOR_BGR2GRAY);
    showAndPlace("05 Grayscale", gray, START_X + 1*CELL_W, START_Y + 1*CELL_H, MAXSIDE);

    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(0,0), 2.0, 2.0);
    showAndPlace("06 Blurred", blurred, START_X + 2*CELL_W, START_Y + 1*CELL_H, MAXSIDE);

    cv::Mat edges;
    cv::Canny(blurred, edges, 20, 60);
    showAndPlace("07 Edges", edges, START_X + 0*CELL_W, START_Y + 2*CELL_H, MAXSIDE);

    if (!cv::imwrite("output.jpg", edges))
        std::cerr << "Warning: Failed to write output.jpg\n";
    else
        std::cout << "Saved edges to output.jpg\n";

    // Interactive smoothing window with trackbar
    SmoothingUIContext smoothCtx;
    smoothCtx.gray      = gray;
    smoothCtx.winName   = "Interactive Smoothing";
    smoothCtx.trackName = "Sigma x10 (0-100)";
    smoothCtx.sigmaInit = 20;

    cv::namedWindow(smoothCtx.winName, cv::WINDOW_AUTOSIZE);
    cv::createTrackbar(smoothCtx.trackName, smoothCtx.winName, nullptr, 100,
                       onSmoothingChange, &smoothCtx);
    cv::setTrackbarPos(smoothCtx.trackName, smoothCtx.winName, smoothCtx.sigmaInit);
    onSmoothingChange(0, &smoothCtx);
    cv::moveWindow(smoothCtx.winName, START_X + 1*CELL_W, START_Y + 2*CELL_H);

    // Edge detection lab with multiple trackbars
    EdgeLabContext lab;
    lab.gray    = gray;
    lab.winName = "Edge Detection Lab";

    cv::namedWindow(lab.winName, cv::WINDOW_AUTOSIZE);
    cv::createTrackbar(lab.tkK,   lab.winName, nullptr, 15,  onEdgeLabChange, &lab);
    cv::createTrackbar(lab.tkSig, lab.winName, nullptr, 100, onEdgeLabChange, &lab);
    cv::createTrackbar(lab.tkT1,  lab.winName, nullptr, 255, onEdgeLabChange, &lab);
    cv::createTrackbar(lab.tkT2,  lab.winName, nullptr, 255, onEdgeLabChange, &lab);

    cv::setTrackbarPos(lab.tkK,   lab.winName, lab.initK);
    cv::setTrackbarPos(lab.tkSig, lab.winName, lab.initSig);
    cv::setTrackbarPos(lab.tkT1,  lab.winName, lab.initT1);
    cv::setTrackbarPos(lab.tkT2,  lab.winName, lab.initT2);
    onEdgeLabChange(0, &lab);
    cv::moveWindow(lab.winName, START_X + 2*CELL_W, START_Y + 2*CELL_H);

    // Additional effect: bilateral filter + color mapping
    // Bilateral filter smooths while preserving edges
    // Color map applies a vivid color gradient for artistic effect
    cv::Mat bilateral, stylized;
    cv::bilateralFilter(input, bilateral, 9, 75, 75);
    cv::applyColorMap(bilateral, stylized, cv::COLORMAP_TURBO);
    showAndPlace("08 Stylized Effect", stylized, START_X + 3*CELL_W, START_Y + 2*CELL_H, MAXSIDE);

    if (!cv::imwrite("output_effect.jpg", stylized))
        std::cerr << "Warning: Failed to write output_effect.jpg\n";
    else
        std::cout << "Saved stylized effect to output_effect.jpg\n";

    // Main loop - wait for ESC or 'q' to exit
    for (;;) {
        int key = cv::waitKey(30);
        if (key == 27 || key == 'q' || key == 'Q')
            break;
    }

    cv::destroyAllWindows();
    return 0;
}