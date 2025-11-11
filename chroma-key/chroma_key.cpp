// Chroma key implementation (green screen technique)
// Replaces pixels of the most common color in foreground with background pixels
//
// Algorithm:
// 1. Build 3D color histogram of foreground image (manual implementation)
// 2. Find most common color bin
// 3. Replace pixels close to that color with background pixels
// 4. Interactive tolerance adjustment via trackbar

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <limits>

using std::cout;
using std::cerr;
using std::endl;

// Display image with optional scaling for large images
static void safeImShow(const std::string& winName, const cv::Mat& img, int maxSide = 1400)
{
    cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);
    if (img.empty()) {
        cv::imshow(winName, img);
        return;
    }

    const int h = img.rows, w = img.cols;
    const int longest = std::max(h, w);
    if (maxSide > 0 && longest > maxSide) {
        const double s = double(maxSide) / double(longest);
        cv::Mat scaled;
        cv::resize(img, scaled, cv::Size(), s, s, cv::INTER_AREA);
        cv::imshow(winName, scaled);
    } else {
        cv::imshow(winName, img);
    }
}

// Clamp value between min and max
template <typename T>
static inline T clamp(T v, T lo, T hi) {
    return std::max(lo, std::min(hi, v));
}

// Build 3D color histogram with manual binning
// Returns histogram with shape [buckets, buckets, buckets] for B,G,R channels
static cv::Mat buildHistogram3D(const cv::Mat& imgBGR, int buckets)
{
    int dims[3] = { buckets, buckets, buckets };
    cv::Mat hist(3, dims, CV_32S, cv::Scalar::all(0));
    const int bucketSize = 256 / buckets;

    for (int r = 0; r < imgBGR.rows; ++r) {
        const cv::Vec3b* row = imgBGR.ptr<cv::Vec3b>(r);
        for (int c = 0; c < imgBGR.cols; ++c) {
            const uchar B = row[c][0];
            const uchar G = row[c][1];
            const uchar R = row[c][2];

            int x = B / bucketSize;
            int y = G / bucketSize;
            int z = R / bucketSize;

            // Clamp to valid bucket range
            x = clamp(x, 0, buckets - 1);
            y = clamp(y, 0, buckets - 1);
            z = clamp(z, 0, buckets - 1);

            // Increment 3D histogram bin
            int idx[3] = { x, y, z };
            hist.at<int>(idx) += 1;
        }
    }
    return hist;
}

// Find bin with maximum count in 3D histogram
static void argmax3D(const cv::Mat& hist, cv::Vec3i& maxIdx, int& maxVal)
{
    const int* sizes = hist.size.p;
    const int bx = sizes[0], by = sizes[1], bz = sizes[2];

    maxVal = std::numeric_limits<int>::min();
    maxIdx = cv::Vec3i(0, 0, 0);

    for (int x = 0; x < bx; ++x)
    for (int y = 0; y < by; ++y)
    for (int z = 0; z < bz; ++z) {
        int idx[3] = { x, y, z };
        int v = hist.at<int>(idx);
        if (v > maxVal) {
            maxVal = v;
            maxIdx = cv::Vec3i(x, y, z);
        }
    }
}

// Calculate representative color from bin center
static cv::Vec3i binCenterBGR(const cv::Vec3i& idx, int bucketSize)
{
    const int cBlue  = idx[0] * bucketSize + bucketSize / 2;
    const int cGreen = idx[1] * bucketSize + bucketSize / 2;
    const int cRed   = idx[2] * bucketSize + bucketSize / 2;
    return cv::Vec3i(cBlue, cGreen, cRed);
}

// Perform chroma key replacement
// Pixels within tolerance of target color are replaced with background pixels
static void chromaReplace(const cv::Mat& fg, const cv::Mat& bg,
                          const cv::Vec3i& cBGR, int tol, cv::Mat& out)
{
    out.create(fg.size(), fg.type());

    for (int r = 0; r < fg.rows; ++r) {
        const cv::Vec3b* frow = fg.ptr<cv::Vec3b>(r);
        cv::Vec3b* orow = out.ptr<cv::Vec3b>(r);

        int br = (bg.rows > 0) ? (r % bg.rows) : 0;
        const cv::Vec3b* brow = (bg.rows > 0) ? bg.ptr<cv::Vec3b>(br) : nullptr;

        for (int c = 0; c < fg.cols; ++c) {
            const cv::Vec3b fpx = frow[c];
            const int dB = std::abs(int(fpx[0]) - cBGR[0]);
            const int dG = std::abs(int(fpx[1]) - cBGR[1]);
            const int dR = std::abs(int(fpx[2]) - cBGR[2]);

            const bool isClose = (dB <= tol) && (dG <= tol) && (dR <= tol);

            if (isClose && brow != nullptr) {
                int bc = (bg.cols > 0) ? (c % bg.cols) : 0;
                orow[c] = brow[bc];
            } else {
                orow[c] = fpx;
            }
        }
    }
}

// Context for interactive tolerance trackbar
struct OverlayUIContext {
    cv::Mat fg;
    cv::Mat bg;
    cv::Vec3i cBGR;
    int tolInit;
    int tolMax;
    std::string winName;
    std::string tkName;
    cv::Mat result;
};

// Trackbar callback - recomputes overlay when tolerance changes
static void onToleranceChange(int /*pos*/, void* userdata)
{
    auto* ctx = reinterpret_cast<OverlayUIContext*>(userdata);
    if (!ctx) return;

    int tol = cv::getTrackbarPos(ctx->tkName, ctx->winName);
    tol = clamp(tol, 0, ctx->tolMax);

    chromaReplace(ctx->fg, ctx->bg, ctx->cBGR, tol, ctx->result);
    safeImShow(ctx->winName, ctx->result);

    cv::imwrite("overlay.jpg", ctx->result);
}

int main()
{
    // Load foreground and background images
    const std::string fgPath = "foreground.jpg";
    const std::string bgPath = "background.jpg";

    cv::Mat fg = cv::imread(fgPath, cv::IMREAD_COLOR);
    cv::Mat bg = cv::imread(bgPath, cv::IMREAD_COLOR);

    if (fg.empty() || bg.empty()) {
        cerr << "Error: Could not load 'foreground.jpg' and 'background.jpg'\n";
        return 1;
    }

    // Build 3D histogram of foreground (manual implementation)
    const int buckets = 4;
    const int bucketSize = 256 / buckets;
    cv::Mat hist = buildHistogram3D(fg, buckets);

    // Find most common color bin
    cv::Vec3i maxIdx;
    int maxVal = 0;
    argmax3D(hist, maxIdx, maxVal);
    cv::Vec3i cBGR = binCenterBGR(maxIdx, bucketSize);

    cout << "Most common bin (B,G,R): [" << maxIdx[0] << ", " << maxIdx[1] << ", " << maxIdx[2] << "]\n";
    cout << "Representative color:     [" << cBGR[0]  << ", " << cBGR[1]  << ", " << cBGR[2]  << "]\n";
    cout << "Pixel count: " << maxVal << endl;

    // Setup interactive window with tolerance trackbar
    OverlayUIContext ctx;
    ctx.fg      = fg;
    ctx.bg      = bg;
    ctx.cBGR    = cBGR;
    ctx.tolInit = bucketSize / 2;
    ctx.tolMax  = std::max(bucketSize, 255);
    ctx.winName = "Chroma Key Result";
    ctx.tkName  = "Tolerance";

    cv::namedWindow(ctx.winName, cv::WINDOW_AUTOSIZE);
    cv::createTrackbar(ctx.tkName, ctx.winName, nullptr, ctx.tolMax, onToleranceChange, &ctx);
    cv::setTrackbarPos(ctx.tkName, ctx.winName, ctx.tolInit);

    // Generate initial result
    onToleranceChange(0, &ctx);
    cv::moveWindow(ctx.winName, 60, 60);

    // Wait for user to exit
    for (;;) {
        int key = cv::waitKey(30);
        if (key == 27 || key == 'q' || key == 'Q' || key == ' ')
            break;
    }

    cv::destroyAllWindows();
    if (!ctx.result.empty())
        cv::imwrite("overlay.jpg", ctx.result);

    return 0;
}