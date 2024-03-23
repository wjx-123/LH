#include <opencv2/opencv.hpp>

#define SS 0
#define SE 1
#define ES 2
#define EE 3

#define EDGE_VERTICAL   1
#define EDGE_HORIZONTAL 2

#define ANCHOR_PIXEL  254
#define EDGE_PIXEL    255

#define LEFT  1
#define RIGHT 2
#define UP    3
#define DOWN  4

enum GradientOperator { PREWITT_OPERATOR = 101, SOBEL_OPERATOR = 102, SCHARR_OPERATOR = 103 };

struct StackNode {
    int r, c;   // �ߧѧ�ѧݧ�ߧ�� ��ڧܧ�֧�
    int parent; // ���էڧ�֧ݧ� (-1 �֧�ݧ� �ߧ֧��)
    int dir;    // �ߧѧ��ѧӧݧ֧ߧڧ�
};

// �էݧ� ���֧էڧߧ֧ߧڧ� �ԧ�ѧߧ֧�
struct Chain {

    int dir;                   // �ߧѧ��ѧӧݧ֧ߧڧ� ��֧���ܧ�
    int len;                   // �ܧ��-�ӧ� ��ڧܧ�֧ݧ֧� �� ��֧��
    int parent;                // ���էڧ�֧ݧ� (-1 �֧�ݧ� �ߧ֧��)
    int children[2];           // �է֧�� (-1 �֧�ݧ� �ߧ֧��)
    cv::Point* pixels;         // ��ܧѧ٧ѧ�֧ݧ� �ߧ� �ߧѧ�ѧݧ� �ާѧ��ڧӧ� ��ڧܧ�֧ݧ֧�
};




struct LS {
    cv::Point2d start;
    cv::Point2d end;

    LS(cv::Point2d _start, cv::Point2d _end)
    {
        start = _start;
        end = _end;
    }
};


struct LineSegment {
    double a, b;
    int invert;

    double sx, sy;        // �ߧѧ�ѧݧ�
    double ex, ey;        // �ܧ�ߧ֧�

    int segmentNo;        // ��֧ԧާ֧ߧ� �ܧ�����ާ� ���ڧߧѧէݧ֧اڧ� ����֧٧��
    int firstPixelIndex;  // �ڧߧէ֧ܧ� ��֧�ӧ�ԧ� ��ڧܧ�֧ݧ� �� ��֧ԧާ֧ߧ��
    int len;              // �էݧڧߧ� �� ��ڧܧ�֧ݧ��

    LineSegment(double _a, double _b, int _invert, double _sx, double _sy, double _ex, double _ey, int _segmentNo, int _firstPixelIndex, int _len) {
        a = _a;
        b = _b;
        invert = _invert;
        sx = _sx;
        sy = _sy;
        ex = _ex;
        ey = _ey;
        segmentNo = _segmentNo;
        firstPixelIndex = _firstPixelIndex;
        len = _len;
    }
};

class EDLines {
public:
    EDLines();
    ~EDLines();
public:
    EDLines(cv::Mat _srcImage, GradientOperator _op = PREWITT_OPERATOR, int _gradThresh = 20, int _anchorThresh = 0, int _scanInterval = 1, int _minPathLen = 10, double _sigma = 1.5, bool _sumFlag = true, double _line_error = 1.0, int _min_line_len = -1, double _max_distance_between_two_lines = 6.0, double _max_error = 1.3);

    cv::Mat getEdgeImage();
    cv::Mat getAnchorImage();
    cv::Mat getSmoothImage();
    cv::Mat getGradImage();
    cv::Mat getLineImage();
    cv::Mat drawOnImage(std::vector<LS>& lines);
    cv::Mat drawOwn(std::vector<LS>& lines);

    int getSegmentNo();
    int getAnchorNo();

    std::vector<cv::Point> getAnchorPoints();
    std::vector<std::vector<cv::Point>> getSegments();
    std::vector<std::vector<cv::Point>> getSortedSegments();

    cv::Mat drawParticularSegments(std::vector<int> list);

    std::vector<LS> getLines();
    int getLinesNo();

    std::vector<cv::Rect> getROIs(cv::Mat& image, int x, int y);//����x*y�ָ�ͼƬ

protected:
    int width; // ��ڧ�ڧߧ� �ڧ���էߧ�ԧ� �ڧ٧�ҧ�ѧا֧ߧڧ�
    int height; // �ӧ����� �ڧ���էߧ�ԧ� �ڧ٧�ҧ�ѧا֧ߧڧ�
    uchar* srcImg;
    std::vector<std::vector< cv::Point> > segmentPoints;
    double sigma; // ��ڧԧާ� ���ѧ����
    cv::Mat smoothImage;
    uchar* edgeImg;
    uchar* smoothImg;
    int segmentNos;
    int minPathLen;
    cv::Mat srcImage;

private:
    void ComputeGradient();
    void ComputeAnchorPoints();
    void JoinAnchorPointsUsingSortedAnchors();
    int* sortAnchorsByGradValue1();

    static int LongestChain(Chain* chains, int root);
    static int RetrieveChainNos(Chain* chains, int root, int chainNos[]);

    int anchorNos;
    std::vector<cv::Point> anchorPoints;
    std::vector<cv::Point> edgePoints;

    cv::Mat edgeImage;
    cv::Mat gradImage;
    cv::Mat threshImage;

    uchar* dirImg; // ��ܧѧ٧ѧ�֧ݧ� �ߧ� �ߧѧ��ѧӧݧ֧ߧڧ� �ԧ�ѧէڧ֧ߧ�� ��ڧܧ�֧ݧ�
    short* gradImg; // ��ܧѧ٧ѧ�֧ݧ� �ߧ� �ԧ�ѧէڧ֧� ��ڧܧ�֧ݧ�

    GradientOperator gradOperator; // ���֧�ѧ��� �ԧ�ѧէڧ֧ߧ��
    int gradThresh;
    int anchorThresh;
    int scanInterval;
    bool sumFlag;

    std::vector<LineSegment> lines;
    std::vector<LineSegment> invalidLines;
    std::vector<LS> linePoints;
    int linesNo;
    int min_line_len;
    double line_error;
    double max_distance_between_two_lines;
    double max_error;
    double prec;


    int ComputeMinLineLength();
    void SplitSegment2Lines(double* x, double* y, int noPixels, int segmentNo);
    void JoinCollinearLines();

    bool TryToJoinTwoLineSegments(LineSegment* ls1, LineSegment* ls2, int changeIndex);

    static double ComputeMinDistance(double x1, double y1, double a, double b, int invert);
    static void ComputeClosestPoint(double x1, double y1, double a, double b, int invert, double& xOut, double& yOut);
    static void LineFit(double* x, double* y, int count, double& a, double& b, int invert);
    static void LineFit(double* x, double* y, int count, double& a, double& b, double& e, int& invert);
    static double ComputeMinDistanceBetweenTwoLines(LineSegment* ls1, LineSegment* ls2, int* pwhich);
    static void UpdateLineParameters(LineSegment* ls);
    static void EnumerateRectPoints(double sx, double sy, double ex, double ey, int ptsx[], int ptsy[], int* pNoPoints);

};


