opencv_createsamples -info stop.txt -vec stop.vec -w 24 -h 24 -num 10
opencv_traincascade -data classifier -vec stop.vec -bg neg.txt  -numPos 8 -numNeg 50 -w 24 -h 24 
