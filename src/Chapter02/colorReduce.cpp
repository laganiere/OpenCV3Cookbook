/*------------------------------------------------------------------------------------------*\
   This file contains material supporting chapter 2 of the cookbook:
   Computer Vision Programming using the OpenCV Library
   Second Edition
   by Robert Laganiere, Packt Publishing, 2013.

   This program is free software; permission is hereby granted to use, copy, modify,
   and distribute this source code, or portions thereof, for any purpose, without fee,
   subject to the restriction that the copyright notice may not be removed
   or altered from any source or altered source distribution.
   The software is released on an as-is basis and without any warranties of any kind.
   In particular, the software is not guaranteed to be fault-tolerant or free from failure.
   The author disclaims all warranties with regard to this software, any use,
   and any consequent failure, is purely the responsibility of the user.

   Copyright (C) 2013 Robert Laganiere, www.laganiere.name
\*------------------------------------------------------------------------------------------*/

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// Test 0
// using .ptr and []
void colorReduce0(cv::Mat image, int div=64) {

      int nl= image.rows; // number of lines
      int nc= image.cols * image.channels(); // total number of elements per line

      for (int j=0; j<nl; j++) {

          // get the address of row j
          uchar* data= image.ptr<uchar>(j);

          for (int i=0; i<nc; i++) {

            // process each pixel ---------------------

                data[i]= data[i]/div*div + div/2;

            // end of pixel processing ----------------

          } // end of line
      }
}

// Test 1
// using .ptr and * ++
void colorReduce1(cv::Mat image, int div=64) {

      int nl= image.rows; // number of lines
      int nc= image.cols * image.channels(); // total number of elements per line

      for (int j=0; j<nl; j++) {

          uchar* data= image.ptr<uchar>(j);

          for (int i=0; i<nc; i++) {

            // process each pixel ---------------------

                 *data= *data/div*div + div/2;
                 data++;

            // end of pixel processing ----------------

            } // end of line
      }
}

// Test 2
// using .ptr and * ++ and modulo
void colorReduce2(cv::Mat image, int div=64) {

      int nl= image.rows; // number of lines
      int nc= image.cols * image.channels(); // total number of elements per line

      for (int j=0; j<nl; j++) {

          uchar* data= image.ptr<uchar>(j);

          for (int i=0; i<nc; i++) {

            // process each pixel ---------------------

                 int v= *data;
                 *data++= v - v%div + div/2;
//                 *data++= *data - *data%div + div/2;

            // end of pixel processing ----------------

            } // end of line
      }
}

// Test 3
// using .ptr and * ++ and bitwise
void colorReduce3(cv::Mat image, uchar div=64) {

      int nl= image.rows; // number of lines
      int nc= image.cols * image.channels(); // total number of elements per line
      int n= static_cast<int>(log(static_cast<double>(div))/log(2.0) + 0.5);
      // mask used to round the pixel value
      uchar mask= 0xFF<<n; // e.g. for div=16, mask= 0xF0
      uchar div2= div>>1;

      for (int j=0; j<nl; j++) {

          uchar* data= image.ptr<uchar>(j);

          for (int i=0; i<nc; i++) {

            // process each pixel ---------------------

            *data &= mask;     // masking
            *data++ += div2;   // add div/2

            // end of pixel processing ----------------

            } // end of line
      }
}


// Test 4
// direct pointer arithmetic
void colorReduce4(cv::Mat image, int div=64) {

      int nl= image.rows; // number of lines
      int nc= image.cols * image.channels(); // total number of elements per line
      int n= static_cast<int>(log(static_cast<double>(div))/log(2.0) + 0.5);
      int step= image.step; // effective width
      // mask used to round the pixel value
      uchar mask= 0xFF<<n; // e.g. for div=16, mask= 0xF0

      // get the pointer to the image buffer
      uchar *data= image.data;

      for (int j=0; j<nl; j++) {

          for (int i=0; i<nc; i++) {

            // process each pixel ---------------------

            *(data+i) &= mask;
            *(data+i) += div>>1;

            // end of pixel processing ----------------

            } // end of line

            data+= step;  // next line
      }
}

// Test 5
// using .ptr and * ++ and bitwise with image.cols * image.channels()
void colorReduce5(cv::Mat image, int div=64) {

      int nl= image.rows; // number of lines
      int n= static_cast<int>(log(static_cast<double>(div))/log(2.0) + 0.5);
      // mask used to round the pixel value
      uchar mask= 0xFF<<n; // e.g. for div=16, mask= 0xF0

      for (int j=0; j<nl; j++) {

          uchar* data= image.ptr<uchar>(j);

          for (int i=0; i<image.cols * image.channels(); i++) {

            // process each pixel ---------------------

            *data &= mask;
            *data++ += div>>1;

            // end of pixel processing ----------------

            } // end of line
      }
}

// Test 6
// using .ptr and * ++ and bitwise (continuous)
void colorReduce6(cv::Mat image, int div=64) {

      int nl= image.rows; // number of lines
      int nc= image.cols * image.channels(); // total number of elements per line

      if (image.isContinuous())  {
          // then no padded pixels
          std::cout << "Image is continuous" << std::endl;
          nc= nc*nl;
          nl= 1;  // it is now a 1D array
       }

      int n= static_cast<int>(log(static_cast<double>(div))/log(2.0) + 0.5);
      // mask used to round the pixel value
      uchar mask= 0xFF<<n; // e.g. for div=16, mask= 0xF0

      for (int j=0; j<nl; j++) {

          uchar* data= image.ptr<uchar>(j);

          for (int i=0; i<nc; i++) {

            // process each pixel ---------------------

            *data &= mask;
            *data++ += div>>1;

            // end of pixel processing ----------------

            } // end of line
      }
}

// Test 7
// using .ptr and * ++ and bitwise (continuous+reshape)
void colorReduce7(cv::Mat image, int div=64) {

      if (image.isContinuous()) {
        // no padded pixels
        image.reshape(1,   // new number of channels
                      1) ; // new number of rows
      }
      // number of columns set accordingly

      int nl= image.rows; // number of lines
      int nc= image.cols*image.channels() ; // number of columns

      int n= static_cast<int>(log(static_cast<double>(div))/log(2.0) + 0.5);
      // mask used to round the pixel value
      uchar mask= 0xFF<<n; // e.g. for div=16, mask= 0xF0

      for (int j=0; j<nl; j++) {

          uchar* data= image.ptr<uchar>(j);

          for (int i=0; i<nc; i++) {

            // process each pixel ---------------------

            *data &= mask;
            *data++ += div>>1;

            // end of pixel processing ----------------

            } // end of line
      }
}

// Test 8
// using Mat_ iterator
void colorReduce8(cv::Mat image, int div=64) {

      // get iterators
      cv::Mat_<cv::Vec3b>::iterator it= image.begin<cv::Vec3b>();
      cv::Mat_<cv::Vec3b>::iterator itend= image.end<cv::Vec3b>();

      for ( ; it!= itend; ++it) {

        // process each pixel ---------------------

        (*it)[0]= (*it)[0]/div*div + div/2;
        (*it)[1]= (*it)[1]/div*div + div/2;
        (*it)[2]= (*it)[2]/div*div + div/2;

        // end of pixel processing ----------------
      }
}

// Test 8b
// using Mat_ iterator and Vec3b operator
void colorReduce8b(cv::Mat image, int div=64) {

      // get iterators
      cv::Mat_<cv::Vec3b>::iterator it= image.begin<cv::Vec3b>();
      cv::Mat_<cv::Vec3b>::iterator itend= image.end<cv::Vec3b>();

      const cv::Vec3b offset(div/2,div/2,div/2);

      for ( ; it!= itend; ++it) {

        // process each pixel ---------------------

        *it= *it/div*div+offset;
        // end of pixel processing ----------------
      }
}

// Test 9
// using Mat_ iterator and bitwise
void colorReduce9(cv::Mat image, int div=64) {

      // div must be a power of 2
      int n= static_cast<int>(log(static_cast<double>(div))/log(2.0) + 0.5);
      // mask used to round the pixel value
      uchar mask= 0xFF<<n; // e.g. for div=16, mask= 0xF0

      // get iterators
      cv::Mat_<cv::Vec3b>::iterator it= image.begin<cv::Vec3b>();
      cv::Mat_<cv::Vec3b>::iterator itend= image.end<cv::Vec3b>();

      // scan all pixels
      for ( ; it!= itend; ++it) {

        // process each pixel ---------------------

        (*it)[0]&= mask;
        (*it)[0]+= div>>1;
        (*it)[1]&= mask;
        (*it)[1]+= div>>1;
        (*it)[2]&= mask;
        (*it)[2]+= div>>1;

        // end of pixel processing ----------------
      }
}

// Test 10
// using MatIterator_
void colorReduce10(cv::Mat image, int div=64) {

      // get iterators
      cv::Mat_<cv::Vec3b> cimage= image;
      cv::Mat_<cv::Vec3b>::iterator it=cimage.begin();
      cv::Mat_<cv::Vec3b>::iterator itend=cimage.end();

      for ( ; it!= itend; it++) {

        // process each pixel ---------------------

        (*it)[0]= (*it)[0]/div*div + div/2;
        (*it)[1]= (*it)[1]/div*div + div/2;
        (*it)[2]= (*it)[2]/div*div + div/2;

        // end of pixel processing ----------------
      }
}


// Test 11
// using at method
void colorReduce11(cv::Mat image, int div=64) {

      int nl= image.rows; // number of lines
      int nc= image.cols; // number of columns

      for (int j=0; j<nl; j++) {
          for (int i=0; i<nc; i++) {

            // process each pixel ---------------------

                  image.at<cv::Vec3b>(j,i)[0]=	 image.at<cv::Vec3b>(j,i)[0]/div*div + div/2;
                  image.at<cv::Vec3b>(j,i)[1]=	 image.at<cv::Vec3b>(j,i)[1]/div*div + div/2;
                  image.at<cv::Vec3b>(j,i)[2]=	 image.at<cv::Vec3b>(j,i)[2]/div*div + div/2;

            // end of pixel processing ----------------

            } // end of line
      }
}

// Test 12
// with input/ouput images
void colorReduce12(const cv::Mat &image, // input image
                 cv::Mat &result,      // output image
                 int div=64) {

      int nl= image.rows; // number of lines
      int nc= image.cols ; // number of columns
      int nchannels= image.channels(); // number of channels

      // allocate output image if necessary
      result.create(image.rows,image.cols,image.type());


      for (int j=0; j<nl; j++) {

        // get the addresses of input and output row j
        const uchar* data_in= image.ptr<uchar>(j);
        uchar* data_out= result.ptr<uchar>(j);

        for (int i=0; i<nc*nchannels; i++) {

            // process each pixel ---------------------

                  data_out[i]= data_in[i]/div*div + div/2;

            // end of pixel processing ----------------

        } // end of line
      }
}

// Test 13
// using overloaded operators
void colorReduce13(cv::Mat image, int div=64) {

      int n= static_cast<int>(log(static_cast<double>(div))/log(2.0) + 0.5);
      // mask used to round the pixel value
      uchar mask= 0xFF<<n; // e.g. for div=16, mask= 0xF0

      // perform color reduction
      image=(image&cv::Scalar(mask,mask,mask))+cv::Scalar(div/2,div/2,div/2);
}

// Test 14
// using look up table
void colorReduce14(cv::Mat image, int div=64) {

      cv::Mat lookup(1,256,CV_8U);

      for (int i=0; i<256; i++) {

        lookup.at<uchar>(i)= i/div*div + div/2;
      }

      cv::LUT(image,lookup,image);
}


#define NTESTS 16
#define NITERATIONS 10

int main()
{
    int64 t[NTESTS],tinit;

    cv::Mat image0;
    cv::Mat image1;
    cv::Mat image2;
    cv::Mat image3;
    cv::Mat image4;
    cv::Mat image5;
    cv::Mat image6;
    cv::Mat image7;
    cv::Mat image8;
    cv::Mat image8b;
    cv::Mat image9;
    cv::Mat image10;
    cv::Mat image11;
    cv::Mat image12;
    cv::Mat image13;
    cv::Mat image14;
    cv::Mat result;

    // read the image
    cv::Mat image= cv::imread("boldt.jpg");


    // time and process the image
    const int64 start = cv::getTickCount();
    colorReduce1(image,64);
    //Elapsed time in seconds
    double duration = (cv::getTickCount()-start)/cv::getTickFrequency();

    // display the image
    std::cout << "Duration= " << duration << "secs" <<std::endl;
    cv::namedWindow("Image");
    cv::imshow("Image",image);

    cv::waitKey();

    // timer values set to 0
    for (int i=0; i<NTESTS; i++)
        t[i]= 0;

    // repeat the tests several times
    int n=NITERATIONS;
    for (int k=0; k<n; k++) {

        std::cout << k << " of " << n << std::endl;

        image0= cv::imread("../image.jpg");
        if (!image0.data)
           return 0;

        // using .ptr and []
        tinit= cv::getTickCount();
        colorReduce0(image0);
        t[0]+= cv::getTickCount()-tinit;

        image1= cv::imread("../image.jpg");
        // using .ptr and * ++
        tinit= cv::getTickCount();
        colorReduce1(image1);
        t[1]+= cv::getTickCount()-tinit;

        image2= cv::imread("../image.jpg");
        // using .ptr and * ++ and modulo
        tinit= cv::getTickCount();
        colorReduce2(image2);
        t[2]+= cv::getTickCount()-tinit;

        image3= cv::imread("../image.jpg");
        // using .ptr and * ++ and bitwise
        tinit= cv::getTickCount();
        colorReduce3(image3);
        t[3]+= cv::getTickCount()-tinit;

        image4= cv::imread("../image.jpg");
        // using direct pointer arithmetic
        tinit= cv::getTickCount();
        colorReduce4(image4);
        t[4]+= cv::getTickCount()-tinit;

        image5= cv::imread("../image.jpg");
        // using .ptr and * ++ and bitwise with image.cols * image.channels()
        tinit= cv::getTickCount();
        colorReduce5(image5);
        t[5]+= cv::getTickCount()-tinit;

        image6= cv::imread("../image.jpg");
        // using .ptr and * ++ and bitwise (continuous)
        tinit= cv::getTickCount();
        colorReduce6(image6);
        t[6]+= cv::getTickCount()-tinit;

        image7= cv::imread("../image.jpg");
        // using .ptr and * ++ and bitwise (continuous+channels)
        tinit= cv::getTickCount();
        colorReduce7(image7);
        t[7]+= cv::getTickCount()-tinit;

        image8= cv::imread("../image.jpg");
        // using Mat_ iterator
        tinit= cv::getTickCount();
        colorReduce8(image8);
        t[8]+= cv::getTickCount()-tinit;

        image8b= cv::imread("../image.jpg");
        // using Mat_ iterator
        tinit= cv::getTickCount();
        colorReduce8b(image8b);
        t[15]+= cv::getTickCount()-tinit;

        image9= cv::imread("../image.jpg");
        // using Mat_ iterator and bitwise
        tinit= cv::getTickCount();
        colorReduce9(image9);
        t[9]+= cv::getTickCount()-tinit;

        image10= cv::imread("../image.jpg");
        // using Mat_ iterator
        tinit= cv::getTickCount();
        colorReduce10(image10);
        t[10]+= cv::getTickCount()-tinit;

        image11= cv::imread("../image.jpg");
        // using at
        tinit= cv::getTickCount();
        colorReduce11(image11);
        t[11]+= cv::getTickCount()-tinit;

        image12= cv::imread("../image.jpg");
        // using input/output images
        tinit= cv::getTickCount();
        colorReduce12(image12, result);
        t[12]+= cv::getTickCount()-tinit;


        image13= cv::imread("../image.jpg");
        // using overloaded operators
        tinit= cv::getTickCount();
        colorReduce13(image13);
        t[13]+= cv::getTickCount()-tinit;

        image14= cv::imread("../image.jpg");
        // using overloaded operators
        tinit= cv::getTickCount();
        colorReduce14(image14);
        t[14]+= cv::getTickCount()-tinit;

        //------------------------------
    }

    cv::namedWindow("Test 0");
    cv::imshow("Test 0",image0);
    cv::namedWindow("Test 1");
    cv::imshow("Test 1",image1);
    cv::namedWindow("Test 2");
    cv::imshow("Test 2",image2);
    cv::namedWindow("Test 3");
    cv::imshow("Test 3",image3);
    cv::namedWindow("Test 4");
    cv::imshow("Test 4",image4);
    cv::namedWindow("Test 5");
    cv::imshow("Test 5",image5);
    cv::namedWindow("Test 6");
    cv::imshow("Test 6",image6);
    cv::namedWindow("Test 7");
    cv::imshow("Test 7",image7);
    cv::namedWindow("Test 8");
    cv::imshow("Test 8",image8);
    cv::namedWindow("Test 8b");
    cv::imshow("Test 8b",image8b);
    cv::namedWindow("Test 9");
    cv::imshow("Test 9",image9);
    cv::namedWindow("Test 10");
    cv::imshow("Test 10",image10);
    cv::namedWindow("Test 11");
    cv::imshow("Test 11",image11);
    cv::namedWindow("Result");
    cv::imshow("Result",result);
    cv::namedWindow("Test 13");
    cv::imshow("Test 13",image13);
    cv::namedWindow("Test 14");
    cv::imshow("Test 14",image14);

    // print average execution time
    std::cout << std::endl << "-------------------------------------------" << std::endl << std::endl;
    std::cout << "0. using .ptr and [] =" << 1000.*t[0]/cv::getTickFrequency()/n << "ms" << std::endl;
    std::cout << "1. using .ptr and * ++ =" << 1000.*t[1]/cv::getTickFrequency()/n << "ms" << std::endl;
    std::cout << "2. using .ptr and * ++ and modulo =" << 1000.*t[2]/cv::getTickFrequency()/n << "ms" << std::endl;
    std::cout << "3. using .ptr and * ++ and bitwise =" << 1000.*t[3]/cv::getTickFrequency()/n << "ms" << std::endl;
    std::cout << "4. using direct pointer arithmetic =" << 1000.*t[4]/cv::getTickFrequency()/n << "ms" << std::endl;
    std::cout << "5. using .ptr and * ++ and bitwise with image.cols * image.channels() =" << 1000.*t[5]/cv::getTickFrequency()/n << "ms" << std::endl;
    std::cout << "6. using .ptr and * ++ and bitwise (continuous) =" << 1000.*t[6]/cv::getTickFrequency()/n << "ms" << std::endl;
    std::cout << "7. using .ptr and * ++ and bitwise (continuous+reshape) =" << 1000.*t[7]/cv::getTickFrequency()/n << "ms" << std::endl;
    std::cout << "8. using Mat_ iterator =" << 1000.*t[8]/cv::getTickFrequency()/n << "ms" << std::endl;
    std::cout << "8. using Mat_ iterator and Vec3b op=" << 1000.*t[15]/cv::getTickFrequency()/n << "ms" << std::endl;
    std::cout << "9. using Mat_ iterator and bitwise =" << 1000.*t[9]/cv::getTickFrequency()/n << "ms" << std::endl;
    std::cout << "10. using MatIterator_ =" << 1000.*t[10]/cv::getTickFrequency()/n << "ms" << std::endl;
    std::cout << "11. using at =" << 1000.*t[11]/cv::getTickFrequency()/n << "ms" << std::endl;
    std::cout << "12. using input/output images =" << 1000.*t[12]/cv::getTickFrequency()/n << "ms" << std::endl;
    std::cout << "13. using overloaded operators =" << 1000.*t[13]/cv::getTickFrequency()/n << "ms" << std::endl;
    std::cout << "14. using lookup table =" << 1000.*t[14]/cv::getTickFrequency()/n << "ms" << std::endl;

    cv::waitKey();
    return 0;
}
