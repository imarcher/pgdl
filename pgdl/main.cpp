//
// Created by hh on 2022/2/26.
//

#include "ManagerClient.h"
#include "CreateNodeClient.h"
#include <cmath>
#include <vector>
#include <iostream>


using namespace dnnl;


int main(){

    void *manager = CreateManager();


    int a1[] = {32,1,28,28};
    int a2[] = {32,1,5,5};
    int a3[] = {32};
    int a4[] = {1,1};
    int a5[] = {2,2};
    int a6[] = {2,2};
    //pool
    int a7[] = {2,2};
    int a8[] = {2,2};
    int a9[] = {0,0};
    int a10[] = {0,0};
    int x = CreatePlaceholder(a1,4,3,manager);
    int w1 = CreateVariable(a2,4,4,manager);
    int b1 = CreateVariable(a3,1,1,manager);
    int conv1 = CConvolutionOp(x, w1, b1, a4, a5, a6, manager);
    int conv1_relu = CBNReluOp(conv1, manager);
    int conv1_out = CPoolingOp(conv1_relu, a7, a8, a9, a10, manager);

    int a11[] = {64,32,5,5};
    int a12[] = {64};
    int a13[] = {1,1};
    int a14[] = {2,2};
    int a15[] = {2,2};
    //pool
    int a16[] = {2,2};
    int a17[] = {2,2};
    int a18[] = {0,0};
    int a19[] = {0,0};
    int w2 = CreateVariable(a11,4,4,manager);
    int b2 = CreateVariable(a12,1,1,manager);
    int conv2 = CConvolutionOp(conv1_out, w2, b2, a13, a14, a15, manager);
    int conv2_relu = CBNReluOp(conv2, manager);
    int conv2_out = CPoolingOp(conv2_relu, a16, a17, a18, a19, manager);

    int conv_flat = CFlatOp(conv2_out, manager);


    int a20[] = {7*7*64, 1024};
    int a21[] = {1,1024};
    int w3 = CreateVariable(a20,2,0,manager);
    int b3 = CreateVariable(a21,2,0,manager);
    int dense1 = CMatMulOp(conv_flat,w3,b3,manager);
    int dense1_relu = CBNReluOp(dense1, manager);

    int dense1_dropout = CDropoutOp(dense1_relu, 0.5, manager);

    int a22[] = {1024, 10};
    int a23[] = {1,10};
    int w4 = CreateVariable(a22,2,0,manager);
    int b4 = CreateVariable(a23,2,0,manager);
    int dense2 = CMatMulOp(dense1_dropout,w4,b4,manager);
    int dense2_relu = CBNReluOp(dense2, manager);

    int a24[] = {32, 10};
    int y = CreatePlaceholder(a24,2,0,manager);
    SoftmaxCrossEntropy_Training_Accuracy_Client(dense2_relu, y, 0.000000001, 1, 1, manager);



    PrepareMemory_Client(manager);
    InitVariables_Client(manager);

    const int image_size = 32 * 784;

    // Allocate a buffer for the image
    std::vector<float> image(image_size);
    int mark =0;
    for (int i = 0; i < image_size; i = i+2){

        if(mark==0){
            image[i] = i;
            image[i+1] = i;
            mark =1;
        }else{
            image[i] = -i;
            image[i+1] = -i;
            mark = 0;
        }
    }


    std::vector<int> yy(32);

    for (int i = 0; i < 32; ++i){
        if(i%2==0){
            yy[i] = 0;
        }else{
            yy[i] = 1;
        }
    }
    for (int i = 0; i < 32; ++i){
        ReadtoNodeArr_Client(image.data()+784*i, x, 0, i, manager);
    }
    InitNodeZero_Client(y, 0, manager);

    for (int i = 0; i < 32; ++i){
        ReadtoNodeScalar_Client(yy.at(i), y, 0, i, manager);
    }
    cout<<1<<endl;
    clock_t start = clock();      //获取当前系统时间

    for (int i=0;i<100;i++){
        ComputeEX_Client(manager);
        cout<< 1<<endl;
    }
    clock_t end  = clock();
    double programTimes = ((double) end -start) / CLOCKS_PER_SEC;
    cout<<programTimes<<endl;



//    for(int i =0;i<20;i++){
//        Compute_Client(manager);
//    }


//    for(int i=0;i<20;i++){
//        cout<<i<<"start:"<<endl;
//        ComputeEX_Client(manager);
//        Wait_Client(manager);
//        cout<<"loss"<<GetLoss_Client(manager)<<endl;
//        cout<<"accuracy:"<<GetAccuracy_Client(manager)<<endl;
//        cout<<i<<"end:"<<endl;
//    }
//    DelManager(manager);
    return 0;
}