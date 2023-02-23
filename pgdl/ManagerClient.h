//
// Created by hh on 2022/2/23.
//

#ifndef PGDL_MANAGERCLIENT_H
#define PGDL_MANAGERCLIENT_H

#include "Autodiff.h"

void *CreateManager();

void DelManager(void *manager);

void SoftmaxCrossEntropy_Training_Client(int x, int y, float lr, int gd, int num, void *manager);

void SoftmaxCrossEntropy_Inference_Client(int x, int y, int num, void *manager);

void SoftmaxCrossEntropy_Accuracy_Client(int x, int y, int num, void *manager);

void SoftmaxCrossEntropy_Training_Accuracy_Client(int x, int y, float lr, int gd, int num, void *manager);

void Predict_Client(int x, int num, void *manager);

void Mse_Training_Client(int x, int y, float lr, int gd, int num, void *manager);

void Mse_Training_Inference_Client(int x, int y, float lr, int gd, int num, void *manager);

void Inference_Client(int loss, int num, void *manager);

void ReadtoNodeScalar_Client(int data, int node, int executeId, int dataId, void *manager);

void ReadtoNodeReal_Client(float data, int node, int executeId, int dataId, void *manager);

void ReadtoNodeArr_Client(float *data, int node, int executeId, int dataId, void *manager);

void InitNodeZero_Client(int node, int executeId, void *manager);

void InitVariables_Client(void *manager);

void PrepareMemory_Client(void *manager);

void Compute_Client(void *manager);

void ComputeEX_Client(void *manager);

void ComputeAccuracy_Client(void *manager);

void Wait_Client(void *manager);

float GetLoss_Client(void *manager);

float GetAccuracy_Client(void *manager);

int GetPredictN_Client(void *manager);

int GetPredictLabel_Client(int dataId, void *manager);

void UpdateSgdLr_Client(float new_lr, void *manager);

#endif //PGDL_MANAGERCLIENT_H
