//
// Created by hh on 2021/12/20.
//

#include "Wrapper.h"

#include "CreateNodeClient.h"
#include "ManagerClient.h"
#include "ModelClient.h"


int CreatePlaceholder_wrapper(int *idims, int num, int tag, void *manager) {
    return CreatePlaceholder(idims, num, tag, manager);
}

int CreateVariable_wrapper(int *idims, int num, int tag, void *manager) {
    return CreateVariable(idims, num, tag, manager);
}

int CReverseOp_wrapper(int node1, void *manager) {
    return CReverseOp(node1, manager);
}

int CReluOp_wrapper(int node1, void *manager) {
    return CReluOp(node1, manager);
}

int CBNOp_wrapper(int node1, void *manager) {
    return CBNOp(node1, manager);
}

int CBNReluOp_wrapper(int node1, void *manager) {
    return CBNReluOp(node1, manager);
}

int CDropoutOp_wrapper(int node1, float keep_prob, void *manager) {
    return CDropoutOp(node1, keep_prob, manager);
}

int CDropoutIOp_wrapper(int node1, float keep_prob, void *manager) {
    return CDropoutIOp(node1, keep_prob, manager);
}

int CAddOp_wrapper(int node1, int node2, void *manager) {
    return CAddOp(node1, node2, manager);
}

int CSubOp_wrapper(int node1, int node2, void *manager) {
    return CSubOp(node1, node2, manager);
}

int CMatMulOp_wrapper(int node1, int node2, int node3, void *manager) {
    return CMatMulOp(node1, node2, node3, manager);
}

int CConvolutionOp_wrapper(int node1, int node2, int node3, int *strides_dims,
                           int *padding_dims_l, int *padding_dims_r, void *manager) {
    return CConvolutionOp(node1, node2, node3, strides_dims, padding_dims_l, padding_dims_r, manager);
}

int CPoolingOp_wrapper(int node1, int *strides_dims, int *kernel, int *padding_dims_l,
                       int *padding_dims_r, void *manager) {
    return CPoolingOp(node1, strides_dims, kernel, padding_dims_l, padding_dims_r, manager);
}

int CFlatOp_wrapper(int node1, void *manager) {
    return CFlatOp(node1, manager);
}







void *CreateManager_wrapper() {
    return CreateManager();
}

void DelManager_wrapper(void *manager) {
    DelManager(manager);
}

void SoftmaxCrossEntropy_Training_wrapper(int x, int y, float lr, int gd, int num, void *manager) {
    SoftmaxCrossEntropy_Training_Client(x, y, lr, gd, num, manager);
}

void SoftmaxCrossEntropy_Inference_wrapper(int x, int y, int num, void *manager) {
    SoftmaxCrossEntropy_Inference_Client(x, y, num, manager);
}

void SoftmaxCrossEntropy_Accuracy_wrapper(int x, int y, int num, void *manager) {
    SoftmaxCrossEntropy_Accuracy_Client(x, y, num, manager);
}

void SoftmaxCrossEntropy_Training_Accuracy_wrapper(int x, int y, float lr, int gd, int num, void *manager) {
    SoftmaxCrossEntropy_Training_Accuracy_Client(x, y, lr, gd, num, manager);
}

void Predict_wrapper(int x, int num, void *manager) {
    Predict_Client(x, num, manager);
}

void Mse_Training_wrapper(int x, int y, float lr, int gd, int num, void *manager) {
    Mse_Training_Client(x, y, lr, gd, num, manager);
}

void Mse_Training_Inference_wrapper(int x, int y, float lr, int gd, int num, void *manager) {
    Mse_Training_Inference_Client(x, y, lr, gd, num, manager);
}

void Inference_wrapper(int loss, int num, void *manager) {
    Inference_Client(loss, num, manager);
}

void ReadtoNodeScalar_wrapper(int data, int node, int executeId, int dataId, void *manager) {
    ReadtoNodeScalar_Client(data, node, executeId, dataId, manager);
}

void ReadtoNodeReal_wrapper(float data, int node, int executeId, int dataId, void *manager) {
    ReadtoNodeReal_Client(data, node, executeId, dataId, manager);
}

void ReadtoNodeArr_wrapper(float *data, int node, int executeId, int dataId, void *manager) {
    ReadtoNodeArr_Client(data, node, executeId, dataId, manager);
}

void InitNodeZero_wrapper(int node, int executeId, void *manager) {
    InitNodeZero_Client(node, executeId, manager);
}

void InitVariables_wrapper(void *manager) {
    InitVariables_Client(manager);
}

void PrepareMemory_wrapper(void *manager) {
    PrepareMemory_Client(manager);
}

void Compute_wrapper(void *manager) {
    Compute_Client(manager);
}

void ComputeEX_wrapper(void *manager) {
    ComputeEX_Client(manager);
}

void ComputeAccuracy_wrapper(void *manager) {
    ComputeAccuracy_Client(manager);
}

void Wait_wrapper(void *manager) {
    Wait_Client(manager);
}

float GetLoss_wrapper(void *manager) {
    return GetLoss_Client(manager);
}

float GetAccuracy_wrapper(void *manager) {
    return GetAccuracy_Client(manager);
}

int GetPredictN_wrapper(void *manager) {
    return GetPredictN_Client(manager);
}

int GetPredictLabel_wrapper(int dataId, void *manager) {
    return GetPredictLabel_Client(dataId, manager);
}

void UpdateSgdLr_wrapper(float new_lr, void *manager) {
    UpdateSgdLr_Client(new_lr, manager);
}



int GetVariableListSize_wrapper(void *manager) {
    return GetVariableListSize_Client(manager);
}

int GetVariableTag_wrapper(int node, void *manager) {
    return GetVariableTag_Client(node, manager);
}

int GetVariableNdim_wrapper(int node, void *manager) {
    return GetVariableNdim_Client(node, manager);
}

int GetVariableSize_wrapper(int node, void *manager) {
    return GetVariableSize_Client(node, manager);
}

int *GetVariableDims_wrapper(int node, void *manager) {
    return GetVariableDims_Client(node, manager);
}

float *GetVariableData_wrapper(int node, void *manager) {
    return GetVariableData_Client(node, manager);
}

void SetVariablePG_wrapper(int node, void *dst, void *manager) {
    SetVariablePG_Client(node, dst, manager);
}

void SetVariableData_wrapper(int node, int tag, int ndim, int *dims, float *data, void *manager) {
    SetVariableData_Client(node, tag, ndim, dims, data, manager);
}


