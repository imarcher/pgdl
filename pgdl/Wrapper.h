//
// Created by hh on 2021/12/20.
//

#ifndef PGDL_WRAPPER_H
#define PGDL_WRAPPER_H




#ifdef __cplusplus
extern "C" {
#endif




//CreateNodeClient start

int CreatePlaceholder_wrapper(int *idims, int num, int tag, void *manager);

int CreateVariable_wrapper(int *idims, int num, int tag, void *manager);

int CReverseOp_wrapper(int node1, void *manager);

int CReluOp_wrapper(int node1, void *manager);

int CBNOp_wrapper(int node1, void *manager);

int CBNReluOp_wrapper(int node1, void *manager);

int CDropoutOp_wrapper(int node1, float keep_prob, void *manager);

int CDropoutIOp_wrapper(int node1, float keep_prob, void *manager);

int CAddOp_wrapper(int node1, int node2, void *manager);

int CSubOp_wrapper(int node1, int node2, void *manager);

int CMatMulOp_wrapper(int node1, int node2, int node3, void *manager);

int CConvolutionOp_wrapper(int node1, int node2, int node3, int *strides_dims,
                   int *padding_dims_l, int *padding_dims_r, void *manager);

int CPoolingOp_wrapper(int node1, int *strides_dims, int *kernel, int *padding_dims_l,
               int *padding_dims_r, void *manager);

int CFlatOp_wrapper(int node1, void *manager);



//CreateNodeClient end

//ManagerClient start

void *CreateManager_wrapper();

void DelManager_wrapper(void *manager);

void SoftmaxCrossEntropy_Training_wrapper(int x, int y, float lr, int gd, int num, void *manager);

void SoftmaxCrossEntropy_Inference_wrapper(int x, int y, int num, void *manager);

void SoftmaxCrossEntropy_Accuracy_wrapper(int x, int y, int num, void *manager);

void SoftmaxCrossEntropy_Training_Accuracy_wrapper(int x, int y, float lr, int gd, int num, void *manager);

void Predict_wrapper(int x, int num, void *manager);

void Mse_Training_wrapper(int x, int y, float lr, int gd, int num, void *manager);

void Mse_Training_Inference_wrapper(int x, int y, float lr, int gd, int num, void *manager);

void Inference_wrapper(int loss, int num, void *manager);

void ReadtoNodeScalar_wrapper(int data, int node, int executeId, int dataId, void *manager);

void ReadtoNodeReal_wrapper(float data, int node, int executeId, int dataId, void *manager);

void ReadtoNodeArr_wrapper(float *data, int node, int executeId, int dataId, void *manager);

void InitNodeZero_wrapper(int node, int executeId, void *manager);

void InitVariables_wrapper(void *manager);

void PrepareMemory_wrapper(void *manager);

void Compute_wrapper(void *manager);

void ComputeEX_wrapper(void *manager);

void ComputeAccuracy_wrapper(void *manager);

void Wait_wrapper(void *manager);

float GetLoss_wrapper(void *manager);

float GetAccuracy_wrapper(void *manager);

int GetPredictN_wrapper(void *manager);

int GetPredictLabel_wrapper(int dataId, void *manager);

void UpdateSgdLr_wrapper(float new_lr, void *manager);

//ManagerClient end

//ModelClient start

int GetVariableListSize_wrapper(void *manager);

int GetVariableTag_wrapper(int node, void *manager);

int GetVariableNdim_wrapper(int node, void *manager);

int GetVariableSize_wrapper(int node, void *manager);

int *GetVariableDims_wrapper(int node, void *manager);

float *GetVariableData_wrapper(int node, void *manager);

void SetVariablePG_wrapper(int node, void *dst, void *manager);

void SetVariableData_wrapper(int node, int tag, int ndim, int *dims, float *data, void *manager);

//ModelClient end
#ifdef __cplusplus
}
#endif



#endif //PGDL_WRAPPER_H
