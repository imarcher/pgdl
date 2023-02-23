/*-------------------------------------------------------------------------
 *
 * 创建node及其op的函数
 *
 *-------------------------------------------------------------------------
 */

#ifndef PGDL_CREATENODE_H
#define PGDL_CREATENODE_H


#include "Autodiff.h"


/*
 * 占为符
 * 数据输入点
 */
Node *CreatePlaceholder(vector<Node *> &newNodeList, vector<Node *> &placeholderList);
Node *CreatePlaceholder(memory::dims &adims, memory::data_type adata_type, memory::format_tag aformat_tag,
        vector<Node *> &newNodeList, vector<Node *> &placeholderList);

/*
 * 变量
 * 模型参数
 */
Node *CreateVariable(vector<Node *> &newNodeList, vector<Node *> &variableList);
Node *CreateVariable(memory::dims &adims, memory::data_type adata_type, memory::format_tag aformat_tag,
        vector<Node *> &newNodeList, vector<Node *> &variableList);

Node *CLinearOp(Node *node1, float a, float b,
                 vector<Node *> &newNodeList);

Node *CReverseOp(Node *node1,
        vector<Node *> &newNodeList);

Node *CReverseBackwardOp(Node *node1,
        vector<Node *> &newNodeList);

Node *CReluOp(Node *node1,
                 vector<Node *> &newNodeList);

Node *CReluBackwardOp(Node *node1, Node *node2,
                         vector<Node *> &newNodeList);

Node *CBNOp(Node *node1,
                vector<Node *> &newNodeList);

Node *CBNBackwardOp(Node *node1, Node *node2,
                        vector<Node *> &newNodeList);

Node *CBNReluOp(Node *node1,
              vector<Node *> &newNodeList);

Node *CBNReluBackwardOp(Node *node1, Node *node2,
                      vector<Node *> &newNodeList);

Node *CDropoutOp(Node *node1, float keep_prob,
                vector<Node *> &newNodeList, vector<Node *> &dropoutNodeList, unordered_map<Node *, float> &dropoutKeepProb_map);

Node *CDropoutBackwardOp(Node *node1, Node *node2,
             vector<Node *> &newNodeList);

Node *CDropoutOp_Inference(Node *node1, float keep_prob,
                         vector<Node *> &newNodeList);

Node *CAddOp(Node *node1, Node *node2,
        vector<Node *> &newNodeList);

Node *CSubOp(Node *node1, Node *node2,
        vector<Node *> &newNodeList);

Node *CMatMulOp(Node *node1, Node *node2, Node *node3,
        vector<Node *> &newNodeList);

Node *CMatMulBackwardDataOp(Node *node1, Node *node2,
        vector<Node *> &newNodeList);

Node *CMatMulBackwardWeightsOp(Node *node1, Node *node2,
        vector<Node *> &newNodeList);

Node *CMatMulBackwardBiasOp(Node *node1,
        vector<Node *> &newNodeList);

Node *CReOrderOp(Node *node1, memory::desc *desc,
                            vector<Node *> &newNodeList);

Node *CConvolutionOp(Node *node1, Node *node2, Node *node3, memory::dims &strides_dims,
                     memory::dims &padding_dims_l, memory::dims &padding_dims_r,
                     vector<Node *> &newNodeList);

Node *CConvolutionBackwardDataOp(Node *node1, Node *node2, memory::dims &strides_dims,
                     memory::dims &padding_dims_l, memory::dims &padding_dims_r,
                     vector<Node *> &newNodeList);

Node *CConvolutionBackwardWeightsOp(Node *node1, Node *node2, Node *node3, memory::dims &strides_dims,
                     memory::dims &padding_dims_l, memory::dims &padding_dims_r,
                     vector<Node *> &newNodeList);

Node *CConvolutionBackwardBiasOp(memory::desc *desc,
                 vector<Node *> &newNodeList);

Node *CPoolingOp(Node *node1, memory::dims &strides_dims, memory::dims &kernel, memory::dims &padding_dims_l,
        memory::dims &padding_dims_r,
              vector<Node *> &newNodeList);

Node *CPoolingBackwardOp(Node *node1, memory::dims &strides_dims, memory::dims &kernel, memory::dims &padding_dims_l,
        memory::dims &padding_dims_r,
                      vector<Node *> &newNodeList);

Node *CFlatOp(Node *node1,
                 vector<Node *> &newNodeList);

Node *CFlatBackwardOp(Node *node1,
                 vector<Node *> &newNodeList);

/*
 * 专门用于softmax交叉熵训练
 * 不算loss
 * x y
 */
Node *SoftmaxCrossEntropy_Training(Node *node1, Node *node2,
        vector<Node *> &newNodeList);

/*
 * node1导数
 */
Node *CSgdOp(Node *node1, Node *node2, vector<Node *> &newNodeList, vector<Node *> &sgdLrNodes);

/*
 * node1导数
 */
Node *CAdamOp(Node *node1, Node *node2, float lr, vector<Node *> &newNodeList, vector<Node *> &adamMVNodes);

/*
 * 专门用于softmax交叉熵推理
 * 算loss
 * x y
 */
Node *SoftmaxCrossEntropy_Inference(Node *node1, Node *node2,
        vector<Node *> &newNodeList);
/*
 * 专门用于softmax交叉熵推理
 * 算loss和准确率
 * x y
 */
void SoftmaxCrossEntropy_Accuracy(Node *node1, Node *node2, vector<Node *> &resNodes,
        vector<Node *> &newNodeList);

/*
 * 用于预测
 * 算出one-hot编码
 * x
 */
Node *CPredictOp(Node *node1, vector<Node *> &newNodeList);

Node *Mse_Training_I(Node *node1, Node *node2,
                   vector<Node *> &newNodeList);

Node *Mse_Training(Node *node1, Node *node2,
                     vector<Node *> &newNodeList);
/*
 * x-y
 */
Node *Mse_Inference(Node *node1,
                   vector<Node *> &newNodeList);



#endif //PGDL_CREATENODE_H
