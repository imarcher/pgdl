//
// Created by hh on 2022/1/11.
//

#include "CreateNode.h"







Node *CreatePlaceholder(vector<Node *> &newNodeList, vector<Node *> &placeholderList) {
    Node *new_node = new Node(Placeholder);
    newNodeList.push_back(new_node);
    placeholderList.push_back(new_node);
    return new_node;
}

Node *CreatePlaceholder(memory::dims &adims, memory::data_type adata_type, memory::format_tag aformat_tag,
        vector<Node *> &newNodeList, vector<Node *> &placeholderList) {
    Node *new_node = new Node(Placeholder);
    newNodeList.push_back(new_node);
    placeholderList.push_back(new_node);
    new_node->desc = new memory::desc(adims, adata_type, aformat_tag);
    return new_node;
}

Node *CreateVariable(vector<Node *> &newNodeList, vector<Node *> &variableList) {
    Node *new_node = new Node(Variable);
    newNodeList.push_back(new_node);
    variableList.push_back(new_node);
    return new_node;
}

Node *CreateVariable(memory::dims &adims, memory::data_type adata_type, memory::format_tag aformat_tag,
        vector<Node *> &newNodeList, vector<Node *> &variableList) {
    Node *new_node = new Node(Variable);
    newNodeList.push_back(new_node);
    variableList.push_back(new_node);
    new_node->desc = new memory::desc(adims, adata_type, aformat_tag);
    return new_node;
}

Node *CLinearOp(Node *node1, float a, float b,
                vector<Node *> &newNodeList) {
    Node *new_node = new Node(Linear);
    newNodeList.push_back(new_node);
    Op *new_op = new LinearOp(node1, a, b, new_node);
    new_node->op = new_op;
    return new_node;
}

Node *CReverseOp(Node *node1, vector<Node *> &newNodeList) {
    Node *new_node = new Node(Reverse);
    newNodeList.push_back(new_node);
    Op *new_op = new ReverseOp(node1, new_node);
    new_node->op = new_op;
    return new_node;
}

Node *CReverseBackwardOp(Node *node1, vector<Node *> &newNodeList) {
    Node *new_node = new Node(ReverseBackward);
    newNodeList.push_back(new_node);
    Op *new_op = new ReverseBackwardOp(node1, new_node);
    new_node->op = new_op;
    return new_node;
}

Node *CReluOp(Node *node1,
              vector<Node *> &newNodeList) {
    Node *new_node = new Node(Relu);
    newNodeList.push_back(new_node);
    Op *new_op = new ReluOp(node1, new_node);
    new_node->op = new_op;
    return new_node;
}

Node *CReluBackwardOp(Node *node1, Node *node2,
                      vector<Node *> &newNodeList) {
    Node *new_node = new Node(ReluBackward);
    newNodeList.push_back(new_node);
    Op *new_op = new ReluBackwardOp(node1, node2, new_node);
    new_node->op = new_op;
    return new_node;
}

Node *CBNOp(Node *node1,
                vector<Node *> &newNodeList) {
    Node *new_node = new Node(BN);
    newNodeList.push_back(new_node);
    Op *new_op = new BNOp(node1, new_node);
    new_node->op = new_op;
    return new_node;
}

Node *CBNBackwardOp(Node *node1, Node *node2,
                        vector<Node *> &newNodeList) {
    Node *new_node = new Node(BNBackward);
    newNodeList.push_back(new_node);
    Op *new_op = new BNBackwardOp(node1, node2, new_node);
    new_node->op = new_op;
    return new_node;
}

Node *CBNReluOp(Node *node1,
                vector<Node *> &newNodeList) {
    Node *new_node = new Node(BNRelu);
    newNodeList.push_back(new_node);
    Op *new_op = new BNReluOp(node1, new_node);
    new_node->op = new_op;
    return new_node;
}

Node *CBNReluBackwardOp(Node *node1, Node *node2,
                        vector<Node *> &newNodeList) {
    Node *new_node = new Node(BNReluBackward);
    newNodeList.push_back(new_node);
    Op *new_op = new BNReluBackwardOp(node1, node2, new_node);
    new_node->op = new_op;
    return new_node;
}

Node *CDropoutOp(Node *node1, float keep_prob,
                 vector<Node *> &newNodeList, vector<Node *> &dropoutNodeList, unordered_map<Node *, float> &dropoutKeepProb_map) {
    Node *new_node1 = new Node(Dropout1);
    newNodeList.push_back(new_node1);
    dropoutNodeList.push_back(new_node1);
    new_node1->inputs.push_back(node1);

    Node *new_node2 = new Node(Dropout_Prob);
    newNodeList.push_back(new_node2);
    dropoutKeepProb_map.insert({new_node2, keep_prob});
    new_node2->inputs.push_back(node1);

    Node *new_node3 = new Node(Dropout2);
    newNodeList.push_back(new_node3);
    Op *new_op3 = new Dropout2Op(new_node1, new_node2, keep_prob, new_node3);
    new_node3->op = new_op3;

    Node *new_node4 = new Node(Dropout3);
    newNodeList.push_back(new_node4);
    Op *new_op4 = new Dropout3Op(node1, new_node3, new_node4);
    new_node4->op = new_op4;
    return new_node4;
}

Node *CDropoutBackwardOp(Node *node1, Node *node2,
                         vector<Node *> &newNodeList) {
    Node *new_node = new Node(DropoutBackward);
    newNodeList.push_back(new_node);
    Op *new_op = new Dropout3Op(node1, node2, new_node);
    new_node->op = new_op;
    return new_node;
}

Node *CDropoutOp_Inference(Node *node1, float keep_prob,
                           vector<Node *> &newNodeList) {
    Node *new_node = new Node(Dropout_Inference);
    newNodeList.push_back(new_node);
    Op *new_op = new LinearOp(node1, keep_prob, 0, new_node);
    new_node->op = new_op;
    return new_node;
}

Node *CAddOp(Node *node1, Node *node2, vector<Node *> &newNodeList) {
    Node *new_node = new Node(Add);
    newNodeList.push_back(new_node);
    Op *new_op = new AddOp(node1, node2, new_node);
    new_node->op = new_op;
    return new_node;
}

Node *CSubOp(Node *node1, Node *node2, vector<Node *> &newNodeList) {
    Node *new_node = new Node(Sub);
    newNodeList.push_back(new_node);
    Op *new_op = new SubOp(node1, node2, new_node);
    new_node->op = new_op;
    return new_node;
}

Node *CMatMulOp(Node *node1, Node *node2, Node *node3, vector<Node *> &newNodeList) {
    Node *new_node = new Node(MatMul);
    newNodeList.push_back(new_node);
    Op *new_op = new MatMulOp(node1, node2, node3, new_node);
    new_node->op = new_op;
    return new_node;
}

Node *CMatMulBackwardDataOp(Node *node1, Node *node2, vector<Node *> &newNodeList) {
    Node *new_node = new Node(MatMulBackwardData);
    newNodeList.push_back(new_node);
    Op *new_op = new MatMulBackwardDataOp(node1, node2, new_node);
    new_node->op = new_op;
    return new_node;
}

Node *CMatMulBackwardWeightsOp(Node *node1, Node *node2, vector<Node *> &newNodeList) {
    Node *new_node = new Node(MatMulBackwardWeights);
    newNodeList.push_back(new_node);
    Op *new_op = new MatMulBackwardWeightsOp(node1, node2, new_node);
    new_node->op = new_op;
    return new_node;
}

Node *CMatMulBackwardBiasOp(Node *node1, vector<Node *> &newNodeList) {
    Node *new_node = new Node(MatMulBackwardBias);
    newNodeList.push_back(new_node);
    Op *new_op = new MatMulBackwardBiasOp(node1, new_node);
    new_node->op = new_op;
    return new_node;
}

Node *CReOrderOp(Node *node1, memory::desc *desc, vector<Node *> &newNodeList) {
    Node *new_node = new Node(ReOrder);
    newNodeList.push_back(new_node);
    new_node->desc = new memory::desc(*desc);
    Op *new_op = new ReOrderOp(node1, new_node);
    new_node->op = new_op;
    return new_node;
}

Node *CConvolutionOp(Node *node1, Node *node2, Node *node3, memory::dims &strides_dims, memory::dims &padding_dims_l,
                     memory::dims &padding_dims_r, vector<Node *> &newNodeList) {
    Node *new_node = new Node(Convolution);
    newNodeList.push_back(new_node);
    Op *new_op = new ConvolutionOp(node1, node2, node3, new_node, strides_dims, padding_dims_l, padding_dims_r);
    new_node->op = new_op;
    return new_node;
}

Node *CConvolutionBackwardDataOp(Node *node1, Node *node2, memory::dims &strides_dims,
                                 memory::dims &padding_dims_l, memory::dims &padding_dims_r,
                                 vector<Node *> &newNodeList) {
    Node *new_node = new Node(ConvolutionBackwardData);
    newNodeList.push_back(new_node);
    Op *new_op = new ConvolutionBackwardDataOp(node1, node2, new_node, strides_dims, padding_dims_l, padding_dims_r);
    new_node->op = new_op;
    return new_node;
}

Node *CConvolutionBackwardWeightsOp(Node *node1, Node *node2, Node *node3, memory::dims &strides_dims,
                                    memory::dims &padding_dims_l, memory::dims &padding_dims_r,
                                    vector<Node *> &newNodeList) {
    Node *new_node = new Node(ConvolutionBackwardWeights);
    newNodeList.push_back(new_node);
    Op *new_op = new ConvolutionBackwardWeightsOp(node1, node2, node3, new_node, strides_dims, padding_dims_l, padding_dims_r);
    new_node->op = new_op;
    return new_node;
}

Node *CConvolutionBackwardBiasOp(memory::desc *desc,
                                 vector<Node *> &newNodeList) {
    Node *new_node = new Node(ConvolutionBackwardBias);
    newNodeList.push_back(new_node);
    new_node->desc = new memory::desc(*desc);
    return new_node;
}

Node *CPoolingOp(Node *node1, memory::dims &strides_dims, memory::dims &kernel, memory::dims &padding_dims_l,
                 memory::dims &padding_dims_r,
                 vector<Node *> &newNodeList) {
    Node *new_node = new Node(Pooling);
    newNodeList.push_back(new_node);
    Op *new_op = new PoolingOp(node1, new_node, strides_dims, kernel, padding_dims_l, padding_dims_r);
    new_node->op = new_op;
    return new_node;
}

Node *CPoolingBackwardOp(Node *node1, memory::dims &strides_dims, memory::dims &kernel, memory::dims &padding_dims_l,
                         memory::dims &padding_dims_r,
                         vector<Node *> &newNodeList) {
    Node *new_node = new Node(PoolingBackward);
    newNodeList.push_back(new_node);
    Op *new_op = new PoolingBackwardOp(node1, new_node, strides_dims, kernel, padding_dims_l, padding_dims_r);
    new_node->op = new_op;
    return new_node;
}

Node *CFlatOp(Node *node1,
              vector<Node *> &newNodeList) {
    Node *new_node = new Node(Flat);
    newNodeList.push_back(new_node);
    Op *new_op = new FlatOp(node1, new_node);
    new_node->op = new_op;
    return new_node;
}

Node *CFlatBackwardOp(Node *node1,
                      vector<Node *> &newNodeList) {
    Node *new_node = new Node(FlatBackward);
    newNodeList.push_back(new_node);
    Op *new_op = new FlatBackwardOp(node1, new_node);
    new_node->op = new_op;
    return new_node;
}


Node *SoftmaxCrossEntropy_Training(Node *node1, Node *node2, vector<Node *> &newNodeList) {
    Node *new_node1 = new Node(SoftmaxCrossEntropy_Training1);
    newNodeList.push_back(new_node1);
    Op *new_op1 = new SoftmaxCrossEntropy_Training1Op(node1, new_node1);
    new_node1->op = new_op1;

    Node *new_node2 = new Node(SoftmaxCrossEntropy_Training2);
    newNodeList.push_back(new_node2);
    Op *new_op2 = new SoftmaxCrossEntropy_Training2Op(new_node1, node2, new_node2);
    new_node2->op = new_op2;
    return new_node2;
}

Node *CSgdOp(Node *node1, Node *node2, vector<Node *> &newNodeList, vector<Node *> &sgdLrNodes) {
    Node *new_node1 = new Node(SgdLr);
    sgdLrNodes.push_back(new_node1);
    new_node1->inputs.push_back(node1);

    Node *new_node2 = new Node(Sgd);
    newNodeList.push_back(new_node2);
    Op *new_op2 = new SgdOp(node1, new_node1, new_node2);
    new_node2->op = new_op2;

    Node *new_node3 = new Node(Sub);
    newNodeList.push_back(new_node3);
    Op *new_op3 = new SubOp(node2, new_node2, new_node3);
    new_node3->op = new_op3;
    return new_node3;
}

Node *CAdamOp(Node *node1, Node *node2, float lr, vector<Node *> &newNodeList, vector<Node *> &adamMVNodes) {

    float b1 = 0.9;
    float b2 = 0.999;
    float e = 1e-8;

    Node *new_node1 = new Node(Adam_M);
    adamMVNodes.push_back(new_node1);
    new_node1->inputs.push_back(node1);

    Node *new_node2 = new Node(Adam_V);
    adamMVNodes.push_back(new_node2);
    new_node2->inputs.push_back(node1);

    Node *new_node3 = new Node(Adam_Gt);
    newNodeList.push_back(new_node3);
    Op *new_op3 = new LinearOp(node1, 1-b1, 0, new_node3);
    new_node3->op = new_op3;

    Node *new_node4 = new Node(Adam_Gt2);
    newNodeList.push_back(new_node4);
    Op *new_op4 = new Adam_Gt2Op(node1, b2, new_node4);
    new_node4->op = new_op4;

    Node *new_node5 = new Node(Adam_UpM);
    newNodeList.push_back(new_node5);
    Op *new_op5 = new Adam_UpMOp(new_node1, new_node3, b1, new_node5);
    new_node5->op = new_op5;

    Node *new_node6 = new Node(Adam_UpV);
    newNodeList.push_back(new_node6);
    Op *new_op6 = new Adam_UpMOp(new_node2, new_node4, b2, new_node6);
    new_node6->op = new_op6;

    Node *new_node7 = new Node(Adam_Bt1);
    adamMVNodes.push_back(new_node7);
    new_node7->inputs.push_back(node1);

    Node *new_node8 = new Node(Adam_Bt2);
    adamMVNodes.push_back(new_node8);
    new_node8->inputs.push_back(node1);

    Node *new_node9 = new Node(Adam_V_);
    newNodeList.push_back(new_node9);
    Op *new_op9 = new Adam_V_Op(new_node6, new_node8, e, new_node9);
    new_node9->op = new_op9;

    Node *new_node10 = new Node(Adam_M_);
    newNodeList.push_back(new_node10);
    Op *new_op10 = new Adam_M_Op(new_node5, new_node7, new_node9, lr, new_node10);
    new_node10->op = new_op10;

    Node *new_node11 = new Node(Sub);
    newNodeList.push_back(new_node11);
    Op *new_op11 = new SubOp(node2, new_node10, new_node11);
    new_node11->op = new_op11;
    return new_node11;
}


Node *SoftmaxCrossEntropy_Inference(Node *node1, Node *node2, vector<Node *> &newNodeList) {
    Node *new_node1 = new Node(SoftmaxCrossEntropy_Inference1);
    newNodeList.push_back(new_node1);
    Op *new_op1 = new SoftmaxCrossEntropy_Inference1Op(node1, new_node1);
    new_node1->op = new_op1;

    Node *new_node2 = new Node(SoftmaxCrossEntropy_Inference2);
    newNodeList.push_back(new_node2);
    Op *new_op2 = new SoftmaxCrossEntropy_Inference2Op(new_node1, node2, new_node2);
    new_node2->op = new_op2;

    Node *new_node3 = new Node(SoftmaxCrossEntropy_Inference3);
    newNodeList.push_back(new_node3);
    Op *new_op3 = new SoftmaxCrossEntropy_Inference3Op(new_node2, new_node3);
    new_node3->op = new_op3;
    return new_node3;
}

void SoftmaxCrossEntropy_Accuracy(Node *node1, Node *node2, vector<Node *> &resNodes,
        vector<Node *> &newNodeList) {
    Node *new_node1 = new Node(SoftmaxCrossEntropy_Inference1);
    newNodeList.push_back(new_node1);
    Op *new_op1 = new SoftmaxCrossEntropy_Inference1Op(node1, new_node1);
    new_node1->op = new_op1;

    Node *new_node2 = new Node(SoftmaxCrossEntropy_Inference2);
    newNodeList.push_back(new_node2);
    Op *new_op2 = new SoftmaxCrossEntropy_Inference2Op(new_node1, node2, new_node2);
    new_node2->op = new_op2;

    Node *new_node3 = new Node(SoftmaxCrossEntropy_Inference3);
    newNodeList.push_back(new_node3);
    Op *new_op3 = new SoftmaxCrossEntropy_Inference3Op(new_node2, new_node3);
    new_node3->op = new_op3;

    resNodes.push_back(new_node3);

    Node *new_node4 = new Node(SoftmaxCrossEntropy_Accuracy1);
    newNodeList.push_back(new_node4);
    Op *new_op4 = new SoftmaxCrossEntropy_Accuracy1Op(new_node1, new_node4);
    new_node4->op = new_op4;

    Node *new_node5 = new Node(SoftmaxCrossEntropy_Accuracy2);
    newNodeList.push_back(new_node5);
    Op *new_op5 = new SoftmaxCrossEntropy_Accuracy2Op(new_node2, new_node4, new_node5);
    new_node5->op = new_op5;

    Node *new_node6 = new Node(SoftmaxCrossEntropy_Accuracy3);
    newNodeList.push_back(new_node6);
    Op *new_op6 = new SoftmaxCrossEntropy_Accuracy3Op(new_node5, new_node6);
    new_node6->op = new_op6;

    resNodes.push_back(new_node6);
}

Node *CPredictOp(Node *node1, vector<Node *> &newNodeList) {
    Node *new_node1 = new Node(SoftmaxCrossEntropy_Accuracy1);
    newNodeList.push_back(new_node1);
    Op *new_op1 = new SoftmaxCrossEntropy_Accuracy1Op(node1, new_node1);
    new_node1->op = new_op1;

    Node *new_node2 = new Node(Predict);
    newNodeList.push_back(new_node2);
    Op *new_op2 = new PredictOp(node1, new_node1, new_node2);
    new_node2->op = new_op2;
    return new_node2;
}

Node *Mse_Training_I(Node *node1, Node *node2,
                   vector<Node *> &newNodeList) {
    Node *new_node = new Node(Mse_Training1);
    newNodeList.push_back(new_node);
    Op *new_op = new Mse_Training1Op(node1, node2, new_node);
    new_node->op = new_op;
    return new_node;
}

Node *Mse_Training(Node *node1, Node *node2,
                   vector<Node *> &newNodeList) {
    Node *new_node = new Node(Mse_Training2);
    newNodeList.push_back(new_node);
    Op *new_op = new Mse_Training2Op(node1, node2, new_node);
    new_node->op = new_op;
    return new_node;
}

Node *Mse_Inference(Node *node1,
                    vector<Node *> &newNodeList) {
    Node *new_node1 = new Node(Mse_Inference1);
    newNodeList.push_back(new_node1);
    Op *new_op1 = new Mse_Inference1Op(node1, new_node1);
    new_node1->op = new_op1;

    Node *new_node2 = new Node(Mse_Inference2);
    newNodeList.push_back(new_node2);
    Op *new_op2 = new SoftmaxCrossEntropy_Accuracy3Op(new_node1, new_node2);
    new_node2->op = new_op2;
    return new_node2;
}





