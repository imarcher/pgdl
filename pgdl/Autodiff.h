/*-------------------------------------------------------------------------
 *
 * 组成计算图的点以及点的对应op操作
 * todo 所有compute操作得规划
 *
 *
 *-------------------------------------------------------------------------
 */

#ifndef PGDL_AUTODIFF_H
#define PGDL_AUTODIFF_H

#include <vector>
#include "oneapi/dnnl/dnnl.hpp"
using namespace std;
using namespace dnnl;

/*
 * 点的类型，代表点是干嘛的
 * Placeholder传数据的
 * Variable是变量
 */
typedef enum Nodetype{
    Placeholder,
    Variable,
    Linear,
    Reverse,
    ReverseBackward,
    Relu,
    ReluBackward,
    BN,
    BNBackward,
    BNRelu,
    BNReluBackward,
    Dropout1,
    Dropout_Prob,
    Dropout2,
    Dropout3,
    DropoutBackward,
    Dropout_Inference,
    Add,
    Sub,
    MatMul,
    MatMulBackwardData,
    MatMulBackwardWeights,
    MatMulBackwardBias,
    ReOrder,
    Convolution,
    ConvolutionBackwardData,
    ConvolutionBackwardWeights,
    ConvolutionBackwardBias,
    Pooling,
    PoolingBackward,
    Flat,
    FlatBackward,
    SoftmaxCrossEntropy_Training1,
    SoftmaxCrossEntropy_Training2,
    SgdLr,
    Sgd,
    Adam_M,
    Adam_V,
    Adam_Gt,
    Adam_Gt2,
    Adam_UpM,
    Adam_UpV,
    Adam_Bt1,
    Adam_Bt2,
    Adam_V_,
    Adam_M_,
    SoftmaxCrossEntropy_Inference1,
    SoftmaxCrossEntropy_Inference2,
    SoftmaxCrossEntropy_Inference3,
    SoftmaxCrossEntropy_Accuracy1,
    SoftmaxCrossEntropy_Accuracy2,
    SoftmaxCrossEntropy_Accuracy3,
    Predict,
    Mse_Training1,
    Mse_Training2,
    Mse_Inference1,
    Mse_Inference2
} Nodetype;


class Node;
class Op;




/*
 * 点的存储部分
 * 用来存计算需要的内容
 * 是可以在外面调用属性的
 */
class Node {
public:
    Node(Nodetype nodetype);
    Node(Nodetype nodetype, Op *op);
    ~Node();
    Nodetype nodetype;
    Op *op;
    vector<Node *> inputs;
    memory::desc *desc;
    /* 只有一些需要前向的信息的导数点有，代表前向的点，用于传primitive_desc */
    Node *forwardNode;
};

/*
 * 点的计算部分，用来计算
 * 存特别计算需要的东西，在op内部用的属性，不在外面调用
 */
class Op{
public:
    virtual ~Op();
    virtual void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                         unordered_map<Node *, vector<memory>> &workspace_map) = 0;
    virtual primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) = 0;
    virtual void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) = 0;
    virtual void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map, 
            unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) = 0;
};

class LinearOp: public Op{
public:
    LinearOp(Node *node1, float a, float b, Node *node);
    ~LinearOp() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                 unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
                     unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:
    float a;
    float b;
};

/*
 * Eltwise
 * 矩阵取反
 */
class ReverseOp: public Op{
public:
    ReverseOp(Node *node1, Node *node);
    ~ReverseOp() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                         unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map, 
            unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:

};

/*
 * Eltwise
 * 矩阵取反导数
 */
class ReverseBackwardOp: public Op{
public:
    ReverseBackwardOp(Node *node1, Node *node);
    ~ReverseBackwardOp() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                         unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map, 
            unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:

};

/*
 * Eltwise
 * relu
 */
class ReluOp: public Op{
public:
    ReluOp(Node *node1, Node *node);
    ~ReluOp() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                         unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map, 
            unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:

};

/*
 * Eltwise
 * relu反向 导数，relu原输入
 */
class ReluBackwardOp: public Op{
public:
    ReluBackwardOp(Node *node1, Node *node2, Node *node);
    ~ReluBackwardOp() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                         unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map, 
            unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:

};


/*
 * BN
 */
class BNOp: public Op{
public:
    BNOp(Node *node1, Node *node);
    ~BNOp() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                 unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
                     unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:

};

/*
 * BN反向
 */
class BNBackwardOp: public Op{
public:
    BNBackwardOp(Node *node1, Node *node2, Node *node);
    ~BNBackwardOp() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                 unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
                     unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:

};


/*
 * BN+relu
 * relu
 */
class BNReluOp: public Op{
public:
    BNReluOp(Node *node1, Node *node);
    ~BNReluOp() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                 unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
                     unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:

};

/*
 * BN+reluf反向
 * relu
 */
class BNReluBackwardOp: public Op{
public:
    BNReluBackwardOp(Node *node1, Node *node2, Node *node);
    ~BNReluBackwardOp() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                 unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
                     unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:

};


/*
 * dropout
 * 1/p * (dropout1 < keep_prob) * x
 * 这个点运行1/keep_prob * (dropout1 < keep_prob)
 * 这个点在原内存上运行就可以
 * dropout1 keep_prob
 */
class Dropout2Op: public Op{
public:
    Dropout2Op(Node *node1, Node *node2, float keep_prob, Node *node);
    ~Dropout2Op() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                 unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
                     unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:
    float keep_prob;
};

/*
 * dropout 前向和反向
 * 1/p * (dropout1 < keep_prob) * x
 * 这个点运行 * x
 * 这个点也可以在原内存上运行，反正x和导数没用了
 * x 1/p*(dropout1 < keep_prob)
 * 导数 1/p*(dropout1 < keep_prob)
 */
class Dropout3Op: public Op{
public:
    Dropout3Op(Node *node1, Node *node2, Node *node);
    ~Dropout3Op() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                 unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
                     unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:

};

/*
 * 矩阵相加
 */
class AddOp: public Op{
public:
    AddOp(Node *node1, Node *node2, Node *node);
    ~AddOp() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                         unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map, 
            unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:

};

/*
 * 矩阵相减
 */
class SubOp: public Op{
public:
    SubOp(Node *node1, Node *node2, Node *node);
    ~SubOp() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                         unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map, 
            unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:

};

/*
 * dense
 * 矩阵乘法，带偏置
 * src, weights, bias
 * mk
 * kn
 * 1n
 */
class MatMulOp: public Op{
public:
    MatMulOp(Node *node1, Node *node2, Node *node3, Node *node);
    ~MatMulOp() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                         unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map, 
            unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:

};

/*
 * dense
 * 矩阵乘法反向传播
 * 导数，weights
 */
class MatMulBackwardDataOp: public Op{
public:
    MatMulBackwardDataOp(Node *node1, Node *node2, Node *node);
    ~MatMulBackwardDataOp() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                         unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map, 
            unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:
    /* weights转置后的desc*/
    memory::desc *node2_new_desc;
};

/*
 * dense
 * 矩阵乘法反向传播
 * data，导数
 */
class MatMulBackwardWeightsOp: public Op{
public:
    MatMulBackwardWeightsOp(Node *node1, Node *node2, Node *node);
    ~MatMulBackwardWeightsOp() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                         unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map, 
            unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:
    /* data转置后的desc*/
    memory::desc *node1_new_desc;
};

/*
 * dense
 * 矩阵乘法反向传播
 * 导数
 */
class MatMulBackwardBiasOp: public Op{
public:
    MatMulBackwardBiasOp(Node *node1, Node *node);
    ~MatMulBackwardBiasOp() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                         unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map, 
            unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:

};

/*
 * 重排序,卷积用
 */
class ReOrderOp: public Op{
public:
    ReOrderOp(Node *node1, Node *node);
    ~ReOrderOp() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                         unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map, 
            unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:

};

/*
 * 卷积
 * 步长，高扩展，宽扩展
 * strides_dims = {SH, SW};
 * padding_dims_l = {PH_L, PW_L}  padding_dims_r = {PH_R, PW_R}
 */
class ConvolutionOp: public Op{
public:
    ConvolutionOp(Node *node1, Node *node2, Node *node3, Node *node, memory::dims &strides_dims,
            memory::dims &padding_dims_l, memory::dims &padding_dims_r);
    ~ConvolutionOp() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                         unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map, 
            unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:
    memory::dims strides_dims;
    memory::dims padding_dims_l;
    memory::dims padding_dims_r;
};

/*
 * 卷积反向
 * weights 导数
 */
class ConvolutionBackwardDataOp: public Op{
public:
    ConvolutionBackwardDataOp(Node *node1, Node *node2, Node *node, memory::dims &strides_dims,
                  memory::dims &padding_dims_l, memory::dims &padding_dims_r);
    ~ConvolutionBackwardDataOp() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                         unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map, 
            unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:
    memory::dims strides_dims;
    memory::dims padding_dims_l;
    memory::dims padding_dims_r;
};

/*
 * 卷积反向
 * data bias导数的内存(还没算) 导数
 */
class ConvolutionBackwardWeightsOp: public Op{
public:
    ConvolutionBackwardWeightsOp(Node *node1, Node *node2, Node *node3, Node *node, memory::dims &strides_dims,
                              memory::dims &padding_dims_l, memory::dims &padding_dims_r);
    ~ConvolutionBackwardWeightsOp() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                         unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map, 
            unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:
    memory::dims strides_dims;
    memory::dims padding_dims_l;
    memory::dims padding_dims_r;
};

/*
 * 池化
 */
class PoolingOp: public Op{
public:
    PoolingOp(Node *node1, Node *node, memory::dims &strides_dims, memory::dims &kernel,
                                 memory::dims &padding_dims_l, memory::dims &padding_dims_r);
    ~PoolingOp() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                         unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map, 
            unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:
    memory::dims strides_dims;
    memory::dims kernel;
    memory::dims padding_dims_l;
    memory::dims padding_dims_r;
};

/*
 * 池化反向
 * data bias导数的内存(还没算) 导数
 */
class PoolingBackwardOp: public Op{
public:
    PoolingBackwardOp(Node *node1, Node *node, memory::dims &strides_dims, memory::dims &kernel,
              memory::dims &padding_dims_l, memory::dims &padding_dims_r);
    ~PoolingBackwardOp() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                 unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
                     unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:
    memory::dims strides_dims;
    memory::dims kernel;
    memory::dims padding_dims_l;
    memory::dims padding_dims_r;
};

/*
 * 平坦化 4维转2维
 * 可能前面使用了其他内存格式，这里转换为nchw，再转为ab
 */
class FlatOp: public Op{
public:
    FlatOp(Node *node1, Node *node);
    ~FlatOp() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                 unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
                     unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:
    memory::desc *node_new_desc;
};

/*
 * 平坦化反向 2维转4维
 * 可能前面使用了其他内存格式，这里转换为nchw，再转为原来格式
 */
class FlatBackwardOp: public Op{
public:
    FlatBackwardOp(Node *node1, Node *node);
    ~FlatBackwardOp() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                 unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
                     unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:
    memory::desc *node_new_desc;
};


/*
 * softmax交叉熵
 * 导数
 * softmax(x)-y
 * 这个点运行softmax(x)
 * 导数是特化的人工求导，跟普通点不一样
 */
class SoftmaxCrossEntropy_Training1Op: public Op{
public:
    SoftmaxCrossEntropy_Training1Op(Node *node1, Node *node);
    ~SoftmaxCrossEntropy_Training1Op() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                         unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map, 
            unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:

};



/*
 * softmax交叉熵
 * 导数
 * softmax(x)-y
 * 这个点运行-y
 * 导数是特化的人工求导，跟普通点不一样
 */
class SoftmaxCrossEntropy_Training2Op: public Op{
public:
    SoftmaxCrossEntropy_Training2Op(Node *node1, Node *node2, Node *node);
    ~SoftmaxCrossEntropy_Training2Op() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                         unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map, 
            unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:

};

/*
 * sgd
 * lr*导数
 * 导数 lr
 */
class SgdOp: public Op{
public:
    SgdOp(Node *node1, Node *node2, Node *node);
    ~SgdOp() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                         unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map, 
            unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:

};

/*
 * adam
 * (1-b2)gt^2
 */
class Adam_Gt2Op: public Op{
public:
    Adam_Gt2Op(Node *node1, float b2, Node *node);
    ~Adam_Gt2Op() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                 unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
                     unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:
    float b2;
};

/*
 * adam
 * m = b1m + (1-b1)gt
 */
class Adam_UpMOp: public Op{
public:
    Adam_UpMOp(Node *node1, Node *node2, float b1, Node *node);
    ~Adam_UpMOp() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                 unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
                     unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:
    float b1;
};

/*h
 * adam
 * v_ = v/(1-b2^t)
 * v_ = sqrt(v_) + e
 */
class Adam_V_Op: public Op{
public:
    Adam_V_Op(Node *node1, Node *node2, float e, Node *node);
    ~Adam_V_Op() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                 unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
                     unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:
    float e;
};

/*
 * adam
 * lr*m/v_
 */
class Adam_M_Op: public Op{
public:
    Adam_M_Op(Node *node1, Node *node2, Node *node3, float lr, Node *node);
    ~Adam_M_Op() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                 unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
                     unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:
    float lr;
};


/*
 * softmax交叉熵
 * 推理 输出一个 1，1的loss
 * logsoftmax(x)*y -reducemean
 * 这个点运行logsoftmax(x)
 * 不准求导
 */
class SoftmaxCrossEntropy_Inference1Op: public Op{
public:
    SoftmaxCrossEntropy_Inference1Op(Node *node1, Node *node);
    ~SoftmaxCrossEntropy_Inference1Op() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                         unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map, 
            unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:

};

/*
 * softmax交叉熵
 * 推理 输出一个 1，1的loss
 * logsoftmax(x)*y -reducemean
 * 这个点运行logsoftmax(x)*y
 * 不准求导
 *  logsoftmax(x) y
 */
class SoftmaxCrossEntropy_Inference2Op: public Op{
public:
    SoftmaxCrossEntropy_Inference2Op(Node *node1, Node *node2, Node *node);
    ~SoftmaxCrossEntropy_Inference2Op() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                         unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map, 
            unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:

};

/*
 * softmax交叉熵
 * logsoftmax(x)*y -reducemean
 * 这个点运行 -reducemean
 * 不准求导
 * 推理 输出一个 1，1的loss
 */
class SoftmaxCrossEntropy_Inference3Op: public Op{
public:
    SoftmaxCrossEntropy_Inference3Op(Node *node1, Node *node);
    ~SoftmaxCrossEntropy_Inference3Op() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                         unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map, 
            unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:

};

/*
 * softmax交叉熵
 * 求准确率 输出一个 1，1的准确率 跟loss一起算，真不一定有手动快
 * reducemean(reducesum(logsoftmax(x)*y) == reducemax(logsoftmax(x)))
 * 这个点运行 reducemax(logsoftmax(x))
 * 不准求导
 */
class SoftmaxCrossEntropy_Accuracy1Op: public Op{
public:
    SoftmaxCrossEntropy_Accuracy1Op(Node *node1, Node *node);
    ~SoftmaxCrossEntropy_Accuracy1Op() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                         unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map, 
            unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:

};

/*
 * softmax交叉熵
 * 求准确率 输出一个 1，1的准确率 跟loss一起算，真不一定有手动快
 * reducemean(reducesum(logsoftmax(x)*y) == reducemax(logsoftmax(x)))
 * 这个点运行 reducesum(logsoftmax(x)*y) == reducemax(logsoftmax(x))
 * 不准求导
 * logsoftmax(x)*y reducemax(logsoftmax(x)
 */
class SoftmaxCrossEntropy_Accuracy2Op: public Op{
public:
    SoftmaxCrossEntropy_Accuracy2Op(Node *node1, Node *node2, Node *node);
    ~SoftmaxCrossEntropy_Accuracy2Op() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                         unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map, 
            unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:

};


/*
 * softmax交叉熵
 * 求准确率 输出一个 1，1的准确率 跟loss一起算，真不一定有手动快
 * reducemean(reducesum(logsoftmax(x)*y) == reducemax(logsoftmax(x)))
 * 这个点运行 reducemean
 * 不准求导
 */
class SoftmaxCrossEntropy_Accuracy3Op: public Op{
public:
    SoftmaxCrossEntropy_Accuracy3Op(Node *node1, Node *node);
    ~SoftmaxCrossEntropy_Accuracy3Op() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                         unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map, 
            unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:

};



/*
 * 预测
 * logsoftmax(x) == reducemax(logsoftmax(x)))
 * 这个点运行 ==
 * 不准求导
 */
class PredictOp: public Op{
public:
    PredictOp(Node *node1, Node *node2, Node *node);
    ~PredictOp() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                 unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
                     unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:

};


/*
 * mse训练
 * reducemean ((x-y)^2)
 * 这个点运行 x-y 导数2*(x-y)
 */
class Mse_Training1Op: public Op{
public:
    Mse_Training1Op(Node *node1, Node *node2, Node *node);
    ~Mse_Training1Op() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                 unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
                     unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:

};

/*
 * mse训练
 * reducemean ((x-y)^2)
 * 这个点运行 x-y 导数2*(x-y)
 */
class Mse_Training2Op: public Op{
public:
    Mse_Training2Op(Node *node1, Node *node2, Node *node);
    ~Mse_Training2Op() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                 unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
                     unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:

};


/*
 * mse训练
 * reducemean ((x-y)^2)
 * 这个点运行 ^2
 * 不准求导
 */
class Mse_Inference1Op: public Op{
public:
    Mse_Inference1Op(Node *node1, Node *node);
    ~Mse_Inference1Op() override;
    void getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                 unordered_map<Node *, vector<memory>> &workspace_map) override;
    primitive getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) override;
    void gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) override;
    void infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
                     unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) override;

private:

};




#endif //PGDL_AUTODIFF_H
