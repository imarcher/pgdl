//
// Created by hh on 2021/12/20.
//

#ifndef PGDL_MANAGER_H
#define PGDL_MANAGER_H

#include <vector>
#include <thread>
#include "Autodiff.h"

#include "AutodiffMethod.h"
#include "Execute.h"
#include "Graph.h"

using namespace std;

typedef enum Gd {
    gd_sgd,
    gd_adam
} Gd;

/*
 * 流程控制程序
 * 保存节点
 * 构建计算图
 * 声明Execute执行
 *
 */
class Manager {



private:


    /* 构造计算图topoNodes，topoNodesWithoutPV, 这里是train，计算导数，先不管计算loss准确率 */
    void initTraining(Node *loss);
    /* 构造计算图topoNodes，topoNodesWithoutPV, 这里是Inference，计算loss */
    void initInference(Node *loss);
    /* 构造计算图topoNodes，topoNodesWithoutPV, 这里是Accuracy，计算loss准确率 */
    void initAccuracy(vector<Node *> &nodes);
    /* 根据变量和gNodes构造sgd的Node */
    void getSgdNode(Gd gd);
    /* 用reorder后的导数替换导数 */
    void replaceGNodes(Node *old_node, Node *new_node, vector<Node *> &old_gNodes);
    /* 计算topoNodesWithoutPV的desc */
    void getShapes();
    /* 计算sgdLrNodes的desc */
    void getSgdLrShapes();
    /* 计算sgdNodes的desc */
    void getSgdShapes();
    /* 计算extraNodeList的desc */
    void getExtraShapes();
    /* 创建几个execute来计算(sgd)*/
    void getExecutes(int num);
    /* 获得这个点在第几个execute的值（必须是一个数） */
    float getNodeValue(Node *anode, int id);
    /* 获得第几个execute的loss */
    float getloss(int id);
    /* 获得第几个execute的准确率 */
    float getaccuracy(int id);
    /* 获得引擎和对应的计算图，这里和execute的创建目前是手动调 */
    void getGraphEngines();
    /* 获得x在topoNodeswithoutPV中的位置 */
    void getXNodeId();
    /* 从保存的参数中读入 */
    void setVariableData(Node *node, memory::desc &adesc, float *data, Execute *aexecute);
    /* 初始化adam的1-bt^t */
    void iniAdamBt();


public:

    Manager();
    ~Manager();

    void createNewNodeList();
    void deleteNewNodeList();
    void deleteSgdNodeList();
    void createPlaceholderList();
    void createVariableList();
    /* 客户端获得点的标记用 */
    int getClientNode();


    //构造计算图topoNodes，topoNodesWithoutPV,计算按这个顺序来算

    /* softmax交叉商train，用几个线程来并行，计算导数，不管计算loss准确率，会调用initTraining，getSgdNode，getShapes, getExecutes */
    void softmaxCrossEntropy_Training(Node *x, Node *y, float lr, Gd gd, int num);
    /* softmax交叉商Inference，用几个线程来并行，计算loss，会调用initInference，getShapes, getExecutes */
    void softmaxCrossEntropy_Inference(Node *x, Node *y, int num);
    /* softmax交叉商Accuracy，用几个线程来并行，计算loss准确率，会调用initAccuracy，getShapes, getExecutes */
    void softmaxCrossEntropy_Accuracy(Node *x, Node *y, int num);
    /* softmax交叉商train，用几个线程来并行，计算导数，可以选择计算loss准确率，会调用initTraining，getSgdNode，getShapes, getExecutes */
    void softmaxCrossEntropy_Training_Accuracy(Node *x, Node *y, float lr, Gd gd, int num);
    /* 预测出one-hot编码 */
    void predict(Node *x, int num);

    /* msetrain, 计算loss */
    void mse_Training(Node *x, Node *y, float lr, Gd gd, int num);
    void mse_Training_Inference(Node *x, Node *y, float lr, Gd gd, int num);

    /* 计算结果 */
    void inference(Node *x, int num);

    /* 准备内存 与设备挂钩，这里先设置为使用cpu0，以后再改 */
    void prepareMemory();
    /* 计算所有的execute */
    void compute();
    /* 计算所有的execute,并算下额外的，比如训练算loss准确率 */
    void computeEX();
    /* 计算前向的execute,并算下额外的，比如训练算loss准确率，但不训练 */
    void computeAccuracy();
    /* 等待所有execute计算完，这里先同步，异步再说 */
    void wait();
    /* 获得所有execute的平均loss */
    float getLoss();
    /* 获得所有execute的平均准确率 */
    float getAccuracy();

    /* 把数据读入对应execute的对应点的内存中,标量 */
    void readtoNodeScalar(int data, Node * node, int executeId, int dataId);
    /* 把数据读入对应execute的对应点的内存中,标量 */
    void readtoNodeReal(float data, Node * node, int executeId, int dataId);
    /* 把数据读入对应execute的对应点的内存中，数组 */
    void readtoNodeArr(float *data, Node * node, int executeId, int dataId);
    /* 初始化所有变量，这里先设置为0 */
    void initVariables();
    /* 初始化所有dropout1 */
    void initDropout();
    /* 初始化所有dropout1为0，为所有都保留 */
    void initDropoutZero();
    /* 初始化node为0 */
    void initNodeZero(Node *node, Execute * aexecute);
    /* 初始化node为高斯随机数 */
    void initNodeNormal(Node *node, Execute * aexecute);
    /* 初始化node为0到1的数 */
    void initNodeProb(Node *node, Execute * aexecute);


    /* 获得variableList的size */
    int getVariableListSize();
    /* 如果改变了形状（比如卷积核），要加reorder转换 */
    float *getVariableData(Node *node);
    /* 如果改变了形状（比如卷积核），要加reorder转换 */
    void setVariablePG(Node *node, void *dst);
    /* 从保存的参数中读入 */
    void setVariableData(Node *node, memory::dims &adims, memory::format_tag atag, float *data);



    /* 获得predict的n */
    int getPredictN();
    /* 获得predict的第i个的标签 */
    int getPredictLabel(int dataId);

    /* 更新sgd学习率 */
    void updateSgdLr(float new_lr);


    /* 这个vector保存声明的所有node节点用于释放，避免内存泄漏 */
    vector<Node *> newNodeList;
    /* 这个vector保存声明的所有占为符 创建的一定要用*/
    vector<Node *> placeholderList;
    /* 这个vector保存声明的所有变量 创建的一定要用*/
    vector<Node *> variableList;
    /* sgd的node */
    vector<Node *> sgdNodeList;
    /* 训练过程中获得loss准确率的node */
    vector<Node *> extraNodeList;
    /* 训练过程中dropout1的node */
    vector<Node *> dropoutNodeList;

    /* 要计算导数所需点的计算顺序 这个顺序是有说法的，调度并行，但这里先不管，先串行，先主干，再分支 */
    vector<Node *> topoNodes;
    /* topoNodes,但没有placeholder，variable */
    vector<Node *> topoNodesWithoutPV;
    /* variable对应的导数 */
    vector<Node *> gNodes;
    Node *loss;
    Node *accuracy;
    /* x是训练完成后查看准确率用的，没有训练过程, x是计算extra的在普通list中的最后一个点*/
    Node *xNode;
    Node *predictNode;




    /* 存相同引擎的计算图 */
    vector<Graph *> graphList;
    vector<engine> engineList;

    /* 关于内存和源语和运行(变量更新除外)，使用这个类 */
    vector<Execute *> executeList;
    int executeSize;

    /* 存一个node除了n之外的内存大小 */
    unordered_map<Node *, int> memorySize_map;


    //model
    vector<memory::format_tag> variableTagList;

    //dropout
    unordered_map<Node *, float> dropoutKeepProb_map;


    //要初始化
    /* 学习率 */
    float lr;
    /* 迭代次数 */
    Gd gd;


};






#endif //PGDL_MANAGER_H
