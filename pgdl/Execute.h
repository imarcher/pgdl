//
// Created by hh on 2022/1/15.
//

#ifndef PGDL_EXECUTE_H
#define PGDL_EXECUTE_H


#include "Autodiff.h"
#include "CreateNode.h"
#include "AutodiffMethod.h"
#include "Graph.h"


/*
 * execute用于对内存原语和计算的运行的维护，使用设备和流
 */
class Execute {
private:
    /* 构造给定的每个点的memory*/
    void getMemory(vector<Node *> &nodes);
    /* 构造占位符的 */
    void getMemoryPlaceholder(vector<Node *> &nodes);
    /* 构造Sgd点，直接用的对应变量的memory，方便执行就地操作*/
    void getMemorySgd(vector<Node *> &nodes);
    /* 构造SgdLr*/
    void getMemorySgdLr();
    /* 构造workspace的内存 */
    void getMemoryWorkpace();
    /* 构造TodoMemoryNodes的内存 */
    void getMemoryTodoMemoryNodes();
    /* 计算给定的每个点的原语*/
    void getPrimitive(vector<Node *> &nodes, vector<primitive> &primitives);
    /* 构造每个Primitive的args*/
    void getMemoryArgs(vector<Node *> &nodes, vector<unordered_map<int, memory>> &memory_args);
    /* 计算 */
    static void compute(stream& astream, vector<primitive> &primitives, vector<unordered_map<int, memory>> &memory_args);
    /* 计算 */
    static void computeHalf(stream& astream, vector<primitive> &primitives, vector<unordered_map<int, memory>> &memory_args, int halfId);
    /* 计算 */
    void compute(stream& astream);
    /* 计算 */
    void computeEX(stream& astream);
    /* 计算 */
    void computeAccuracy(stream& astream);
public:

    Execute(engine &eng, vector<Node *> &new_topoNodesWithoutPV, vector<Node *> &new_sgdNodes, vector<Node *> &new_sgdLrNodes,
            vector<Node *> &new_todoMemoryNodes, unordered_map<Node *, primitive_desc_base> &new_primitive_desc_map,
            unordered_map<Node *, vector<memory::desc>> &new_workspace_desc_map, int xNodeId, vector<Node *> &extraNodeList,
            vector<Node *> &placeholderList, vector<Node *> &variableList, unordered_map<Node *, float> &dropoutKeepProb_map,
            float lr);


    /* 构造每个点的memory（sgdNodes不需要）*/
    void getMemory();
    /* 计算topoNodesWithoutPV和sgdNodes的原语*/
    void getPrimitive();
    /* 构造每个Primitive的args*/
    void getMemoryArgs();
    /* 创建这个execute的流 */
    void setStream();
    /* 计算 */
    void compute();
    /* 计算 */
    void computeEX();
    /* 计算 */
    void computeAccuracy();
    /* 等待stream计算完 */
    void wait();
    /* 更新sgd学习率 */
    void updateSgdLr(float lr);
    /* 每次迭代初始化adam要用的 */
    void iniAdamBt();


    /* 这个execute计算用的引擎 */
    engine eng;
    /* 训练过程中获得loss准确率的node */
    vector<Node *> extraNodeList;
    /* 这个vector保存声明的所有占为符 创建的一定要用*/
    vector<Node *> placeholderList;
    /* 这个vector保存声明的所有变量 创建的一定要用*/
    vector<Node *> variableList;

    /* topoNodes,但没有placeholder，variable */
    vector<Node *> topoNodesWithoutPV;
    /* sgd的node */
    vector<Node *> sgdNodes;
    /* sgdLr的node */
    vector<Node *> sgdLrNodes;
    vector<Node *> todoMemoryNodes;
    unordered_map<Node *, primitive_desc_base> primitive_desc_map;
    unordered_map<Node *, vector<memory::desc>> workspace_desc_map;

    unordered_map<Node *, memory> memory_map;
    unordered_map<Node *, vector<memory>> workspace_map;
    vector<primitive> primitives_topoNodesWithoutPV;
    vector<primitive> primitives_sgdNodes;
    vector<primitive> primitives_extraNodeList;
    unordered_map<Node *, primitive> primitive_map;
    vector<unordered_map<int, memory>> memory_args_topoNodesWithoutPV;
    vector<unordered_map<int, memory>> memory_args_sgdNodes;
    vector<unordered_map<int, memory>> memory_args_extraNodeList;
    float lr;

    //dropout
    unordered_map<Node *, float> dropoutKeepProb_map;

    stream astream;

    int xNodeId;


    //sgdLr的memory
    vector<memory> sgdLrMemorys;
    vector<bool> adamBt;
    float Bt1;
    float Bt2;
};


#endif //PGDL_EXECUTE_H
