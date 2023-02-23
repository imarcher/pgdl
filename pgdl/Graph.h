//
// Created by hh on 2022/3/4.
//

#ifndef PGDL_GRAPH_H
#define PGDL_GRAPH_H


#include "vector"
#include "Autodiff.h"
#include "unordered_set"
/*
 * 保存计算图的相关list
 * 对于一个引擎eng一个Gragh
 */
class Graph {
public:
    /* 复制一份topoNodes */
    Graph(vector<Node *> gNodes);



    /* topoNodes,但没有placeholder，variable */
    vector<Node *> topoNodesWithoutPV;
    /* 要先声明内存的*/
    vector<Node *> todoMemoryNodes;
    /* 剃度下降的学习率的node */
    vector<Node *> sgdLrNodes;
    /* 剃度下降的node */
    vector<Node *> sgdNodes;

    /* variable对应的导数 */
    vector<Node *> gNodes;

    /* 调用initshape后所得到的逻辑primitive，与eng匹配 */
    unordered_map<Node *, primitive_desc_base> primitive_desc_map;
    /* 调用initshape后所得到的逻辑workspace */
    unordered_map<Node *, vector<memory::desc>> workspace_desc_map;
    int xNodeId;

    //要delete的
    unordered_map<Node *, memory::desc *> tagChangeMap;

};


#endif //PGDL_GRAPH_H
