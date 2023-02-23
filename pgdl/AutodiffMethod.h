//
// Created by hh on 2022/1/14.
//

#ifndef PGDL_AUTODIFFMETHOD_H
#define PGDL_AUTODIFFMETHOD_H
#include "Autodiff.h"
#include <unordered_set>




/*
 * 获得这些点的导数点
 * loss 要求导数的点（变量） 求得的导数输出
 */
void getGradients(Node *loss, vector<Node *> &nodes, vector<Node *> &gNodes, vector<Node *> &newNodeList);

/*
 * 获得一些点的topo序列用于计算
 * 要计算的点 topo序列
 */
void findTopoSort(vector<Node *> &computeNodes, vector<Node *> &topoNodes);

/*
 * 获得一个点的topo序列用于计算，被findTopoSort调用
 * 要计算的点 已经在序列中的点 topo序列
 */
void topoSortDfs(Node *node, unordered_set<Node *> &nodeSet, vector<Node *> &topoNodes);



#endif //PGDL_AUTODIFFMETHOD_H
