//
// Created by hh on 2022/1/14.
//

#include <set>
#include <iostream>
#include "AutodiffMethod.h"



void getGradients(Node *loss, vector<Node *> &nodes,  vector<Node *> &gNodes, vector<Node *> &newNodeList){

    /* 每个前向传播点对应的outgrad */
    unordered_map<Node *, Node *> nodeGradMap;
    /* 前向传播点按topo序列排 */
    vector<Node *> topoNodes;
    /* 要计算的点序列 */
    vector<Node *> computeNodes;
    computeNodes.push_back(loss);
    findTopoSort(computeNodes, topoNodes);

    nodeGradMap.insert({loss, NULL});

    for(int i=int(topoNodes.size()-1);i>-1;i--){
        Node *thisNode = topoNodes.at(i);
        if(thisNode->inputs.empty() || thisNode->nodetype==Dropout1 || thisNode->nodetype==Dropout_Prob) continue;
        vector<Node *> inputs_grad;
        thisNode->op->gradient(thisNode, nodeGradMap.at(thisNode), inputs_grad, newNodeList);
        for(int j=0;j<thisNode->inputs.size();j++){
            nodeGradMap.insert({thisNode->inputs.at(j), inputs_grad.at(j)});
        }
    }

    for(Node *anode : nodes){
        gNodes.push_back(nodeGradMap.at(anode));
    }
}

void findTopoSort(vector<Node *> &computeNodes, vector<Node *> &topoNodes) {

    unordered_set<Node *> nodeSet;

    for(int i=0;i<computeNodes.size();i++){
        topoSortDfs(computeNodes.at(i), nodeSet, topoNodes);
    }


}

void topoSortDfs(Node *node, unordered_set<Node *> &nodeSet, vector<Node *> &topoNodes) {
    if(nodeSet.count(node)) return;
    nodeSet.insert(node);
    for(int i=0;i<node->inputs.size();i++){
        topoSortDfs(node->inputs.at(i), nodeSet, topoNodes);
    }
    topoNodes.push_back(node);

}
