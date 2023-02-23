//
// Created by hh on 2022/1/15.
//

#include <iostream>
#include <utility>
#include <cstring>
#include <thread>
#include "Execute.h"


void Execute::
getMemory(vector<Node *> &nodes) {
    for(Node *anode: nodes){
        if(anode->nodetype==Dropout2 || anode->nodetype==Dropout3) {
            memory_map.insert({anode, memory_map.at(anode->inputs.at(0))});
            continue;
        }
        memory_map.insert({anode, memory(*(anode->desc), eng)});
    }
}

void Execute::getMemoryPlaceholder(vector<Node *> &nodes) {
    for(Node *anode: nodes){
        memory_map.insert({anode, memory(*(anode->desc), eng)});
    }
}

void Execute::getMemorySgd(vector<Node *> &nodes) {
    for(Node *anode: nodes){
        if(anode->nodetype == Sgd
        || anode->nodetype == Adam_Gt
        || anode->nodetype == Adam_Gt2
        || anode->nodetype == Adam_V_
        || anode->nodetype == Adam_M_) {
            memory_map.insert({anode, memory(*(anode->desc), eng)});
        } else{
            memory_map.insert({anode, memory_map.at(anode->inputs.at(0))});
        }
    }
}

void Execute::getMemorySgdLr() {
    for(Node *anode: sgdLrNodes){
        memory nodeMemory = memory(*(anode->desc), eng);
        if(anode->nodetype==SgdLr){
            float *pNodeMemory = (float *)(nodeMemory.get_data_handle());
            pNodeMemory[0] = lr;
            memory_map.insert({anode, nodeMemory});
            sgdLrMemorys.push_back(nodeMemory);
        }
        else if(anode->nodetype==Adam_Bt1){
            memory_map.insert({anode, nodeMemory});
            sgdLrMemorys.push_back(nodeMemory);
            adamBt.push_back(true);
        }
        else if(anode->nodetype==Adam_Bt2){
            memory_map.insert({anode, nodeMemory});
            sgdLrMemorys.push_back(nodeMemory);
            adamBt.push_back(false);
        }
        else {
            float *pNodeMemory = (float *)(nodeMemory.get_data_handle());
            size_t anodesize = anode->desc->get_size();
            memset(pNodeMemory, 0, anodesize);
            memory_map.insert({anode, nodeMemory});
        }
    }
}

void Execute::getMemoryWorkpace() {
    unordered_map<Node *, vector<memory::desc>>::iterator it;
    for(it=workspace_desc_map.begin();it!=workspace_desc_map.end();it++){
        vector<memory> vw;
        for(memory::desc adesc: it->second){
            vw.emplace_back(adesc, eng);
        }
        workspace_map.insert({it->first, vw});
    }
}

void Execute::getMemoryTodoMemoryNodes() {
    for(Node *anode: todoMemoryNodes){
        if(anode->nodetype==Dropout_Prob){
            memory nodeMemory = memory(*(anode->desc), eng);
            float *pNodeMemory = (float *)(nodeMemory.get_data_handle());
            pNodeMemory[0] = dropoutKeepProb_map.at(anode);
            memory_map.insert({anode, nodeMemory});
            continue;
        }
        memory_map.insert({anode, memory(*(anode->desc), eng)});
    }
}

void Execute::getPrimitive(vector<Node *> &nodes, vector<primitive> &primitives) {
    for(Node * anode: nodes){
        primitive p = anode->op->getPrimitive(anode, eng, primitive_desc_map);
        primitives.push_back(p);
    }

}

void Execute::getMemoryArgs(vector<Node *> &nodes, vector<unordered_map<int, memory>> &memory_args) {
    for (Node *anode: nodes) {
        vector<memory> ms;
        unordered_map<int, memory> args;
        for (Node *inputsnode: anode->inputs) {
            ms.push_back(memory_map.at(inputsnode));
        }
        ms.push_back(memory_map.at(anode));
        anode->op->getargs(anode, ms, args, workspace_map);
        memory_args.push_back(args);
    }
}

void Execute::compute(stream &astream, vector<primitive> &primitives, vector<unordered_map<int, memory>> &memory_args) {
    for(int i=0;i<primitives.size();i++){
        primitives.at(i).execute(astream, memory_args.at(i));
    }
}

void Execute::computeHalf(stream& astream, vector<primitive> &primitives, vector<unordered_map<int, memory>> &memory_args, int halfId) {
    for(int i=0;i<=halfId;i++){
        primitives.at(i).execute(astream, memory_args.at(i));
    }
}


void Execute::compute(stream &astream) {
    compute(astream,primitives_topoNodesWithoutPV,memory_args_topoNodesWithoutPV);
    compute(astream,primitives_sgdNodes,memory_args_sgdNodes);
}

void Execute::computeEX(stream &astream) {
    compute(astream,primitives_topoNodesWithoutPV,memory_args_topoNodesWithoutPV);
    compute(astream,primitives_sgdNodes,memory_args_sgdNodes);
    compute(astream,primitives_extraNodeList,memory_args_extraNodeList);
}

void Execute::computeAccuracy(stream& astream) {
    computeHalf(astream,primitives_topoNodesWithoutPV,memory_args_topoNodesWithoutPV,xNodeId);
    compute(astream,primitives_extraNodeList,memory_args_extraNodeList);
}


Execute::Execute(engine &eng, vector<Node *> &new_topoNodesWithoutPV, vector<Node *> &new_sgdNodes, vector<Node *> &new_sgdLrNodes,
                 vector<Node *> &new_todoMemoryNodes, unordered_map<Node *, primitive_desc_base> &new_primitive_desc_map,
                 unordered_map<Node *, vector<memory::desc>> &new_workspace_desc_map, int xNodeId, vector<Node *> &extraNodeList,
                 vector<Node *> &placeholderList, vector<Node *> &variableList, unordered_map<Node *, float> &dropoutKeepProb_map,
                 float lr) {
    this->eng = eng;
    this->topoNodesWithoutPV = new_topoNodesWithoutPV;
    this->sgdNodes = new_sgdNodes;
    this->sgdLrNodes = new_sgdLrNodes;
    this->todoMemoryNodes = new_todoMemoryNodes;
    this->primitive_desc_map = new_primitive_desc_map;
    this->workspace_desc_map = new_workspace_desc_map;
    this->xNodeId = xNodeId;
    this->extraNodeList = extraNodeList;
    this->placeholderList = placeholderList;
    this->variableList = variableList;
    this->dropoutKeepProb_map = dropoutKeepProb_map;
    this->lr = lr;
    Bt1 = 1;
    Bt2 = 1;
}



void Execute::getMemory() {
    getMemoryPlaceholder(placeholderList);
    getMemory(variableList);
    getMemoryTodoMemoryNodes();
    getMemoryWorkpace();
    getMemory(topoNodesWithoutPV);
    getMemorySgdLr();
    getMemorySgd(sgdNodes);
    getMemory(extraNodeList);
}

void Execute::getPrimitive() {
    getPrimitive(topoNodesWithoutPV, primitives_topoNodesWithoutPV);
    getPrimitive(sgdNodes, primitives_sgdNodes);
    getPrimitive(extraNodeList, primitives_extraNodeList);
}


void Execute::getMemoryArgs() {
    getMemoryArgs(topoNodesWithoutPV, memory_args_topoNodesWithoutPV);
    getMemoryArgs(sgdNodes, memory_args_sgdNodes);
    getMemoryArgs(extraNodeList, memory_args_extraNodeList);
}


void Execute::setStream() {
    astream = stream(eng);
}

void Execute::compute() {
    compute(astream);
}



void Execute::computeEX() {
    computeEX(astream);
//    astream.wait();
//    for(Node *anode: topoNodesWithoutPV){
//        cout<<anode->nodetype<<":"<<endl;
//        memory m = memory_map.at(anode);
//        int nodesize = m.get_desc().get_size() / 4;
//        float * da = (float *)m.get_data_handle();
//        for(int i=0;i<10;i++) {
//            cout<<da[i]<<" ";
//        }
//        cout<<endl;
//    }
//    cout<<"sgd:"<<endl;
//    for(Node *anode: sgdNodes){
//        cout<<anode->nodetype<<":"<<endl;
//        memory m = memory_map.at(anode);
//        int nodesize = m.get_desc().get_size() / 4;
//        float * da = (float *)m.get_data_handle();
//        for(int i=0;i<10;i++) {
//            cout<<da[i]<<" ";
//        }
//        cout<<endl;
//    }
//    cout<<"ext:"<<endl;
//    for(Node *anode: extraNodeList){
//        cout<<anode->nodetype<<":"<<endl;
//        memory m = memory_map.at(anode);
//        int nodesize = m.get_desc().get_size() / 4;
//        float * da = (float *)m.get_data_handle();
//        for(int i=0;i<10;i++) {
//            cout<<da[i]<<" ";
//        }
//        cout<<endl;
//    }
}

void Execute::computeAccuracy() {
    computeAccuracy(astream);
}

void Execute::wait() {
    astream.wait();
}

void Execute::updateSgdLr(float lr) {
    for(memory& amemory: sgdLrMemorys){
        auto *pNodeMemory = (float *)(amemory.get_data_handle());
        pNodeMemory[0] = lr;
    }
}

void Execute::iniAdamBt() {
    Bt1 = Bt1 * (float)0.9;
    Bt2 = Bt2 * (float)0.999;
    for(int i=0;i<adamBt.size();i++){
        float *pNodeMemory = (float *)(sgdLrMemorys.at(i).get_data_handle());
        if (adamBt.at(i) == true) pNodeMemory[0] = 1 - Bt1;
        else pNodeMemory[0] = 1 - Bt2;
    }
}










