//
// Created by hh on 2022/3/9.
//

#include "ModelClient.h"
#include "Manager.h"


int GetVariableListSize_Client(void *manager) {
    Manager *pManager = (Manager *)manager;
    return pManager->getVariableListSize();
}

int GetVariableTag_Client(int nodeId, void *manager) {
    Manager *pManager = (Manager *)manager;
    memory::format_tag tag = pManager->variableTagList.at(nodeId);
    int tag1 = 0;
    switch (tag) {
        case memory::format_tag::ab:
            tag1 = 0;
            break;
        case memory::format_tag::a:
            tag1 = 1;
            break;
        case memory::format_tag::nhwc:
            tag1 = 2;
            break;
        case memory::format_tag::nchw:
            tag1 = 3;
            break;
        default:
            return -1;
    }
    return tag1;
}

int GetVariableNdim_Client(int nodeId, void *manager) {
    Manager *pManager = (Manager *)manager;
    return pManager->variableList.at(nodeId)->desc->data.ndims;
}

int GetVariableSize_Client(int nodeId, void *manager) {
    Manager *pManager = (Manager *)manager;
    return pManager->variableList.at(nodeId)->desc->get_size() / 4;
}

int *GetVariableDims_Client(int nodeId, void *manager) {
    Manager *pManager = (Manager *)manager;
    memory::dims adims = pManager->variableList.at(nodeId)->desc->dims();
    int *res = new int[adims.size()];
    for(int i=0;i<adims.size();i++){
        res[i] = adims.at(i);
    }
    return res;
}

float *GetVariableData_Client(int nodeId, void *manager) {
    Manager *pManager = (Manager *)manager;
    return pManager->getVariableData(pManager->variableList.at(nodeId));
}

void SetVariablePG_Client(int nodeId, void *dst, void *manager) {
    Manager *pManager = (Manager *)manager;
    pManager->setVariablePG(pManager->variableList.at(nodeId), dst);
}

void SetVariableData_Client(int nodeId, int tag, int ndim, int *dims, float *data, void *manager) {
    Manager *pManager = (Manager *)manager;
    memory::format_tag tag1;
    switch (tag) {
        case 0:
            tag1 = memory::format_tag::ab;
            break;
        case 1:
            tag1 = memory::format_tag::a;
            break;
        case 2:
            tag1 = memory::format_tag::nhwc;
            break;
        case 3:
            tag1 = memory::format_tag::nchw;
            break;
        case 4:
            tag1 = memory::format_tag::oihw;
            break;
        default:
            return;
    }
    memory::dims dims1;
    for(int i=0;i<ndim;i++){
        dims1.push_back(dims[i]);
    }

    pManager->setVariableData(pManager->variableList.at(nodeId), dims1, tag1, data);
}
