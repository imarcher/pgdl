//
// Created by hh on 2022/2/22.
//

#include "CreateNodeClient.h"
#include "oneapi/dnnl/dnnl.hpp"

int CreatePlaceholder(int *idims, int num, int tag, void *manager){
    Manager *pManager = (Manager *)manager;
    memory::dims dims1;
    for(int i=0;i<num;i++) {
        dims1.push_back(idims[i]);
    }
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
            return -1;
    }
    CreatePlaceholder(dims1, memory::data_type::f32, tag1,
            pManager->newNodeList, pManager->placeholderList);
    return pManager->getClientNode();
}

int CreateVariable(int *idims, int num, int tag, void *manager){
    Manager *pManager = (Manager *)manager;
    memory::dims dims1;
    for(int i=0;i<num;i++) {
        dims1.push_back(idims[i]);
    }
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
            return -1;
    }
    CreateVariable(dims1, memory::data_type::f32, tag1,
            pManager->newNodeList, pManager->variableList);
    pManager->variableTagList.push_back(tag1);
    return pManager->getClientNode();
}

int CReverseOp(int node1, void *manager) {
    Manager *pManager = (Manager *)manager;
    CReverseOp(pManager->newNodeList.at(node1), pManager->newNodeList);
    return pManager->getClientNode();
}

int CReluOp(int node1, void *manager) {
    Manager *pManager = (Manager *)manager;
    CReluOp(pManager->newNodeList.at(node1), pManager->newNodeList);
    return pManager->getClientNode();
}

int CBNOp(int node1, void *manager) {
    Manager *pManager = (Manager *)manager;
    CBNOp(pManager->newNodeList.at(node1), pManager->newNodeList);
    return pManager->getClientNode();
}

int CBNReluOp(int node1, void *manager) {
    Manager *pManager = (Manager *)manager;
    CBNReluOp(pManager->newNodeList.at(node1), pManager->newNodeList);
    return pManager->getClientNode();
}

int CDropoutOp(int node1, float keep_prob, void *manager) {
    Manager *pManager = (Manager *)manager;
    CDropoutOp(pManager->newNodeList.at(node1), keep_prob,
            pManager->newNodeList, pManager->dropoutNodeList, pManager->dropoutKeepProb_map);
    return pManager->getClientNode();
}

int CDropoutIOp(int node1, float keep_prob, void *manager) {
    Manager *pManager = (Manager *)manager;
    CDropoutOp_Inference(pManager->newNodeList.at(node1), keep_prob, pManager->newNodeList);
    return pManager->getClientNode();
}

int CAddOp(int node1, int node2, void *manager) {
    Manager *pManager = (Manager *)manager;
    CAddOp(pManager->newNodeList.at(node1), pManager->newNodeList.at(node2), pManager->newNodeList);
    return pManager->getClientNode();
}

int CSubOp(int node1, int node2, void *manager) {
    Manager *pManager = (Manager *)manager;
    CSubOp(pManager->newNodeList.at(node1), pManager->newNodeList.at(node2), pManager->newNodeList);
    return pManager->getClientNode();
}

int CMatMulOp(int node1, int node2, int node3, void *manager) {
    Manager *pManager = (Manager *)manager;
    CMatMulOp(pManager->newNodeList.at(node1), pManager->newNodeList.at(node2), pManager->newNodeList.at(node3),
              pManager->newNodeList);
    return pManager->getClientNode();
}

int CConvolutionOp(int node1, int node2, int node3, int *strides_dims,
                   int *padding_dims_l, int *padding_dims_r, void *manager) {
    Manager *pManager = (Manager *)manager;
    memory::dims strides_dims1;
    memory::dims padding_dims_l1;
    memory::dims padding_dims_r1;
    for(int i=0;i<2;i++) {
        strides_dims1.push_back(strides_dims[i]);
        padding_dims_l1.push_back(padding_dims_l[i]);
        padding_dims_r1.push_back(padding_dims_r[i]);
    }
    CConvolutionOp(pManager->newNodeList.at(node1), pManager->newNodeList.at(node2), pManager->newNodeList.at(node3),
                   strides_dims1, padding_dims_l1, padding_dims_r1, pManager->newNodeList);
    return pManager->getClientNode();
}

int CPoolingOp(int node1, int *strides_dims, int *kernel, int *padding_dims_l,
               int *padding_dims_r, void *manager) {
    Manager *pManager = (Manager *)manager;
    memory::dims strides_dims1;
    memory::dims kernel1;
    memory::dims padding_dims_l1;
    memory::dims padding_dims_r1;
    for(int i=0;i<2;i++) {
        strides_dims1.push_back(strides_dims[i]);
        kernel1.push_back(kernel[i]);
        padding_dims_l1.push_back(padding_dims_l[i]);
        padding_dims_r1.push_back(padding_dims_r[i]);
    }
    CPoolingOp(pManager->newNodeList.at(node1), strides_dims1, kernel1, padding_dims_l1, padding_dims_r1,
            pManager->newNodeList);
    return pManager->getClientNode();
}

int CFlatOp(int node1, void *manager) {
    Manager *pManager = (Manager *)manager;
    CFlatOp(pManager->newNodeList.at(node1), pManager->newNodeList);
    return pManager->getClientNode();
}










