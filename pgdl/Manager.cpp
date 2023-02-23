//
// Created by hh on 2021/12/20.
//




#include <cstring>
#include "Manager.h"
#include "CreateNode.h"
#include <example_utils.hpp>
#include <random>
#include <chrono>
#include <thread>


void Manager::initTraining(Node *loss) {
    getGradients(loss, variableList, gNodes, newNodeList);
    // 获得导数的计算
    findTopoSort(gNodes, topoNodes);
}

void Manager::initInference(Node *loss) {
    vector<Node *> aloss;
    aloss.push_back(loss);
    // 获得导数的计算
    findTopoSort(aloss, topoNodes);
}

void Manager::initAccuracy(vector<Node *> &nodes) {
    findTopoSort(nodes, topoNodes);
}

void Manager::getSgdNode(Gd gd) {
    for(int i=0;i<engineList.size();i++){
        Graph *graph = graphList.at(i);
        engine &eng = engineList.at(i);
        vector<Node *> &new_gNodes = graph->gNodes;
        for(int i=0;i<variableList.size();i++){
            if(gd == gd_sgd) CSgdOp(new_gNodes.at(i), variableList.at(i), graph->sgdNodes, graph->sgdLrNodes);
            else if(gd == gd_adam) CAdamOp(new_gNodes.at(i), variableList.at(i), lr, graph->sgdNodes, graph->sgdLrNodes);
        }
    }
}


void Manager::replaceGNodes(Node *old_node, Node *new_node, vector<Node *> &old_gNodes) {
    for(int i=0;i<old_gNodes.size();i++){
        if(old_gNodes.at(i) == old_node){
            old_gNodes.at(i) = new_node;
            break;
        }
    }
}

void Manager::getShapes() {
    for(int i=0;i<engineList.size();i++){
        //reorder的map，原点和改后的点
        unordered_map<Node *, Node *> reorder_map;
        // backward 加了data
        unordered_set<Node *> reorder_set;

        Graph *graph = graphList.at(i);
        engine &eng = engineList.at(i);
        vector<Node *> &new_topoNodesWithoutPV = graph->topoNodesWithoutPV;

        vector<Node *> &new_gNodes = graph->gNodes;
        vector<Node *> &new_todoMemoryNodes = graph->todoMemoryNodes;
        unordered_map<Node *, primitive_desc_base> &primitive_desc_map = graph->primitive_desc_map;
        unordered_map<Node *, vector<memory::desc>> &new_workspace_desc_map = graph->workspace_desc_map;
        unordered_map<Node *, memory::desc *> &new_tagChangeMap = graph->tagChangeMap;
        for(int j=0; j < topoNodes.size(); j++) {
            Node *anode = topoNodes.at(j);
            if(anode->nodetype==Placeholder || anode->nodetype==Variable) continue;

            if(anode->nodetype==ConvolutionBackwardBias) {
                new_todoMemoryNodes.push_back(anode);
                continue;
            }

            if(anode->nodetype==Dropout1) {
                anode->desc = new memory::desc(anode->inputs.at(0)->desc->data);
                new_todoMemoryNodes.push_back(anode);
                continue;
            }

            if(anode->nodetype==Dropout_Prob) {
                memory::dims dropout_prob_dims;
                int dropout_prob_ndim = anode->inputs.at(0)->desc->data.ndims;
                for(int k=0;k<dropout_prob_ndim;k++) dropout_prob_dims.push_back(1);
                memory::format_tag dropout_prob_tag = memory::format_tag::ab;
                if(dropout_prob_ndim==4){
                    dropout_prob_tag = memory::format_tag::nchw;
                }
                anode->desc = new memory::desc(dropout_prob_dims, memory::data_type::f32, dropout_prob_tag);
                new_todoMemoryNodes.push_back(anode);
                continue;
            }

            //查看是否输入有变化
            for(Node *reorder_set_node: reorder_set){
                int mark = 1;
                for(int k=0; k < anode->inputs.size(); k++){
                    if(reorder_set_node == anode->inputs.at(k)){
                        anode->inputs.at(k) = reorder_map.at(reorder_set_node);
                        reorder_set.erase(reorder_set_node);
                        mark = 0;
                        break;
                    }
                }
                if(mark == 0) break;
            }
            
            anode->op->infer_shape(anode, eng, primitive_desc_map, new_workspace_desc_map);
            if(anode->nodetype==Convolution) {
                //卷积查看data和weights
                primitive_desc_base *pd_b = &(primitive_desc_map.at(anode));
                convolution_forward::primitive_desc *pd = (convolution_forward::primitive_desc*)pd_b;
                //data不同加reorder
                memory::desc src_md = pd->src_desc();
                if(src_md != *(anode->inputs.at(0)->desc)){
                    Node *new_node = CReOrderOp(anode->inputs.at(0), &src_md, newNodeList);
                    new_node->op->infer_shape(new_node, eng, primitive_desc_map, new_workspace_desc_map);
                    reorder_map.insert({anode->inputs.at(0), new_node});
                    anode->inputs.at(0) = new_node;
                    new_topoNodesWithoutPV.push_back(new_node);
                }
                //weights不同改weights的desc
                memory::desc weights_md = pd->weights_desc();
                Node * weights_node = anode->inputs.at(1);
                if(weights_md != *(weights_node->desc)){
                    //标记tag改变
                    new_tagChangeMap.insert({weights_node, weights_node->desc});
                    weights_node->desc = new memory::desc(weights_md);
                }
                new_topoNodesWithoutPV.push_back(anode);
            }
            else if(anode->nodetype==ConvolutionBackwardData){
                //卷积反向查看weights和导数，再看前向的data是否有改变，如果有reorder，要改回去
                primitive_desc_base *pd_b = &(primitive_desc_map.at(anode));
                convolution_backward_data::primitive_desc *pd = (convolution_backward_data::primitive_desc*)pd_b;
                //weights不同加reorder
                memory::desc weights_md = pd->weights_desc();
                Node *weights_node = anode->inputs.at(0);
                if(weights_md != *(weights_node->desc)){
                    Node * new_node = CReOrderOp(weights_node, &weights_md, newNodeList);
                    new_node->op->infer_shape(new_node, eng, primitive_desc_map, new_workspace_desc_map);
                    reorder_map.insert({weights_node, new_node});
                    anode->inputs.at(0) = new_node;
                    new_topoNodesWithoutPV.push_back(new_node);
                }
                //导数不同加reorder，先看是否有新的
                memory::desc diff_dst_md = pd->diff_dst_desc();
                Node *diff_dst_node = anode->inputs.at(1);
                if(reorder_map.count(diff_dst_node)){
                    //有新的
                    Node *re_diff_dst_node = reorder_map.at(diff_dst_node);
                    if(diff_dst_md == *(re_diff_dst_node->desc)){
                        // = 新的，直接换新的
                        anode->inputs.at(1) = re_diff_dst_node;
                    }
                    else if(diff_dst_md != *(diff_dst_node->desc)){
                        Node * new_node = CReOrderOp(diff_dst_node, &diff_dst_md, newNodeList);
                        new_node->op->infer_shape(new_node, eng, primitive_desc_map, new_workspace_desc_map);
                        anode->inputs.at(1) = new_node;
                        new_topoNodesWithoutPV.push_back(new_node);
                    }
                } else{
                    //没新的
                    if(diff_dst_md != *(diff_dst_node->desc)){
                        Node * new_node = CReOrderOp(diff_dst_node, &diff_dst_md, newNodeList);
                        new_node->op->infer_shape(new_node, eng, primitive_desc_map, new_workspace_desc_map);
                        reorder_map.insert({diff_dst_node, new_node});
                        anode->inputs.at(1) = new_node;
                        new_topoNodesWithoutPV.push_back(new_node);
                    }
                }
                new_topoNodesWithoutPV.push_back(anode);
                //查看前向是否改data
                memory::desc diff_src_md = pd->diff_src_desc();
                Node *src_node = anode->forwardNode->inputs.at(0);
                if(src_node->nodetype == ReOrder){
                    src_node = src_node->inputs.at(0);
                }
                if(diff_src_md != *(src_node->desc)){
                    Node * new_node = CReOrderOp(anode, src_node->desc, newNodeList);
                    new_node->op->infer_shape(new_node, eng, primitive_desc_map, new_workspace_desc_map);
                    reorder_map.insert({anode, new_node});
                    reorder_set.insert(anode);
                    new_topoNodesWithoutPV.push_back(new_node);
                }

            }
            else if(anode->nodetype==ConvolutionBackwardWeights){
                //卷积查看data 和导数
                primitive_desc_base *pd_b = &(primitive_desc_map.at(anode));
                convolution_backward_weights::primitive_desc *pd = (convolution_backward_weights::primitive_desc*)pd_b;
                //data不同加reorder，先看是否有新的
                memory::desc src_md = pd->src_desc();
                Node *src_node = anode->inputs.at(0);
                if(reorder_map.count(src_node)){
                    //有新的
                    Node *re_src_node = reorder_map.at(src_node);
                    if(src_md == *(re_src_node->desc)){
                        // = 新的，直接换新的
                        anode->inputs.at(0) = re_src_node;
                    }
                    else if(src_md != *(src_node->desc)){
                        Node * new_node = CReOrderOp(src_node, &src_md, newNodeList);
                        new_node->op->infer_shape(new_node, eng, primitive_desc_map, new_workspace_desc_map);
                        anode->inputs.at(0) = new_node;
                        new_topoNodesWithoutPV.push_back(new_node);
                    }
                } else{
                    //没新的
                    if(src_md != *(src_node->desc)){
                        Node * new_node = CReOrderOp(src_node, &src_md, newNodeList);
                        new_node->op->infer_shape(new_node, eng, primitive_desc_map, new_workspace_desc_map);
                        reorder_map.insert({src_node, new_node});//无所谓
                        anode->inputs.at(0) = new_node;
                        new_topoNodesWithoutPV.push_back(new_node);
                    }
                }
                //导数不同加reorder
                memory::desc diff_dst_md = pd->diff_dst_desc();
                Node *diff_dst_node = anode->inputs.at(2);
                if(reorder_map.count(diff_dst_node)){
                    //有新的
                    Node *re_diff_dst_node = reorder_map.at(diff_dst_node);
                    if(diff_dst_md == *(re_diff_dst_node->desc)){
                        // = 新的，直接换新的
                        anode->inputs.at(2) = re_diff_dst_node;
                    }
                    else if(diff_dst_md != *(diff_dst_node->desc)){
                        Node * new_node = CReOrderOp(diff_dst_node, &diff_dst_md, newNodeList);
                        new_node->op->infer_shape(new_node, eng, primitive_desc_map, new_workspace_desc_map);
                        anode->inputs.at(2) = new_node;
                        new_topoNodesWithoutPV.push_back(new_node);
                    }
                } else{
                    //没新的
                    if(diff_dst_md != *(diff_dst_node->desc)){
                        Node * new_node = CReOrderOp(diff_dst_node, &diff_dst_md, newNodeList);
                        new_node->op->infer_shape(new_node, eng, primitive_desc_map, new_workspace_desc_map);
                        reorder_map.insert({diff_dst_node, new_node});
                        anode->inputs.at(2) = new_node;
                        new_topoNodesWithoutPV.push_back(new_node);
                    }
                }
                new_topoNodesWithoutPV.push_back(anode);
                //查看diff_weights是否和前向weights一致
                memory::desc diff_weights_md = pd->diff_weights_desc();
                Node *weights_node = anode->forwardNode->inputs.at(1);
                if(diff_weights_md != *(weights_node->desc)){
                    Node * new_node = CReOrderOp(anode, weights_node->desc, newNodeList);
                    new_node->op->infer_shape(new_node, eng, primitive_desc_map, new_workspace_desc_map);
                    replaceGNodes(anode, new_node, new_gNodes);
                    new_topoNodesWithoutPV.push_back(new_node);
                }
            }
            else{
                new_topoNodesWithoutPV.push_back(anode);
            }

        }
    }
}



void Manager::getSgdLrShapes() {
    for(int i=0;i<engineList.size();i++){
        Graph *graph = graphList.at(i);

        vector<Node *> &new_sgdLrNodes = graph->sgdLrNodes;
        for(Node *anode: new_sgdLrNodes){
            if(anode->nodetype==SgdLr || anode->nodetype==Adam_Bt1 || anode->nodetype==Adam_Bt2){
                memory::dims sgdLr_dims;
                int sgdLr_ndim = anode->inputs.at(0)->desc->data.ndims;
                for(int k=0;k<sgdLr_ndim;k++) sgdLr_dims.push_back(1);
                memory::format_tag sgdLr_tag = memory::format_tag::nchw;
                if(sgdLr_ndim==1){
                    sgdLr_tag = memory::format_tag::a;
                }
                else if(sgdLr_ndim==2){
                    sgdLr_tag = memory::format_tag::ab;
                }
                anode->desc = new memory::desc(sgdLr_dims, memory::data_type::f32, sgdLr_tag);
            }
            else {
                anode->desc = new memory::desc(anode->inputs.at(0)->desc->data);
            }
        }
    }
}

void Manager::getSgdShapes() {
    for(int i=0;i<engineList.size();i++){
        Graph *graph = graphList.at(i);
        engine &eng = engineList.at(i);

        unordered_map<Node *, primitive_desc_base> &primitive_desc_map = graph->primitive_desc_map;
        unordered_map<Node *, vector<memory::desc>> &new_workspace_desc_map = graph->workspace_desc_map;
        vector<Node *> &new_sgdNodes = graph->sgdNodes;
        for(Node *anode: new_sgdNodes){
            anode->op->infer_shape(anode, eng, primitive_desc_map, new_workspace_desc_map);
        }
    }
}

void Manager::getExtraShapes() {
    for(int i=0;i<engineList.size();i++){
        Graph *graph = graphList.at(i);
        engine &eng = engineList.at(i);

        unordered_map<Node *, primitive_desc_base> &primitive_desc_map = graph->primitive_desc_map;
        unordered_map<Node *, vector<memory::desc>> &new_workspace_desc_map = graph->workspace_desc_map;
        for(Node *anode: extraNodeList){
            anode->op->infer_shape(anode, eng, primitive_desc_map, new_workspace_desc_map);
        }
    }
}



void Manager::getExecutes(int num) {
    //这里只用一个graph
    Graph * agraph = graphList.at(0);
    for(int i=0;i<num;i++) executeList.push_back(new Execute(engineList.at(0), agraph->topoNodesWithoutPV,
            agraph->sgdNodes, agraph->sgdLrNodes, agraph->todoMemoryNodes, agraph->primitive_desc_map,
            agraph->workspace_desc_map, agraph->xNodeId,extraNodeList, placeholderList,
            variableList, dropoutKeepProb_map, lr));
    executeSize = num;
}

float Manager::getNodeValue(Node *anode, int id) {
    Execute *aexecute = executeList.at(id);
    float *res = (float *)(aexecute->memory_map.at(anode).get_data_handle());
    return res[0];
}

float Manager::getloss(int id) {
    return getNodeValue(loss, id);
}

float Manager::getaccuracy(int id) {
    return getNodeValue(accuracy, id);
}

void Manager::getGraphEngines() {
    graphList.push_back(new Graph(gNodes));
    engineList.emplace_back(engine::kind::cpu, 0);
}

void Manager::getXNodeId() {
    for(Graph *agraph: graphList){
        vector<Node *> &new_topoNodesWithoutPV = agraph->topoNodesWithoutPV;
        for(int i=0;i<new_topoNodesWithoutPV.size();i++){
            if(new_topoNodesWithoutPV.at(i) == xNode) {
                agraph->xNodeId = i;
                break;
            }
        }
    }
}

void Manager::setVariableData(Node *node, memory::desc &adesc, float *data, Execute *aexecute) {
    memory data_memory;
    memory now_memory = aexecute->memory_map.at(node);


    if(now_memory.get_desc() != adesc){
        //格式不等于
        data_memory = memory(adesc, aexecute->eng);
        write_to_dnnl_memory(data, data_memory);
        reorder re(data_memory, now_memory);
        re.execute(aexecute->astream, {{DNNL_ARG_FROM,data_memory},{DNNL_ARG_TO,now_memory}});
    }else{
        write_to_dnnl_memory(data, now_memory);
    }
}

void Manager::iniAdamBt() {
    for(Execute *aexecute: executeList){
        aexecute->iniAdamBt();
    }
}




//public


Manager::Manager() {
    lr = 0;
    gd = gd_sgd;
}

Manager::~Manager() {
//    deleteNewNodeList();
//    deleteSgdNodeList();
}

void Manager::createNewNodeList(){
    newNodeList.clear();
}

void Manager::deleteNewNodeList(){
    for(Node *anode : newNodeList){
        delete anode;
    }
    newNodeList.clear();
}

void Manager::deleteSgdNodeList(){
    for(Node *anode : sgdNodeList){
        delete anode;
    }
    sgdNodeList.clear();
}

void Manager::createPlaceholderList() {
    placeholderList.clear();
}

void Manager::createVariableList(){
    variableList.clear();
}

int Manager::getClientNode() {
    return newNodeList.size()-1;
}





void Manager::softmaxCrossEntropy_Training(Node *x, Node *y, float lr, Gd gd, int num) {
    this->gd = gd;
    Node *aloss = SoftmaxCrossEntropy_Training(x, y, newNodeList);
    this->lr = lr;
    initTraining(aloss);
    getGraphEngines();
    getShapes();
    getSgdNode(gd);
    getSgdLrShapes();
    getSgdShapes();
    getExecutes(num);

}

void Manager::softmaxCrossEntropy_Inference(Node *x, Node *y, int num) {
    Node *aloss = SoftmaxCrossEntropy_Inference(x, y, newNodeList);
    this->loss = aloss;
    initInference(aloss);
    getGraphEngines();
    getShapes();
    getExecutes(num);

}

void Manager::softmaxCrossEntropy_Accuracy(Node *x, Node *y, int num) {
    vector<Node *> resNodes;
    SoftmaxCrossEntropy_Accuracy(x, y, resNodes, newNodeList);
    this->loss = resNodes.at(0);
    this->accuracy = resNodes.at(1);
    initAccuracy(resNodes);
    getGraphEngines();
    getShapes();
    getExecutes(num);

}

void Manager::softmaxCrossEntropy_Training_Accuracy(Node *x, Node *y, float lr, Gd gd, int num) {
    this->gd = gd;
    xNode = x;
    this->lr = lr;
    Node *aloss = SoftmaxCrossEntropy_Training(x, y, newNodeList);
    initTraining(aloss);
    vector<Node *> resNodes;
    SoftmaxCrossEntropy_Accuracy(x, y, resNodes, extraNodeList);
    this->loss = resNodes.at(0);
    this->accuracy = resNodes.at(1);
    getGraphEngines();
    getShapes();
    getSgdNode(gd);
    getSgdLrShapes();
    getSgdShapes();
    getExtraShapes();
    getXNodeId();
    getExecutes(num);

}

void Manager::predict(Node *x, int num) {
    predictNode = CPredictOp(x, newNodeList);
    initInference(predictNode);
    getGraphEngines();
    getShapes();
    getExecutes(num);
}

void Manager::mse_Training(Node *x, Node *y, float lr, Gd gd, int num) {
    this->gd = gd;
    this->lr = lr;
    Node *aloss = Mse_Training(x, y, newNodeList);
    initTraining(aloss);
    getGraphEngines();
    getShapes();
    getSgdNode(gd);
    getSgdLrShapes();
    getSgdShapes();
    getExecutes(num);
}

void Manager::mse_Training_Inference(Node *x, Node *y, float lr, Gd gd, int num) {
    this->gd = gd;
    this->lr = lr;
    Node *aloss = Mse_Training_I(x, y, newNodeList);
    xNode = aloss;
    initTraining(aloss);
    this->loss = Mse_Inference(aloss, extraNodeList);
    getGraphEngines();
    getShapes();
    getSgdNode(gd);
    getSgdLrShapes();
    getSgdShapes();
    getExtraShapes();
    getXNodeId();
    getExecutes(num);
}

void Manager::inference(Node *x, int num) {
    this->loss = x;
    initInference(x);
    getGraphEngines();
    getShapes();
    getExecutes(num);
}

void Manager::prepareMemory() {
    for(Execute *aexecute: executeList){
        aexecute->getMemory();
        aexecute->getPrimitive();
        aexecute->getMemoryArgs();
        aexecute->setStream();
    }
}

void Manager::compute() {
    initDropout();
    if(gd == gd_adam) iniAdamBt();
    for(Execute *aexecute: executeList){
        aexecute->compute();
    }
    wait();
}

void Manager::computeEX() {


    initDropout();
    if(gd == gd_adam) {
        iniAdamBt();
    }
    for(Execute *aexecute: executeList){
        aexecute->computeEX();
    }
}

void Manager::computeAccuracy() {
    initDropoutZero();
    for(Execute *aexecute: executeList){
        aexecute->computeAccuracy();
    }
    wait();
}


void Manager::wait() {
    for(Execute *aexecute: executeList){
        aexecute->wait();
    }
}



float Manager::getLoss() {
    float res = 0;
    for(int i=0;i<executeSize;i++){
        res += getloss(i);
    }
    return res/executeSize;
}

float Manager::getAccuracy() {
    float res = 0;
    for(int i=0;i<executeSize;i++){
        res += getaccuracy(i);
    }
    return res/executeSize;
}

void Manager::readtoNodeScalar(int data, Node *node, int executeId, int dataId) {
    memory node_mem = executeList.at(executeId)->memory_map.at(node);
    float *dst = (float *)node_mem.get_data_handle();
    if(!memorySize_map.count(node)){
        memorySize_map.insert({node,(int)(node_mem.get_desc().get_size() / node_mem.get_desc().dims().at(0) / 4)});
    }
    int baseSize = memorySize_map.at(node);
    int baseid = dataId*baseSize;
    dst[baseid+data] = 1;
}

void Manager::readtoNodeReal(float data, Node * node, int executeId, int dataId) {
    memory node_mem = executeList.at(executeId)->memory_map.at(node);
    float *dst = (float *)node_mem.get_data_handle();
    dst[dataId] = data;
}

void Manager::readtoNodeArr(float *data, Node *node, int executeId, int dataId) {
    memory node_mem = executeList.at(executeId)->memory_map.at(node);
    float *dst = (float *)node_mem.get_data_handle();
    if(!memorySize_map.count(node)){
        memorySize_map.insert({node,(int)(node_mem.get_desc().get_size() / node_mem.get_desc().dims().at(0) / 4)});
    }
    int baseSize = memorySize_map.at(node);
    int baseid = dataId*baseSize;
    mempcpy(dst+baseid, data, baseSize * 4);
//    for(int i=0;i<baseSize;i++) {
//        dst[baseid+i] = data[i];
//    }
}

void Manager::initVariables() {
    for(Node * anode: variableList) {
        for(Execute * aexecute: executeList) {
            initNodeNormal(anode, aexecute);
        }
    }
}

void Manager::initDropout() {
    for(Node * anode: dropoutNodeList) {
        for(Execute * aexecute: executeList) {
            initNodeProb(anode, aexecute);
        }
    }
}

void Manager::initDropoutZero() {
    for(Node * anode: dropoutNodeList) {
        for(Execute * aexecute: executeList) {
            initNodeZero(anode, aexecute);
        }
    }
}

void Manager::initNodeZero(Node *node, Execute *aexecute) {
    size_t anodesize = node->desc->get_size();
    memset(aexecute->memory_map.at(node).get_data_handle(), 0, anodesize);
}

void Manager::initNodeNormal(Node *node, Execute *aexecute) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    std::normal_distribution<float> dis(0,1);
    int anodesize = node->desc->get_size() / 4;
    float *data = (float *)(aexecute->memory_map.at(node).get_data_handle());
    for(int i=0;i<anodesize;i++){
        data[i] = dis(gen);
    }
}

void Manager::initNodeProb(Node *node, Execute * aexecute) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    std::uniform_real_distribution<float> dis(0,1);
    int anodesize = node->desc->get_size() / 4;
    float *data = (float *)(aexecute->memory_map.at(node).get_data_handle());
    for(int i=0;i<anodesize;i++){
        data[i] = dis(gen);
    }
}

int Manager::getVariableListSize() {
    return variableList.size();
}

float *Manager::getVariableData(Node *node) {
    //第一个execute的参数
    Execute * aexecute = executeList.at(0);
    Graph * agraph = graphList.at(0);
    memory now_memory = aexecute->memory_map.at(node);
    //查看是否变化过
    if(agraph->tagChangeMap.count(node)){
        //变化过
        memory new_memory = memory(*(agraph->tagChangeMap.at(node)), aexecute->eng);
        reorder re(now_memory, new_memory);
        re.execute(aexecute->astream, {{DNNL_ARG_FROM,now_memory},{DNNL_ARG_TO,new_memory}});

        float *d = (float *)(new_memory.get_data_handle());
        float dd = d[0];
        return d;
    }else{
        float *d = (float *)(now_memory.get_data_handle());
        float dd = d[0];
        return d;
    }
}

void Manager::setVariablePG(Node *node, void *dst) {
    //第一个execute的参数
    Execute * aexecute = executeList.at(0);
    Graph * agraph = graphList.at(0);
    memory now_memory = aexecute->memory_map.at(node);
    //查看是否变化过
    if(agraph->tagChangeMap.count(node)){
        //变化过
        memory new_memory = memory(*(agraph->tagChangeMap.at(node)), aexecute->eng);
        reorder re(now_memory, new_memory);
        re.execute(aexecute->astream, {{DNNL_ARG_FROM,now_memory},{DNNL_ARG_TO,new_memory}});
        float *d = (float *)(new_memory.get_data_handle());
        memcpy(dst, d, new_memory.get_desc().get_size());

    }else{
        float *d = (float *)(now_memory.get_data_handle());
        memcpy(dst, d, now_memory.get_desc().get_size());
    }
}

void Manager::setVariableData(Node *node, memory::dims &adims, memory::format_tag atag, float *data) {

    memory::desc adesc(adims, memory::data_type::f32, atag);

    for(Execute * aexecute: executeList) {
        setVariableData(node, adesc, data, aexecute);
    }
}

int Manager::getPredictN() {
    return predictNode->desc->dims().at(0);
}

int Manager::getPredictLabel(int dataId) {
    int labelNum = predictNode->desc->dims().at(1);
    float *labelData = (float *)(executeList.at(0)->memory_map.at(predictNode).get_data_handle());
    for(int i=0;i<labelNum;i++){
        if(labelData[dataId*labelNum+i]>0.5) return i;
    }
    return 0;
}

void Manager::updateSgdLr(float new_lr) {
    lr = new_lr;
    for(Execute *aexecute: executeList){
        aexecute->updateSgdLr(lr);
    }
}












