//
// Created by hh on 2022/2/23.
//

#include "ManagerClient.h"
#include "Manager.h"

void *CreateManager() {
    return new Manager;
}

void DelManager(void *manager) {
    Manager *pManager = (Manager *)manager;
    delete pManager;
}

void SoftmaxCrossEntropy_Training_Client(int x, int y, float lr, int gd, int num, void *manager) {
    Manager *pManager = (Manager *)manager;
    Gd agd;
    switch (agd) {
        case 0:
            agd = gd_sgd;
            break;
        case 1:
            agd = gd_adam;
            break;
        default:
            return;
    }
    pManager->softmaxCrossEntropy_Training(pManager->newNodeList.at(x), pManager->newNodeList.at(y), lr, agd, num);
}

void SoftmaxCrossEntropy_Inference_Client(int x, int y, int num, void *manager) {
    Manager *pManager = (Manager *)manager;
    pManager->softmaxCrossEntropy_Inference(pManager->newNodeList.at(x), pManager->newNodeList.at(y), num);
}

void SoftmaxCrossEntropy_Accuracy_Client(int x, int y, int num, void *manager) {
    Manager *pManager = (Manager *)manager;
    pManager->softmaxCrossEntropy_Accuracy(pManager->newNodeList.at(x), pManager->newNodeList.at(y), num);
}

void SoftmaxCrossEntropy_Training_Accuracy_Client(int x, int y, float lr, int gd, int num, void *manager) {
    Manager *pManager = (Manager *)manager;
    Gd agd;
    switch (gd) {
        case 0:
            agd = gd_sgd;
            break;
        case 1:
            agd = gd_adam;
            break;
        default:
            return;
    }
    pManager->softmaxCrossEntropy_Training_Accuracy(pManager->newNodeList.at(x), pManager->newNodeList.at(y), lr, agd, num);
}

void Predict_Client(int x, int num, void *manager){
    Manager *pManager = (Manager *)manager;
    pManager->predict(pManager->newNodeList.at(x), num);
}

void Mse_Training_Client(int x, int y, float lr, int gd, int num, void *manager) {
    Manager *pManager = (Manager *)manager;
    Gd agd;
    switch (gd) {
        case 0:
            agd = gd_sgd;
            break;
        case 1:
            agd = gd_adam;
            break;
        default:
            return;
    }
    pManager->mse_Training(pManager->newNodeList.at(x), pManager->newNodeList.at(y), lr, agd, num);
}

void Mse_Training_Inference_Client(int x, int y, float lr, int gd, int num, void *manager) {
    Manager *pManager = (Manager *)manager;
    Gd agd;
    switch (gd) {
        case 0:
            agd = gd_sgd;
            break;
        case 1:
            agd = gd_adam;
            break;
        default:
            return;
    }
    pManager->mse_Training_Inference(pManager->newNodeList.at(x), pManager->newNodeList.at(y), lr, agd, num);
}

void Inference_Client(int loss, int num, void *manager) {
    Manager *pManager = (Manager *)manager;
    pManager->inference(pManager->newNodeList.at(loss), num);
}

void ReadtoNodeScalar_Client(int data, int node, int executeId, int dataId, void *manager) {
    Manager *pManager = (Manager *)manager;
    pManager->readtoNodeScalar(data, pManager->newNodeList.at(node), executeId, dataId);
}

void ReadtoNodeReal_Client(float data, int node, int executeId, int dataId, void *manager) {
    Manager *pManager = (Manager *)manager;
    pManager->readtoNodeReal(data, pManager->newNodeList.at(node), executeId, dataId);
}

void ReadtoNodeArr_Client(float *data, int node, int executeId, int dataId, void *manager) {
    Manager *pManager = (Manager *)manager;
    pManager->readtoNodeArr(data, pManager->newNodeList.at(node), executeId, dataId);
}

void InitNodeZero_Client(int node, int executeId, void *manager) {
    Manager *pManager = (Manager *)manager;
    pManager->initNodeZero(pManager->newNodeList.at(node), pManager->executeList.at(executeId));
}

void InitVariables_Client(void *manager) {
    Manager *pManager = (Manager *)manager;
    pManager->initVariables();
}

void PrepareMemory_Client(void *manager) {
    Manager *pManager = (Manager *)manager;
    pManager->prepareMemory();
}

void Compute_Client(void *manager) {
    Manager *pManager = (Manager *)manager;
    pManager->compute();
}

void ComputeEX_Client(void *manager) {
    Manager *pManager = (Manager *)manager;
    pManager->computeEX();
}

void ComputeAccuracy_Client(void *manager) {
    Manager *pManager = (Manager *)manager;
    pManager->computeAccuracy();
}

void Wait_Client(void *manager) {
    Manager *pManager = (Manager *)manager;
    pManager->wait();
}

float GetLoss_Client(void *manager) {
    Manager *pManager = (Manager *)manager;
    return pManager->getLoss();
}

float GetAccuracy_Client(void *manager) {
    Manager *pManager = (Manager *)manager;
    return pManager->getAccuracy();
}

int GetPredictN_Client(void *manager) {
    Manager *pManager = (Manager *)manager;
    return pManager->getPredictN();
}

int GetPredictLabel_Client(int dataId, void *manager) {
    Manager *pManager = (Manager *)manager;
    return pManager->getPredictLabel(dataId);
}

void UpdateSgdLr_Client(float new_lr, void *manager) {
    Manager *pManager = (Manager *)manager;
    pManager->updateSgdLr(new_lr);
}




