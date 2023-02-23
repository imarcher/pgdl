//
// Created by hh on 2022/3/9.
//

#ifndef PGDL_MODELCLIENT_H
#define PGDL_MODELCLIENT_H



int GetVariableListSize_Client(void *manager);


int GetVariableTag_Client(int nodeId, void *manager);

int GetVariableNdim_Client(int nodeId, void *manager);

int GetVariableSize_Client(int nodeId, void *manager);

//要free
int *GetVariableDims_Client(int nodeId, void *manager);

float *GetVariableData_Client(int nodeId, void *manager);

void SetVariablePG_Client(int nodeId, void *dst, void *manager);

void SetVariableData_Client(int nodeId, int tag, int ndim, int *dims, float *data, void *manager);


#endif //PGDL_MODELCLIENT_H
