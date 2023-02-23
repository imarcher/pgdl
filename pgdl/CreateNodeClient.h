/*-------------------------------------------------------------------------
 *
 * 创建node及其op的函数的封装
 * 返回node在list的排位（int）用于客户端调用node
 * 这里没有反向的node ,反向的node都是自动构造
 *
 *-------------------------------------------------------------------------
 */

#ifndef PGDL_CREATENODECLIENT_H
#define PGDL_CREATENODECLIENT_H


#include "CreateNode.h"
#include "Manager.h"
#include "Autodiff.h"

int CreatePlaceholder(int *idims, int num, int tag, void *manager);

int CreateVariable(int *idims, int num, int tag, void *manager);

int CReverseOp(int node1, void *manager);

int CReluOp(int node1, void *manager);

int CBNOp(int node1, void *manager);

int CBNReluOp(int node1, void *manager);

int CDropoutOp(int node1, float keep_prob, void *manager);

int CDropoutIOp(int node1, float keep_prob, void *manager);

int CAddOp(int node1, int node2, void *manager);

int CSubOp(int node1, int node2, void *manager);

int CMatMulOp(int node1, int node2, int node3, void *manager);

int CConvolutionOp(int node1, int node2, int node3, int *strides_dims,
                   int *padding_dims_l, int *padding_dims_r, void *manager);

int CPoolingOp(int node1, int *strides_dims, int *kernel, int *padding_dims_l,
                 int *padding_dims_r, void *manager);

int CFlatOp(int node1, void *manager);

//求交叉商准确率还没弄

#endif //PGDL_CREATENODECLIENT_H
