//
// Created by hh on 2021/12/19.
//


#include <stdio.h>
#include "postgres.h"
#include "fmgr.h"
#include <utils/builtins.h>
#include "executor/spi.h"
#include "Wrapper.h"
#include "string.h"


PG_MODULE_MAGIC;


ArrayType *
new_intArrayType(int num)
{
    ArrayType  *r;
    int			nbytes;

    if (num <= 0)
    {
        Assert(0);
        return r;
    }

    nbytes = ARR_OVERHEAD_NONULLS(1) + sizeof(int) * num;

    r = (ArrayType *) palloc0(nbytes);

    SET_VARSIZE(r, nbytes);
    ARR_NDIM(r) = 1;
    r->dataoffset = 0;			/* marker for no null bitmap */
    ARR_ELEMTYPE(r) = 23;
    ARR_DIMS(r)[0] = num;
    ARR_LBOUND(r)[0] = 1;

    return r;
}

ArrayType *
new_floatArrayType(int num)
{
    ArrayType  *r;
    int			nbytes;

    if (num <= 0)
    {
        Assert(0);
        return r;
    }

    nbytes = ARR_OVERHEAD_NONULLS(1) + sizeof(float) * num;

    r = (ArrayType *) palloc0(nbytes);

    SET_VARSIZE(r, nbytes);
    ARR_NDIM(r) = 1;
    r->dataoffset = 0;			/* marker for no null bitmap */
    ARR_ELEMTYPE(r) = 700;
    ARR_DIMS(r)[0] = num;
    ARR_LBOUND(r)[0] = 1;

    return r;
}



static void *manager = NULL;


//CreateNodeClient start

PG_FUNCTION_INFO_V1(create_placeholder);
Datum
create_placeholder(PG_FUNCTION_ARGS)
{

    if (manager==NULL) ereport(ERROR,(errmsg("manager is null")));

    ArrayType * idimsArr = PG_GETARG_ARRAYTYPE_P_COPY(0);
    int *idims = (int *)ARR_DATA_PTR(idimsArr);
    int num = PG_GETARG_INT32(1);
    int tag = PG_GETARG_INT32(2);

    int res = CreatePlaceholder_wrapper(idims, num, tag, manager);


    //正确
    PG_RETURN_INT32(res);
}

PG_FUNCTION_INFO_V1(create_variable);
Datum
create_variable(PG_FUNCTION_ARGS)
{

    if (manager==NULL) ereport(ERROR,(errmsg("manager is null")));

    ArrayType * idimsArr = PG_GETARG_ARRAYTYPE_P_COPY(0);
    int *idims = (int *)ARR_DATA_PTR(idimsArr);
    int num = PG_GETARG_INT32(1);
    int tag = PG_GETARG_INT32(2);

    int res = CreateVariable_wrapper(idims, num, tag, manager);

    //正确
    PG_RETURN_INT32(res);
}

PG_FUNCTION_INFO_V1(c_reverse_op);
Datum
c_reverse_op(PG_FUNCTION_ARGS)
{
    if (manager==NULL) ereport(ERROR,(errmsg("manager is null")));

    int node1 = PG_GETARG_INT32(0);

    int res = CReverseOp_wrapper(node1, manager);

    //正确
    PG_RETURN_INT32(res);
}

PG_FUNCTION_INFO_V1(c_relu_op);
Datum
c_relu_op(PG_FUNCTION_ARGS)
{
    if (manager==NULL) ereport(ERROR,(errmsg("manager is null")));

    int node1 = PG_GETARG_INT32(0);

    int res = CReluOp_wrapper(node1, manager);

    //正确
    PG_RETURN_INT32(res);
}

PG_FUNCTION_INFO_V1(c_bn_op);
Datum
c_bn_op(PG_FUNCTION_ARGS)
{
    if (manager==NULL) ereport(ERROR,(errmsg("manager is null")));

    int node1 = PG_GETARG_INT32(0);

    int res = CBNOp_wrapper(node1, manager);

    //正确
    PG_RETURN_INT32(res);
}

PG_FUNCTION_INFO_V1(c_bn_relu_op);
Datum
c_bn_relu_op(PG_FUNCTION_ARGS)
{
    if (manager==NULL) ereport(ERROR,(errmsg("manager is null")));

    int node1 = PG_GETARG_INT32(0);

    int res = CBNReluOp_wrapper(node1, manager);

    //正确
    PG_RETURN_INT32(res);
}

PG_FUNCTION_INFO_V1(c_dropout_op);
Datum
c_dropout_op(PG_FUNCTION_ARGS)
{
    if (manager==NULL) ereport(ERROR,(errmsg("manager is null")));

    int node1 = PG_GETARG_INT32(0);
    float keep_prob = (float)PG_GETARG_FLOAT8(1);

    int res = CDropoutOp_wrapper(node1, keep_prob, manager);

    //正确
    PG_RETURN_INT32(res);
}

PG_FUNCTION_INFO_V1(c_dropout_i_op);
Datum
c_dropout_i_op(PG_FUNCTION_ARGS)
{
    if (manager==NULL) ereport(ERROR,(errmsg("manager is null")));

    int node1 = PG_GETARG_INT32(0);
    float keep_prob = (float)PG_GETARG_FLOAT8(1);

    int res = CDropoutIOp_wrapper(node1, keep_prob, manager);

    //正确
    PG_RETURN_INT32(res);
}

PG_FUNCTION_INFO_V1(c_add_op);
Datum
c_add_op(PG_FUNCTION_ARGS)
{
    if (manager==NULL) ereport(ERROR,(errmsg("manager is null")));

    int node1 = PG_GETARG_INT32(0);
    int node2 = PG_GETARG_INT32(1);

    int res = CAddOp_wrapper(node1, node2, manager);

    //正确
    PG_RETURN_INT32(res);
}

PG_FUNCTION_INFO_V1(c_sub_op);
Datum
c_sub_op(PG_FUNCTION_ARGS)
{
    if (manager==NULL) ereport(ERROR,(errmsg("manager is null")));

    int node1 = PG_GETARG_INT32(0);
    int node2 = PG_GETARG_INT32(1);

    int res = CSubOp_wrapper(node1, node2, manager);

    //正确
    PG_RETURN_INT32(res);
}

PG_FUNCTION_INFO_V1(c_matmul_op);
Datum
c_matmul_op(PG_FUNCTION_ARGS)
{
    if (manager==NULL) ereport(ERROR,(errmsg("manager is null")));

    int node1 = PG_GETARG_INT32(0);
    int node2 = PG_GETARG_INT32(1);
    int node3 = PG_GETARG_INT32(2);
    int res = CMatMulOp_wrapper(node1, node2, node3, manager);

    //正确
    PG_RETURN_INT32(res);
}

PG_FUNCTION_INFO_V1(c_convolution_op);
Datum
c_convolution_op(PG_FUNCTION_ARGS)
{
    if (manager==NULL) ereport(ERROR,(errmsg("manager is null")));

    int node1 = PG_GETARG_INT32(0);
    int node2 = PG_GETARG_INT32(1);
    int node3 = PG_GETARG_INT32(2);
    ArrayType * strides_dimsArr = PG_GETARG_ARRAYTYPE_P_COPY(3);
    int *strides_dims = (int *)ARR_DATA_PTR(strides_dimsArr);
    ArrayType * padding_dims_lArr = PG_GETARG_ARRAYTYPE_P_COPY(4);
    int *padding_dims_l = (int *)ARR_DATA_PTR(padding_dims_lArr);
    ArrayType * padding_dims_rArr = PG_GETARG_ARRAYTYPE_P_COPY(5);
    int *padding_dims_r = (int *)ARR_DATA_PTR(padding_dims_rArr);

    int res = CConvolutionOp_wrapper(node1, node2, node3, strides_dims, padding_dims_l, padding_dims_r, manager);

    //正确
    PG_RETURN_INT32(res);
}

PG_FUNCTION_INFO_V1(c_pooling_op);
Datum
c_pooling_op(PG_FUNCTION_ARGS)
{
    if (manager==NULL) ereport(ERROR,(errmsg("manager is null")));

    int node1 = PG_GETARG_INT32(0);
    ArrayType * strides_dimsArr = PG_GETARG_ARRAYTYPE_P_COPY(1);
    int *strides_dims = (int *)ARR_DATA_PTR(strides_dimsArr);
    ArrayType * kernelArr = PG_GETARG_ARRAYTYPE_P_COPY(2);
    int *kernel = (int *)ARR_DATA_PTR(kernelArr);
    ArrayType * padding_dims_lArr = PG_GETARG_ARRAYTYPE_P_COPY(3);
    int *padding_dims_l = (int *)ARR_DATA_PTR(padding_dims_lArr);
    ArrayType * padding_dims_rArr = PG_GETARG_ARRAYTYPE_P_COPY(4);
    int *padding_dims_r = (int *)ARR_DATA_PTR(padding_dims_rArr);

    int res = CPoolingOp_wrapper(node1, strides_dims, kernel, padding_dims_l, padding_dims_r, manager);

    //正确
    PG_RETURN_INT32(res);
}

PG_FUNCTION_INFO_V1(c_flat_op);
Datum
c_flat_op(PG_FUNCTION_ARGS)
{
    if (manager==NULL) ereport(ERROR,(errmsg("manager is null")));

    int node1 = PG_GETARG_INT32(0);

    int res = CFlatOp_wrapper(node1, manager);

    //正确
    PG_RETURN_INT32(res);
}

//CreateNodeClient end

//ManagerClient start

PG_FUNCTION_INFO_V1(create_manager);
Datum
create_manager(PG_FUNCTION_ARGS)
{

    manager = CreateManager_wrapper();

    //正确
    PG_RETURN_INT32(0);
}

PG_FUNCTION_INFO_V1(del_manager);
Datum
del_manager(PG_FUNCTION_ARGS)
{

    DelManager_wrapper(manager);

    //正确
    PG_RETURN_INT32(0);
}


PG_FUNCTION_INFO_V1(softmax_cross_entropy_training);
Datum
softmax_cross_entropy_training(PG_FUNCTION_ARGS)
{

    if (manager==NULL) ereport(ERROR,(errmsg("manager is null")));

    int x = PG_GETARG_INT32(0);
    int y = PG_GETARG_INT32(1);
    float lr = (float)PG_GETARG_FLOAT8(2);
    int gd = PG_GETARG_INT32(3);
    int num = PG_GETARG_INT32(4);


    SoftmaxCrossEntropy_Training_wrapper(x, y, lr, gd, num, manager);
    int res = 0;
    //正确.
    PG_RETURN_INT32(res);
}

PG_FUNCTION_INFO_V1(softmax_cross_entropy_inference);
Datum
softmax_cross_entropy_inference(PG_FUNCTION_ARGS)
{

    if (manager==NULL) ereport(ERROR,(errmsg("manager is null")));

    int x = PG_GETARG_INT32(0);
    int y = PG_GETARG_INT32(1);
    int num = PG_GETARG_INT32(2);

    SoftmaxCrossEntropy_Inference_wrapper(x, y, num, manager);
    int res = 0;
    //正确
    PG_RETURN_INT32(res);
}

PG_FUNCTION_INFO_V1(softmax_cross_entropy_accuracy);
Datum
softmax_cross_entropy_accuracy(PG_FUNCTION_ARGS)
{
    if (manager==NULL) ereport(ERROR,(errmsg("manager is null")));

    int x = PG_GETARG_INT32(0);
    int y = PG_GETARG_INT32(1);
    int num = PG_GETARG_INT32(2);



    SoftmaxCrossEntropy_Accuracy_wrapper(x, y, num, manager);
    int res = 0;
    //正确
    PG_RETURN_INT32(res);
}

PG_FUNCTION_INFO_V1(softmax_cross_entropy_training_accuracy);
Datum
softmax_cross_entropy_training_accuracy(PG_FUNCTION_ARGS)
{
    if (manager==NULL) ereport(ERROR,(errmsg("manager is null")));

    int x = PG_GETARG_INT32(0);
    int y = PG_GETARG_INT32(1);
    float lr = (float)PG_GETARG_FLOAT8(2);
    int gd = PG_GETARG_INT32(3);
    int num = PG_GETARG_INT32(4);

    SoftmaxCrossEntropy_Training_Accuracy_wrapper(x, y, lr, gd, num, manager);
    int res = 0;
    //正确
    PG_RETURN_INT32(res);
}

PG_FUNCTION_INFO_V1(predict);
Datum
predict(PG_FUNCTION_ARGS)
{
    if (manager==NULL) ereport(ERROR,(errmsg("manager is null")));

    int x = PG_GETARG_INT32(0);
    int num = PG_GETARG_INT32(1);

    Predict_wrapper(x, num, manager);
    int res = 0;
    //正确
    PG_RETURN_INT32(res);
}

PG_FUNCTION_INFO_V1(mse_training);
Datum
mse_training(PG_FUNCTION_ARGS)
{
    if (manager==NULL) ereport(ERROR,(errmsg("manager is null")));

    int x = PG_GETARG_INT32(0);
    int y = PG_GETARG_INT32(1);
    float lr = (float)PG_GETARG_FLOAT8(2);
    int gd = PG_GETARG_INT32(3);
    int num = PG_GETARG_INT32(4);

    Mse_Training_wrapper(x, y, lr, gd, num, manager);
    int res = 0;
    //正确
    PG_RETURN_INT32(res);
}

PG_FUNCTION_INFO_V1(mse_training_inference);
Datum
mse_training_inference(PG_FUNCTION_ARGS)
{
    if (manager==NULL) ereport(ERROR,(errmsg("manager is null")));

    int x = PG_GETARG_INT32(0);
    int y = PG_GETARG_INT32(1);
    float lr = (float)PG_GETARG_FLOAT8(2);
    int gd = PG_GETARG_INT32(3);
    int num = PG_GETARG_INT32(4);

    Mse_Training_Inference_wrapper(x, y, lr, gd, num, manager);
    int res = 0;
    //正确
    PG_RETURN_INT32(res);
}

PG_FUNCTION_INFO_V1(inference);
Datum
inference(PG_FUNCTION_ARGS)
{
    if (manager==NULL) ereport(ERROR,(errmsg("manager is null")));

    int node = PG_GETARG_INT32(0);
    int num = PG_GETARG_INT32(1);


    Inference_wrapper(node, num, manager);
    int res = 0;
    //正确
    PG_RETURN_INT32(res);
}

/*
 * 读入对应execute的对应node的内存，标量，一般用于y
 */
PG_FUNCTION_INFO_V1(read_to_node_scalar);
Datum
read_to_node_scalar(PG_FUNCTION_ARGS)
{

    if (manager==NULL) ereport(ERROR,(errmsg("manager is null")));
    char sql[200];
 //   char    *sql = "select $1 from $2 where id>= $3 and id< $4";
    char *sql1 = "select ";
    char *sql2 = " from ";
    char *sql3 = " where id>= ";
    char *sql4 = " and id< ";
    char v3[10];
    char v4[10];

    char *v1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char *v2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    int sta = PG_GETARG_INT32(2);
    int n = PG_GETARG_INT32(3);
    int node = PG_GETARG_INT32(4);
    int executeId = PG_GETARG_INT32(5);

    sprintf(v3,"%d", sta);
    sprintf(v4,"%d", sta+n);
    strcpy(sql, sql1);
    strcat(sql, v1);
    strcat(sql, sql2);
    strcat(sql, v2);
    strcat(sql, sql3);
    strcat(sql, v3);
    strcat(sql, sql4);
    strcat(sql, v4);



    SPI_connect();

    int isSuccess = SPI_execute(sql, true, 0);

    if (isSuccess<0) PG_RETURN_INT32(isSuccess);

    if ((int)SPI_processed != n) PG_RETURN_INT32(-100);

    InitNodeZero_wrapper(node, executeId, manager);


    bool isnull;
    for(int i=0;i<n;i++){
        Datum val = heap_getattr(SPI_tuptable->vals[i], 1, SPI_tuptable->tupdesc, &isnull);
        int realval = DatumGetInt32(val);
        ReadtoNodeScalar_wrapper(realval, node, executeId, i, manager);
    }
    SPI_finish();

    //正确
    PG_RETURN_INT32(0);
}

/*
 * 读入对应execute的对应node的内存，标量，一般用于y
 */
PG_FUNCTION_INFO_V1(read_to_node_real);
Datum
read_to_node_real(PG_FUNCTION_ARGS)
{

    if (manager==NULL) ereport(ERROR,(errmsg("manager is null")));
    char sql[200];
    //   char    *sql = "select $1 from $2 where id>= $3 and id< $4";
    char *sql1 = "select ";
    char *sql2 = " from ";
    char *sql3 = " where id>= ";
    char *sql4 = " and id< ";
    char v3[10];
    char v4[10];

    char *v1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char *v2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    int sta = PG_GETARG_INT32(2);
    int n = PG_GETARG_INT32(3);
    int node = PG_GETARG_INT32(4);
    int executeId = PG_GETARG_INT32(5);

    sprintf(v3,"%d", sta);
    sprintf(v4,"%d", sta+n);
    strcpy(sql, sql1);
    strcat(sql, v1);
    strcat(sql, sql2);
    strcat(sql, v2);
    strcat(sql, sql3);
    strcat(sql, v3);
    strcat(sql, sql4);
    strcat(sql, v4);



    SPI_connect();

    int isSuccess = SPI_execute(sql, true, 0);

    if (isSuccess<0) PG_RETURN_INT32(isSuccess);

    if ((int)SPI_processed != n) PG_RETURN_INT32(-100);

    bool isnull;
    for(int i=0;i<n;i++){
        Datum val = heap_getattr(SPI_tuptable->vals[i], 1, SPI_tuptable->tupdesc, &isnull);
        float realval = DatumGetFloat4(val);
        ReadtoNodeReal_wrapper(realval, node, executeId, i, manager);
    }
    SPI_finish();

    //正确
    PG_RETURN_INT32(0);
}

/*
 * 读入对应execute的对应node的内存，数组，一般用于x
 */
PG_FUNCTION_INFO_V1(read_to_node_arr);
Datum
read_to_node_arr(PG_FUNCTION_ARGS)
{

    if (manager==NULL) ereport(ERROR,(errmsg("manager is null")));

    char sql[200];
    //   char    *sql = "select $1 from $2 where id>= $3 limit $4";
    char *sql1 = "select ";
    char *sql2 = " from ";
    char *sql3 = " where id>= ";
    char *sql4 = " and id< ";
    char v3[10];
    char v4[10];

    char *v1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char *v2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    int sta = PG_GETARG_INT32(2);
    int n = PG_GETARG_INT32(3);
    int node = PG_GETARG_INT32(4);
    int executeId = PG_GETARG_INT32(5);

    sprintf(v3,"%d", sta);
    sprintf(v4,"%d", sta+n);
    strcpy(sql, sql1);
    strcat(sql, v1);
    strcat(sql, sql2);
    strcat(sql, v2);
    strcat(sql, sql3);
    strcat(sql, v3);
    strcat(sql, sql4);
    strcat(sql, v4);


    SPI_connect();

    int isSuccess = SPI_execute(sql, true, 0);

    if (isSuccess<0) PG_RETURN_INT32(isSuccess);

    if ((int)SPI_processed != n) PG_RETURN_INT32(-100);


    bool isnull;
    for(int i=0;i<n;i++){
        Datum val = heap_getattr(SPI_tuptable->vals[i], 1, SPI_tuptable->tupdesc, &isnull);
        float *realval = (float *)ARR_DATA_PTR(DatumGetArrayTypeP(val));
        ReadtoNodeArr_wrapper(realval, node, executeId, i, manager);
    }

    SPI_finish();

    //正确
    PG_RETURN_INT32(0);
}

PG_FUNCTION_INFO_V1(init_variables);
Datum
init_variables(PG_FUNCTION_ARGS)
{
    if (manager==NULL) ereport(ERROR,(errmsg("manager is null")));

    InitVariables_wrapper(manager);
    int res = 0;
    //正确
    PG_RETURN_INT32(res);
}

PG_FUNCTION_INFO_V1(prepare_memory);
Datum
prepare_memory(PG_FUNCTION_ARGS)
{
    if (manager==NULL) ereport(ERROR,(errmsg("manager is null")));

    PrepareMemory_wrapper(manager);
    int res = 0;
    //正确
    PG_RETURN_INT32(res);
}

PG_FUNCTION_INFO_V1(compute);
Datum
compute(PG_FUNCTION_ARGS)
{
    if (manager==NULL) ereport(ERROR,(errmsg("manager is null")));

    Compute_wrapper(manager);
    int res = 0;
    //正确
    PG_RETURN_INT32(res);
}

PG_FUNCTION_INFO_V1(compute_ex);
Datum
compute_ex(PG_FUNCTION_ARGS)
{
    if (manager==NULL) ereport(ERROR,(errmsg("manager is null")));

    ComputeEX_wrapper(manager);
    int res = 0;
    //正确
    PG_RETURN_INT32(res);
}

PG_FUNCTION_INFO_V1(compute_accuracy);
Datum
compute_accuracy(PG_FUNCTION_ARGS)
{
    if (manager==NULL) ereport(ERROR,(errmsg("manager is null")));

    ComputeAccuracy_wrapper(manager);
    int res = 0;
    //正确
    PG_RETURN_INT32(res);
}

PG_FUNCTION_INFO_V1(wait);
Datum
wait(PG_FUNCTION_ARGS)
{
    if (manager==NULL) ereport(ERROR,(errmsg("manager is null")));

    Wait_wrapper(manager);
    int res = 0;
    //正确
    PG_RETURN_INT32(res);
}

PG_FUNCTION_INFO_V1(get_loss);
Datum
get_loss(PG_FUNCTION_ARGS)
{
    if (manager==NULL) ereport(ERROR,(errmsg("manager is null")));

    float res = GetLoss_wrapper(manager);

    //正确
    PG_RETURN_FLOAT4(res);
}

PG_FUNCTION_INFO_V1(get_accuracy);
Datum
get_accuracy(PG_FUNCTION_ARGS)
{
    if (manager==NULL) ereport(ERROR,(errmsg("manager is null")));

    float res = GetAccuracy_wrapper(manager);

    //正确
    PG_RETURN_FLOAT4(res);
}

PG_FUNCTION_INFO_V1(save_predict);
Datum
save_predict(PG_FUNCTION_ARGS)
{
    if (manager==NULL) ereport(ERROR,(errmsg("manager is null")));

    char *table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));

    int isSuccess;
    SPI_connect();
    //已经有表了

    //insert
    for(int i=0;i<GetPredictN_wrapper(manager);i++){
        int lab = GetPredictLabel_wrapper(i, manager);
        char insert_sql[100];
        char *insert_sql1 = "INSERT INTO ";
        char *insert_sql2 = " (label) VALUES ($1)";
        strcpy(insert_sql, insert_sql1);
        strcat(insert_sql, table_name);
        strcat(insert_sql, insert_sql2);
        Oid argtypes[1] = {23,};
        char nulls[1] = {' ',};
        Datum   values[1];
        values[0] = Int32GetDatum(lab);

        isSuccess = SPI_execute_with_args(insert_sql, 1, argtypes, values, nulls, false, 0);
        if (isSuccess<0) {
            SPI_finish();
            PG_RETURN_INT32(isSuccess);
        }
    }



    //正确
    PG_RETURN_INT32(0);
}

PG_FUNCTION_INFO_V1(update_sgd_lr);
Datum
update_sgd_lr(PG_FUNCTION_ARGS)
{
    if (manager==NULL) ereport(ERROR,(errmsg("manager is null")));
    float lr = (float)PG_GETARG_FLOAT8(0);
    UpdateSgdLr_wrapper(lr, manager);

    //正确
    PG_RETURN_INT32(0);
}




//ManagerClient end


// ModelClient start



PG_FUNCTION_INFO_V1(save_model);
Datum
save_model(PG_FUNCTION_ARGS)
{

    if (manager==NULL) ereport(ERROR,(errmsg("manager is null")));

    char *table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));

    int isSuccess;
    SPI_connect();


    //查看表是否存在
    char exist_sql[100];
    char *exist_sql1 = "select count(*) from pg_class where relname = '";
    char *exist_sql2 = "'";
    strcpy(exist_sql, exist_sql1);
    strcat(exist_sql, table_name);
    strcat(exist_sql, exist_sql2);

    isSuccess = SPI_execute(exist_sql, true, 0);
    bool isnull;
    if (isSuccess<0) {
        SPI_finish();
        PG_RETURN_INT32(isSuccess);
    }
    Datum val = heap_getattr(SPI_tuptable->vals[0], 1, SPI_tuptable->tupdesc, &isnull);
    int isexist = DatumGetInt32(val);
    //存在直接返回
    if (isexist>0) {
        SPI_finish();
        PG_RETURN_INT32(-100);
    }


    //创建表
    char table_sql[200];
    char *table_sql1 = "CREATE TABLE ";
    char *table_sql2 = "( \n"
                 "nodeid integer PRIMARY KEY,\n"
                 "tag integer NOT NULL,\n"
                 "ndim integer NOT NULL,\n"
                 "dims integer[] NOT NULL,\n"
                 "data real[] NOT NULL);";
    strcpy(table_sql, table_sql1);
    strcat(table_sql, table_name);
    strcat(table_sql, table_sql2);

    isSuccess = SPI_execute(table_sql, false, 0);
    if (isSuccess<0) {
        SPI_finish();
        PG_RETURN_INT32(isSuccess);
    }


    //nodeid为在VariableList中的id
    for(int i=0;i<GetVariableListSize_wrapper(manager);i++){
        //insert
        int nodeid = i;
        int tag = GetVariableTag_wrapper(nodeid, manager);
        int ndim = GetVariableNdim_wrapper(nodeid, manager);
        int size = GetVariableSize_wrapper(nodeid, manager);
        int *dims = GetVariableDims_wrapper(nodeid, manager);
//        float *data = GetVariableData_wrapper(nodeid, manager);
        Datum nodeid_d = Int32GetDatum(nodeid);
        Datum tag_d = Int32GetDatum(tag);
        Datum ndim_d = Int32GetDatum(ndim);
        ArrayType * dims_arr = new_intArrayType(ndim);
        ArrayType * data_arr = new_floatArrayType(size);
        memcpy(ARR_DATA_PTR(dims_arr), dims, ndim * sizeof(int));
        SetVariablePG_wrapper(nodeid, ARR_DATA_PTR(data_arr), manager);
//        memcpy(ARR_DATA_PTR(data_arr), data, size * sizeof(float));
        Datum dims_d = PointerGetDatum(dims_arr);
        Datum data_d = PointerGetDatum(data_arr);

        char insert_sql[100];
        char *insert_sql1 = "INSERT INTO ";
        char *insert_sql2 = " VALUES ($1, $2, $3, $4, $5)";
        strcpy(insert_sql, insert_sql1);
        strcat(insert_sql, table_name);
        strcat(insert_sql, insert_sql2);

        Oid argtypes[5] = {23, 23, 23, 1007, 1021};
        char nulls[5] = {' ', ' ', ' ', ' ', ' '};
        Datum   values[5];
        values[0] = nodeid_d;
        values[1] = tag_d;
        values[2] = ndim_d;
        values[3] = dims_d;
        values[4] = data_d;

        isSuccess = SPI_execute_with_args(insert_sql, 5, argtypes, values, nulls, false, 0);
        if (isSuccess<0) {
            SPI_finish();
            PG_RETURN_INT32(isSuccess);
        }
        free(dims);
    }



    SPI_finish();

    //正确
    PG_RETURN_INT32(0);
}


PG_FUNCTION_INFO_V1(load_model);
Datum
load_model(PG_FUNCTION_ARGS)
{
    if (manager==NULL) ereport(ERROR,(errmsg("manager is null")));

    char *table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));

    int isSuccess;
    SPI_connect();

    // select 表
    char sql[100];
    char *sql1 = "select tag, ndim, dims, data from ";
    char *sql2 = " where nodeid = $1";
    strcpy(sql, sql1);
    strcat(sql, table_name);
    strcat(sql, sql2);
    //一个变量一个变量读入
    bool isnull;
    for(int i=0;i<GetVariableListSize_wrapper(manager);i++){

        Oid argtypes[1] = {23,};
        char nulls[1] = {' ',};
        Datum   values[1];
        values[0] = Int32GetDatum(i);

        isSuccess = SPI_execute_with_args(sql, 1, argtypes, values, nulls, true, 0);
        if (isSuccess<0) {
            SPI_finish();
            PG_RETURN_INT32(isSuccess);
        }
        if ((int)SPI_processed == 0) {
            SPI_finish();
            PG_RETURN_INT32(-100);
        }
        //拿出元组
        Datum tag_d = heap_getattr(SPI_tuptable->vals[0], 1, SPI_tuptable->tupdesc, &isnull);
        int tag = DatumGetInt32(tag_d);
        Datum ndim_d = heap_getattr(SPI_tuptable->vals[0], 2, SPI_tuptable->tupdesc, &isnull);
        int ndim = DatumGetInt32(ndim_d);
        Datum dims_d = heap_getattr(SPI_tuptable->vals[0], 3, SPI_tuptable->tupdesc, &isnull);
        int *dims = (int *)ARR_DATA_PTR(DatumGetArrayTypeP(dims_d));
        Datum data_d = heap_getattr(SPI_tuptable->vals[0], 4, SPI_tuptable->tupdesc, &isnull);
        float *data = (float *)ARR_DATA_PTR(DatumGetArrayTypeP(data_d));

        SetVariableData_wrapper(i, tag, ndim, dims, data, manager);

    }




    //正确
    PG_RETURN_INT32(0);
}




PG_FUNCTION_INFO_V1(get_oid);
Datum
get_oid(PG_FUNCTION_ARGS)
{

    int id = PG_GETARG_INT32(0);

    SPI_connect();

    int isSuccess = SPI_execute("select * from testoid", true, 0);

    if (isSuccess<0) PG_RETURN_INT32(isSuccess);
    int oid = SPI_gettypeid(SPI_tuptable->tupdesc, id);

    SPI_finish();

    //正确
    PG_RETURN_INT32(oid);
}






// ModelClient end























