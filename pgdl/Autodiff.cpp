//
// Created by hh on 2022/1/9.
//

#include <assert.h>
#include <iostream>
#include "Autodiff.h"
#include "CreateNode.h"


Node::Node(Nodetype nodetype): nodetype(nodetype) {}

Node::Node(Nodetype nodetype, Op *op): nodetype(nodetype), op(op){}

Node::~Node() {
    delete desc;
    delete op;
}

Op::~Op() = default;

LinearOp::LinearOp(Node *node1, float a, float b, Node *node) {
    node->inputs.push_back(node1);
    this->a = a;
    this->b = b;
}

LinearOp::~LinearOp() {

}

void LinearOp::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                       unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_SRC, input_memory.at(0)});
    args.insert({DNNL_ARG_DST, input_memory.at(1)});
}

primitive
LinearOp::getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    eltwise_forward::primitive_desc *b = (eltwise_forward::primitive_desc*)a;
    return eltwise_forward(*b);
}

void LinearOp::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    //没做
    assert(0);
}

void LinearOp::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
                           unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    node->desc = new memory::desc(node->inputs.at(0)->desc->data);
    eltwise_forward::primitive_desc new_primitive_desc(
            {prop_kind::forward_inference, algorithm::eltwise_linear, *(node->desc), a, b}, eng);
    primitive_desc_map.insert({node, new_primitive_desc});
}


ReverseOp::ReverseOp(Node *node1, Node *node) {
    node->inputs.push_back(node1);
}

ReverseOp::~ReverseOp() {

}

void ReverseOp::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
        unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_SRC, input_memory.at(0)});
    args.insert({DNNL_ARG_DST, input_memory.at(1)});
}

primitive ReverseOp::getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    eltwise_forward::primitive_desc *b = (eltwise_forward::primitive_desc*)a;
    return eltwise_forward(*b);
}

void ReverseOp::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    Node *new_grad_node = CReverseBackwardOp(output_grad, newNodeList);
    new_grad_node->forwardNode = node;
    inputs_grad.push_back(new_grad_node);
}

void ReverseOp::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
        unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    node->desc = new memory::desc(node->inputs.at(0)->desc->data);
    eltwise_forward::primitive_desc new_primitive_desc(
            {prop_kind::forward_training, algorithm::eltwise_linear, *(node->desc), -1.}, eng);
    primitive_desc_map.insert({node, new_primitive_desc});
}


ReverseBackwardOp::ReverseBackwardOp(Node *node1, Node *node) {
    node->inputs.push_back(node1);
}

ReverseBackwardOp::~ReverseBackwardOp() {

}

void ReverseBackwardOp::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
        unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_DIFF_DST, input_memory.at(0)});
    args.insert({DNNL_ARG_DIFF_SRC, input_memory.at(1)});
}

primitive ReverseBackwardOp::getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    eltwise_backward::primitive_desc *b = (eltwise_backward::primitive_desc*)a;
    return eltwise_backward(*b);
}

void ReverseBackwardOp::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    assert(0);
}

void ReverseBackwardOp::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
        unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    node->desc = new memory::desc(node->inputs.at(0)->desc->data);
    primitive_desc_base *a = &(primitive_desc_map.at(node->forwardNode));
    eltwise_forward::primitive_desc *b = (eltwise_forward::primitive_desc*)a;
    eltwise_backward::primitive_desc new_primitive_desc(
            {algorithm::eltwise_linear, *(node->desc), *(node->desc), -1.}, eng, *b);
    primitive_desc_map.insert({node, new_primitive_desc});
}

ReluOp::ReluOp(Node *node1, Node *node) {
    node->inputs.push_back(node1);
}

ReluOp::~ReluOp() {

}

void ReluOp::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
        unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_SRC, input_memory.at(0)});
    args.insert({DNNL_ARG_DST, input_memory.at(1)});
}

primitive ReluOp::getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    eltwise_forward::primitive_desc *b = (eltwise_forward::primitive_desc*)a;
    return eltwise_forward(*b);
}

void ReluOp::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    Node *new_grad_node = CReluBackwardOp(output_grad, node->inputs.at(0), newNodeList);
    new_grad_node->forwardNode = node;
    inputs_grad.push_back(new_grad_node);
}

void ReluOp::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
        unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    node->desc = new memory::desc(node->inputs.at(0)->desc->data);
    eltwise_forward::primitive_desc new_primitive_desc(
            {prop_kind::forward_training, algorithm::eltwise_relu, *(node->desc), 0}, eng);
    primitive_desc_map.insert({node, new_primitive_desc});
}

ReluBackwardOp::ReluBackwardOp(Node *node1, Node *node2, Node *node) {
    node->inputs.push_back(node1);
    node->inputs.push_back(node2);
}

ReluBackwardOp::~ReluBackwardOp() {

}

void ReluBackwardOp::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
        unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_DIFF_DST, input_memory.at(0)});
    args.insert({DNNL_ARG_SRC, input_memory.at(1)});
    args.insert({DNNL_ARG_DIFF_SRC, input_memory.at(2)});
}

primitive ReluBackwardOp::getPrimitive(Node *node, engine &engine,
                                       unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    eltwise_backward::primitive_desc *b = (eltwise_backward::primitive_desc*)a;
    return eltwise_backward(*b);
}

void ReluBackwardOp::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    assert(0);
}

void
ReluBackwardOp::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
        unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    node->desc = new memory::desc(node->inputs.at(0)->desc->data);
    primitive_desc_base *a = &(primitive_desc_map.at(node->forwardNode));
    eltwise_forward::primitive_desc *b = (eltwise_forward::primitive_desc*)a;
    eltwise_backward::primitive_desc new_primitive_desc(
            {algorithm::eltwise_relu, *(node->desc), *(node->desc), 0}, eng, *b);
    primitive_desc_map.insert({node, new_primitive_desc});
}

BNOp::BNOp(Node *node1, Node *node) {
    node->inputs.push_back(node1);
}

BNOp::~BNOp() {

}

void BNOp::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                       unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_SRC, input_memory.at(0)});
    args.insert({DNNL_ARG_DST, input_memory.at(1)});
    args.insert({DNNL_ARG_WORKSPACE, workspace_map.at(node).at(0)});
    args.insert({DNNL_ARG_MEAN, workspace_map.at(node).at(1)});
    args.insert({DNNL_ARG_VARIANCE, workspace_map.at(node).at(2)});
}

primitive
BNOp::getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    batch_normalization_forward::primitive_desc *b = (batch_normalization_forward::primitive_desc*)a;
    return batch_normalization_forward(*b);
}

void BNOp::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    Node *new_grad_node = CBNBackwardOp(output_grad, node->inputs.at(0), newNodeList);
    new_grad_node->forwardNode = node;
    inputs_grad.push_back(new_grad_node);
}

void BNOp::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
                           unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    node->desc = new memory::desc(node->inputs.at(0)->desc->data);
    batch_normalization_forward::primitive_desc new_primitive_desc(
            {prop_kind::forward_training, *(node->desc), 1.e-10f, normalization_flags::none}, eng);
    vector<memory::desc> vmd;
    vmd.push_back(new_primitive_desc.workspace_desc());
    vmd.push_back(new_primitive_desc.mean_desc());
    vmd.push_back(new_primitive_desc.variance_desc());
    workspace_desc_map.insert({node, vmd});
    primitive_desc_map.insert({node, new_primitive_desc});
}

BNBackwardOp::BNBackwardOp(Node *node1, Node *node2, Node *node) {
    node->inputs.push_back(node1);
    node->inputs.push_back(node2);
}

BNBackwardOp::~BNBackwardOp() {

}

void BNBackwardOp::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                               unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_DIFF_DST, input_memory.at(0)});
    args.insert({DNNL_ARG_SRC, input_memory.at(1)});
    args.insert({DNNL_ARG_DIFF_SRC, input_memory.at(2)});
    args.insert({DNNL_ARG_WORKSPACE, workspace_map.at(node->forwardNode).at(0)});
    args.insert({DNNL_ARG_MEAN, workspace_map.at(node->forwardNode).at(1)});
    args.insert({DNNL_ARG_VARIANCE, workspace_map.at(node->forwardNode).at(2)});
}

primitive BNBackwardOp::getPrimitive(Node *node, engine &engine,
                                         unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    batch_normalization_backward::primitive_desc *b = (batch_normalization_backward::primitive_desc*)a;
    return batch_normalization_backward(*b);
}

void
BNBackwardOp::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    assert(0);
}

void
BNBackwardOp::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
                              unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    node->desc = new memory::desc(node->inputs.at(0)->desc->data);
    primitive_desc_base *a = &(primitive_desc_map.at(node->forwardNode));
    batch_normalization_forward::primitive_desc *b = (batch_normalization_forward::primitive_desc*)a;
    batch_normalization_backward::primitive_desc new_primitive_desc(
            {prop_kind::backward_data, *(node->desc), *(node->desc), 1.e-10f, normalization_flags::none}, eng, *b);
    primitive_desc_map.insert({node, new_primitive_desc});
}


BNReluOp::BNReluOp(Node *node1, Node *node) {
    node->inputs.push_back(node1);
}

BNReluOp::~BNReluOp() {

}

void BNReluOp::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                       unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_SRC, input_memory.at(0)});
    args.insert({DNNL_ARG_DST, input_memory.at(1)});
    args.insert({DNNL_ARG_WORKSPACE, workspace_map.at(node).at(0)});
    args.insert({DNNL_ARG_MEAN, workspace_map.at(node).at(1)});
    args.insert({DNNL_ARG_VARIANCE, workspace_map.at(node).at(2)});
}

primitive
BNReluOp::getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    batch_normalization_forward::primitive_desc *b = (batch_normalization_forward::primitive_desc*)a;
    return batch_normalization_forward(*b);
}

void BNReluOp::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    Node *new_grad_node = CBNReluBackwardOp(output_grad, node->inputs.at(0), newNodeList);
    new_grad_node->forwardNode = node;
    inputs_grad.push_back(new_grad_node);
}

void BNReluOp::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
                           unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    node->desc = new memory::desc(node->inputs.at(0)->desc->data);
    batch_normalization_forward::primitive_desc new_primitive_desc(
            {prop_kind::forward_training, *(node->desc), 1.e-10f, normalization_flags::fuse_norm_relu}, eng);
    vector<memory::desc> vmd;
    vmd.push_back(new_primitive_desc.workspace_desc());
    vmd.push_back(new_primitive_desc.mean_desc());
    vmd.push_back(new_primitive_desc.variance_desc());
    workspace_desc_map.insert({node, vmd});
    primitive_desc_map.insert({node, new_primitive_desc});
}

BNReluBackwardOp::BNReluBackwardOp(Node *node1, Node *node2, Node *node) {
    node->inputs.push_back(node1);
    node->inputs.push_back(node2);
}

BNReluBackwardOp::~BNReluBackwardOp() {

}

void BNReluBackwardOp::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                               unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_DIFF_DST, input_memory.at(0)});
    args.insert({DNNL_ARG_SRC, input_memory.at(1)});
    args.insert({DNNL_ARG_DIFF_SRC, input_memory.at(2)});
    args.insert({DNNL_ARG_WORKSPACE, workspace_map.at(node->forwardNode).at(0)});
    args.insert({DNNL_ARG_MEAN, workspace_map.at(node->forwardNode).at(1)});
    args.insert({DNNL_ARG_VARIANCE, workspace_map.at(node->forwardNode).at(2)});
}

primitive BNReluBackwardOp::getPrimitive(Node *node, engine &engine,
                                         unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    batch_normalization_backward::primitive_desc *b = (batch_normalization_backward::primitive_desc*)a;
    return batch_normalization_backward(*b);
}

void
BNReluBackwardOp::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    assert(0);
}

void
BNReluBackwardOp::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
                              unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    node->desc = new memory::desc(node->inputs.at(0)->desc->data);
    primitive_desc_base *a = &(primitive_desc_map.at(node->forwardNode));
    batch_normalization_forward::primitive_desc *b = (batch_normalization_forward::primitive_desc*)a;
    batch_normalization_backward::primitive_desc new_primitive_desc(
            {prop_kind::backward_data, *(node->desc), *(node->desc), 1.e-10f, normalization_flags::fuse_norm_relu}, eng, *b);
    primitive_desc_map.insert({node, new_primitive_desc});
}

Dropout2Op::Dropout2Op(Node *node1, Node *node2, float keep_prob, Node *node) {
    node->inputs.push_back(node1);
    node->inputs.push_back(node2);
    this->keep_prob = keep_prob;
}

Dropout2Op::~Dropout2Op() {

}

void Dropout2Op::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                         unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_SRC_0, input_memory.at(0)});
    args.insert({DNNL_ARG_SRC_1, input_memory.at(1)});
    args.insert({DNNL_ARG_DST, input_memory.at(2)});
}

primitive
Dropout2Op::getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    binary::primitive_desc *b = (binary::primitive_desc*)a;
    return binary(*b);
}

void Dropout2Op::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    //没用
    inputs_grad.push_back(output_grad);
    inputs_grad.push_back(output_grad);
}

void Dropout2Op::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
                             unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    node->desc = new memory::desc(node->inputs.at(0)->desc->data);
    dnnl::post_ops po;
    po.append_eltwise(1, dnnl::algorithm::eltwise_linear, 1/keep_prob, 0);
    dnnl::primitive_attr attr;
    attr.set_post_ops(po);
    binary::primitive_desc new_primitive_desc(
            {algorithm::binary_lt, *(node->desc), *(node->inputs.at(1)->desc), *(node->desc)},
                                              attr, eng);
    primitive_desc_map.insert({node, new_primitive_desc});
}

Dropout3Op::Dropout3Op(Node *node1, Node *node2, Node *node) {
    node->inputs.push_back(node1);
    node->inputs.push_back(node2);
}

Dropout3Op::~Dropout3Op() {

}

void Dropout3Op::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                         unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_SRC_0, input_memory.at(0)});
    args.insert({DNNL_ARG_SRC_1, input_memory.at(1)});
    args.insert({DNNL_ARG_DST, input_memory.at(2)});
}

primitive
Dropout3Op::getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    binary::primitive_desc *b = (binary::primitive_desc*)a;
    return binary(*b);
}

void Dropout3Op::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    inputs_grad.push_back(CDropoutBackwardOp(output_grad, node->inputs.at(1), newNodeList));
    //第二个无所谓
    inputs_grad.push_back(output_grad);
}

void Dropout3Op::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
                             unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    node->desc = new memory::desc(node->inputs.at(0)->desc->data);
    binary::primitive_desc new_primitive_desc(
            {algorithm::binary_mul , *(node->desc), *(node->desc), *(node->desc)}, eng);
    primitive_desc_map.insert({node, new_primitive_desc});
}


AddOp::AddOp(Node *node1, Node *node2, Node *node) {
    node->inputs.push_back(node1);
    node->inputs.push_back(node2);
}

AddOp::~AddOp() {

}

void AddOp::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
        unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_SRC_0, input_memory.at(0)});
    args.insert({DNNL_ARG_SRC_1, input_memory.at(1)});
    args.insert({DNNL_ARG_DST, input_memory.at(2)});
}

primitive AddOp::getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    binary::primitive_desc *b = (binary::primitive_desc*)a;
    return binary(*b);
}

void AddOp::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    inputs_grad.push_back(output_grad);
    inputs_grad.push_back(output_grad);
}

void AddOp::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
        unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    node->desc = new memory::desc(node->inputs.at(0)->desc->data);
    binary::primitive_desc new_primitive_desc({algorithm::binary_add, *(node->desc), *(node->desc), *(node->desc)},
            eng);
    primitive_desc_map.insert({node, new_primitive_desc});
}




SubOp::SubOp(Node *node1, Node *node2, Node *node) {
    node->inputs.push_back(node1);
    node->inputs.push_back(node2);
}

SubOp::~SubOp() {

}

void SubOp::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
        unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_SRC_0, input_memory.at(0)});
    args.insert({DNNL_ARG_SRC_1, input_memory.at(1)});
    args.insert({DNNL_ARG_DST, input_memory.at(2)});
}

primitive SubOp::getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    binary::primitive_desc *b = (binary::primitive_desc*)a;
    return binary(*b);
}

void SubOp::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    inputs_grad.push_back(output_grad);
    inputs_grad.push_back(CReverseOp(output_grad, newNodeList));
}

void SubOp::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
        unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    node->desc = new memory::desc(node->inputs.at(0)->desc->data);
    binary::primitive_desc new_primitive_desc({algorithm::binary_sub,*(node->desc),*(node->desc),*(node->desc)}, eng);
    primitive_desc_map.insert({node, new_primitive_desc});
}


MatMulOp::MatMulOp(Node *node1, Node *node2, Node *node3, Node *node) {
    node->inputs.push_back(node1);
    node->inputs.push_back(node2);
    node->inputs.push_back(node3);
}

MatMulOp::~MatMulOp() {

}

void MatMulOp::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
        unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_SRC, input_memory.at(0)});
    args.insert({DNNL_ARG_WEIGHTS, input_memory.at(1)});
    args.insert({DNNL_ARG_BIAS, input_memory.at(2)});
    args.insert({DNNL_ARG_DST, input_memory.at(3)});
}

primitive MatMulOp::getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    matmul::primitive_desc *b = (matmul::primitive_desc*)a;
    return matmul(*b);
}

void MatMulOp::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    inputs_grad.push_back(CMatMulBackwardDataOp(output_grad, node->inputs.at(1), newNodeList));
    inputs_grad.push_back(CMatMulBackwardWeightsOp(node->inputs.at(0), output_grad, newNodeList));
    inputs_grad.push_back(CMatMulBackwardBiasOp(output_grad, newNodeList));
}

void MatMulOp::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
        unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    memory::desc *input_desc1 = node->inputs.at(0)->desc;
    memory::desc *input_desc2 = node->inputs.at(1)->desc;
    memory::desc *input_desc3 = node->inputs.at(2)->desc;
    node->desc = new memory::desc({input_desc1->dims().at(0), input_desc2->dims().at(1)},
                                           input_desc1->data_type(),
                                           memory::format_tag::ab);
    matmul::primitive_desc new_primitive_desc({*input_desc1, *input_desc2, *input_desc3, *(node->desc)}, eng);
    primitive_desc_map.insert({node, new_primitive_desc});
}

MatMulBackwardDataOp::MatMulBackwardDataOp(Node *node1, Node *node2, Node *node) {
    node->inputs.push_back(node1);
    node->inputs.push_back(node2);
}

MatMulBackwardDataOp::~MatMulBackwardDataOp() {
    delete node2_new_desc;
}

void
MatMulBackwardDataOp::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
        unordered_map<Node *, vector<memory>> &workspace_map) {
    memory new_node2_memory(*node2_new_desc, input_memory.at(1).get_engine(),
            input_memory.at(1).get_data_handle());
    args.insert({DNNL_ARG_SRC, input_memory.at(0)});
    args.insert({DNNL_ARG_WEIGHTS, new_node2_memory});
    args.insert({DNNL_ARG_DST, input_memory.at(2)});
}

primitive MatMulBackwardDataOp::getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    matmul::primitive_desc *b = (matmul::primitive_desc*)a;
    return matmul(*b);
}

void MatMulBackwardDataOp::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    assert(0);
}

void MatMulBackwardDataOp::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
        unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    memory::desc *input_desc1 = node->inputs.at(0)->desc;
    memory::desc *input_desc2 = node->inputs.at(1)->desc;
    memory::dims input1_dims = input_desc1->dims();
    memory::dims input2_dims = input_desc2->dims();
    node2_new_desc = new memory::desc({input2_dims.at(1), input2_dims.at(0)},
                                            input_desc1->data_type(),
                                            memory::format_tag::ba);
    node->desc = new memory::desc({input1_dims.at(0), input2_dims.at(0)},
                                        input_desc1->data_type(),
                                        memory::format_tag::ab);
    matmul::primitive_desc new_primitive_desc({*input_desc1, *node2_new_desc, *(node->desc)}, eng);
    primitive_desc_map.insert({node, new_primitive_desc});
}

MatMulBackwardWeightsOp::MatMulBackwardWeightsOp(Node *node1, Node *node2, Node *node) {
    node->inputs.push_back(node1);
    node->inputs.push_back(node2);
}

MatMulBackwardWeightsOp::~MatMulBackwardWeightsOp() {
    delete node1_new_desc;
}

void MatMulBackwardWeightsOp::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
        unordered_map<Node *, vector<memory>> &workspace_map) {
    memory new_node1_memory(*node1_new_desc, input_memory.at(0).get_engine(),
                            input_memory.at(0).get_data_handle());
    args.insert({DNNL_ARG_SRC, new_node1_memory});
    args.insert({DNNL_ARG_WEIGHTS, input_memory.at(1)});
    args.insert({DNNL_ARG_DST, input_memory.at(2)});
}

primitive MatMulBackwardWeightsOp::getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    matmul::primitive_desc *b = (matmul::primitive_desc*)a;
    return matmul(*b);
}

void MatMulBackwardWeightsOp::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    assert(0);
}

void MatMulBackwardWeightsOp::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
        unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    memory::desc *input_desc1 = node->inputs.at(0)->desc;
    memory::desc *input_desc2 = node->inputs.at(1)->desc;
    memory::dims input1_dims = input_desc1->dims();
    memory::dims input2_dims = input_desc2->dims();
    node1_new_desc = new memory::desc({input1_dims.at(1), input1_dims.at(0)},
                                      input_desc1->data_type(),
                                      memory::format_tag::ba);
    node->desc = new memory::desc({input1_dims.at(1), input2_dims.at(1)},
                                  input_desc1->data_type(),
                                  memory::format_tag::ab);
    matmul::primitive_desc new_primitive_desc({*node1_new_desc, *input_desc2, *(node->desc)}, eng);
    primitive_desc_map.insert({node, new_primitive_desc});
}

MatMulBackwardBiasOp::MatMulBackwardBiasOp(Node *node1, Node *node) {
    node->inputs.push_back(node1);
}

MatMulBackwardBiasOp::~MatMulBackwardBiasOp() {

}

void MatMulBackwardBiasOp::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
        unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_SRC, input_memory.at(0)});
    args.insert({DNNL_ARG_DST, input_memory.at(1)});
}

primitive MatMulBackwardBiasOp::getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    reduction::primitive_desc *b = (reduction::primitive_desc*)a;
    return reduction(*b);
}

void MatMulBackwardBiasOp::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    assert(0);
}

void MatMulBackwardBiasOp::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
        unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    memory::desc *input_desc1 = node->inputs.at(0)->desc;
    node->desc = new memory::desc({1, input_desc1->dims().at(1)},
                                  input_desc1->data_type(),
                                  memory::format_tag::ab);
    reduction::primitive_desc new_primitive_desc({algorithm::reduction_sum, *input_desc1, *(node->desc), 1, 1}, eng);
    primitive_desc_map.insert({node, new_primitive_desc});
}

ReOrderOp::ReOrderOp(Node *node1, Node *node) {
    node->inputs.push_back(node1);
}

ReOrderOp::~ReOrderOp() {

}

void ReOrderOp::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
        unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_FROM, input_memory.at(0)});
    args.insert({DNNL_ARG_TO, input_memory.at(1)});
}

primitive
ReOrderOp::getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    reorder::primitive_desc *b = (reorder::primitive_desc*)a;
    return reorder(*b);
}

void ReOrderOp::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    assert(0);
}

void ReOrderOp::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
        unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    reorder::primitive_desc new_primitive_desc(eng, *(node->inputs.at(0)->desc), eng, *(node->desc));
    primitive_desc_map.insert({node, new_primitive_desc});
}

ConvolutionOp::ConvolutionOp(Node *node1, Node *node2, Node *node3, Node *node, memory::dims &strides_dims,
                             memory::dims &padding_dims_l, memory::dims &padding_dims_r) {
    node->inputs.push_back(node1);
    node->inputs.push_back(node2);
    node->inputs.push_back(node3);
    this->strides_dims = strides_dims;
    this->padding_dims_l = padding_dims_l;
    this->padding_dims_r = padding_dims_r;
}

ConvolutionOp::~ConvolutionOp() {

}

void ConvolutionOp::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
        unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_SRC, input_memory.at(0)});
    args.insert({DNNL_ARG_WEIGHTS, input_memory.at(1)});
    args.insert({DNNL_ARG_BIAS, input_memory.at(2)});
    args.insert({DNNL_ARG_DST, input_memory.at(3)});
}

primitive ConvolutionOp::getPrimitive(Node *node, engine &engine,
                                      unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    convolution_forward::primitive_desc *b = (convolution_forward::primitive_desc*)a;
    return convolution_forward(*b);
}

void ConvolutionOp::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    Node *g_data = CConvolutionBackwardDataOp(node->inputs.at(1), output_grad,
            strides_dims, padding_dims_l, padding_dims_r, newNodeList);
    g_data->forwardNode = node;
    Node *g_bias = CConvolutionBackwardBiasOp(node->inputs.at(2)->desc, newNodeList);
    Node *g_weights = CConvolutionBackwardWeightsOp(node->inputs.at(0),
            g_bias, output_grad, strides_dims, padding_dims_l,padding_dims_r, newNodeList);
    g_weights->forwardNode = node;
    inputs_grad.push_back(g_data);
    inputs_grad.push_back(g_weights);
    inputs_grad.push_back(g_bias);
}

void
ConvolutionOp::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
        unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    memory::desc *inputdesc1 = node->inputs.at(0)->desc;
    memory::desc *inputdesc2 = node->inputs.at(1)->desc;
    memory::dims src_dims = inputdesc1->dims();
    memory::dims weights_dims = inputdesc2->dims();
    int OH = (src_dims.at(2)-weights_dims.at(2)+padding_dims_l.at(0)+padding_dims_r.at(0))
            / strides_dims.at(0) + 1;
    int OW = (src_dims.at(3)-weights_dims.at(3)+padding_dims_l.at(1)+padding_dims_r.at(1))
             / strides_dims.at(1) + 1;
    memory::dims dst_dims = {src_dims.at(0), weights_dims.at(0), OH, OW};
    auto conv_src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::any);
    auto conv_weights_md = memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::any);
    auto conv_dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::any);
    convolution_forward::primitive_desc new_primitive_desc({prop_kind::forward_training,
                                                            algorithm::convolution_direct, conv_src_md, conv_weights_md,
                                                            *(node->inputs.at(2)->desc), conv_dst_md, strides_dims, padding_dims_l,
                                                            padding_dims_r},eng);
    node->desc = new memory::desc(new_primitive_desc.dst_desc());
    primitive_desc_map.insert({node, new_primitive_desc});
}

ConvolutionBackwardDataOp::ConvolutionBackwardDataOp(Node *node1, Node *node2, Node *node, memory::dims &strides_dims,
                                                     memory::dims &padding_dims_l, memory::dims &padding_dims_r) {
    node->inputs.push_back(node1);
    node->inputs.push_back(node2);
    this->strides_dims = strides_dims;
    this->padding_dims_l = padding_dims_l;
    this->padding_dims_r = padding_dims_r;
}

ConvolutionBackwardDataOp::~ConvolutionBackwardDataOp() {

}

void ConvolutionBackwardDataOp::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
        unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_WEIGHTS, input_memory.at(0)});
    args.insert({DNNL_ARG_DIFF_DST, input_memory.at(1)});
    args.insert({DNNL_ARG_DIFF_SRC, input_memory.at(2)});
}

primitive ConvolutionBackwardDataOp::getPrimitive(Node *node, engine &engine,
                                                  unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    convolution_backward_data::primitive_desc *b = (convolution_backward_data::primitive_desc*)a;
    return convolution_backward_data(*b);
}

void ConvolutionBackwardDataOp::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad,
                                         vector<Node *> &newNodeList) {
    assert(0);
}

void ConvolutionBackwardDataOp::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
        unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    memory::desc *inputdesc1 = node->inputs.at(0)->desc;
    memory::desc *inputdesc2 = node->inputs.at(1)->desc;
    memory::dims weights_dims = inputdesc1->dims();
    memory::dims diff_dst_dims = inputdesc2->dims();
    memory::dims diff_src_dims = node->forwardNode->inputs.at(0)->desc->dims();
    auto conv_diff_src_md = memory::desc(diff_src_dims, memory::data_type::f32, memory::format_tag::any);
    auto conv_weights_md = memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::any);
    auto conv_diff_dst_md = memory::desc(diff_dst_dims, memory::data_type::f32, memory::format_tag::any);
    primitive_desc_base *a = &(primitive_desc_map.at(node->forwardNode));
    convolution_forward::primitive_desc *b = (convolution_forward::primitive_desc*)a;
    convolution_backward_data::primitive_desc new_primitive_desc({algorithm::convolution_direct, conv_diff_src_md,
            conv_weights_md, conv_diff_dst_md, strides_dims, padding_dims_l, padding_dims_r}, eng, *b);
    node->desc = new memory::desc(new_primitive_desc.diff_src_desc());
    primitive_desc_map.insert({node, new_primitive_desc});
}

ConvolutionBackwardWeightsOp::ConvolutionBackwardWeightsOp(Node *node1, Node *node2, Node *node3, Node *node,
                                                           memory::dims &strides_dims, memory::dims &padding_dims_l,
                                                           memory::dims &padding_dims_r) {
    node->inputs.push_back(node1);
    node->inputs.push_back(node2);
    node->inputs.push_back(node3);
    this->strides_dims = strides_dims;
    this->padding_dims_l = padding_dims_l;
    this->padding_dims_r = padding_dims_r;
}

ConvolutionBackwardWeightsOp::~ConvolutionBackwardWeightsOp() {

}

void ConvolutionBackwardWeightsOp::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
        unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_SRC, input_memory.at(0)});
    args.insert({DNNL_ARG_DIFF_BIAS, input_memory.at(1)});
    args.insert({DNNL_ARG_DIFF_DST, input_memory.at(2)});
    args.insert({DNNL_ARG_DIFF_WEIGHTS, input_memory.at(3)});
}

primitive ConvolutionBackwardWeightsOp::getPrimitive(Node *node, engine &engine,
                                                     unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    convolution_backward_weights::primitive_desc *b = (convolution_backward_weights::primitive_desc*)a;
    return convolution_backward_weights(*b);
}

void ConvolutionBackwardWeightsOp::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad,
                                            vector<Node *> &newNodeList) {
    assert(0);
}

void ConvolutionBackwardWeightsOp::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
        unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    memory::desc *inputdesc1 = node->inputs.at(0)->desc;
    memory::desc *inputdesc2 = node->inputs.at(1)->desc;
    memory::desc *inputdesc3 = node->inputs.at(2)->desc;
    memory::dims src_dims = inputdesc1->dims();
    memory::dims diff_dst_dims = inputdesc3->dims();
    memory::dims diff_weights_dims = node->forwardNode->inputs.at(1)->desc->dims();
    auto conv_src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::any);
    auto conv_diff_weights_md = memory::desc(diff_weights_dims, memory::data_type::f32, memory::format_tag::any);
    auto conv_diff_dst_md = memory::desc(diff_dst_dims, memory::data_type::f32, memory::format_tag::any);
    primitive_desc_base *a = &(primitive_desc_map.at(node->forwardNode));
    convolution_forward::primitive_desc *b = (convolution_forward::primitive_desc*)a;
    convolution_backward_weights::primitive_desc new_primitive_desc(
            {algorithm::convolution_direct, conv_src_md, conv_diff_weights_md, *inputdesc2, conv_diff_dst_md,
             strides_dims, padding_dims_l, padding_dims_r}, eng, *b);
    node->desc = new memory::desc(new_primitive_desc.diff_weights_desc());
    primitive_desc_map.insert({node, new_primitive_desc});
}


PoolingOp::PoolingOp(Node *node1, Node *node, memory::dims &strides_dims, memory::dims &kernel,
                     memory::dims &padding_dims_l, memory::dims &padding_dims_r) {
    node->inputs.push_back(node1);
    this->strides_dims = strides_dims;
    this->kernel = kernel;
    this->padding_dims_l = padding_dims_l;
    this->padding_dims_r = padding_dims_r;
}

PoolingOp::~PoolingOp() {

}

void PoolingOp::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
        unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_SRC, input_memory.at(0)});
    args.insert({DNNL_ARG_DST, input_memory.at(1)});
    args.insert({DNNL_ARG_WORKSPACE, workspace_map.at(node).at(0)});
}

primitive
PoolingOp::getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    pooling_forward::primitive_desc *b = (pooling_forward::primitive_desc*)a;
    return pooling_forward(*b);
}

void PoolingOp::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    Node *new_grad_node = CPoolingBackwardOp(output_grad, strides_dims, kernel, padding_dims_l, padding_dims_r, newNodeList);
    new_grad_node->forwardNode = node;
    inputs_grad.push_back(new_grad_node);
}

void PoolingOp::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
        unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    memory::desc *inputdesc1 = node->inputs.at(0)->desc;
    memory::dims src_dims = inputdesc1->dims();
    int OH = (src_dims.at(2)-kernel.at(0)+padding_dims_l.at(0)+padding_dims_r.at(0))
             / strides_dims.at(0) + 1;
    int OW = (src_dims.at(3)-kernel.at(1)+padding_dims_l.at(1)+padding_dims_r.at(1))
             / strides_dims.at(1) + 1;
    memory::dims dst_dims = {src_dims.at(0), src_dims.at(1), OH, OW};
    auto dst_md = memory::desc({src_dims.at(0), src_dims.at(1), OH, OW}, memory::data_type::f32, memory::format_tag::any);
    pooling_forward::primitive_desc new_primitive_desc({prop_kind::forward_training, algorithm::pooling_max,
    *inputdesc1, dst_md, strides_dims, kernel, padding_dims_l, padding_dims_r}, eng);
    node->desc = new memory::desc(new_primitive_desc.dst_desc());
    vector<memory::desc> vmd;
    vmd.push_back(new_primitive_desc.workspace_desc());
    workspace_desc_map.insert({node, vmd});
    primitive_desc_map.insert({node, new_primitive_desc});
}

PoolingBackwardOp::PoolingBackwardOp(Node *node1, Node *node, memory::dims &strides_dims, memory::dims &kernel,
                                     memory::dims &padding_dims_l, memory::dims &padding_dims_r) {
    node->inputs.push_back(node1);
    this->strides_dims = strides_dims;
    this->kernel = kernel;
    this->padding_dims_l = padding_dims_l;
    this->padding_dims_r = padding_dims_r;
}

PoolingBackwardOp::~PoolingBackwardOp() {

}

void PoolingBackwardOp::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                                unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_DIFF_DST, input_memory.at(0)});
    args.insert({DNNL_ARG_DIFF_SRC, input_memory.at(1)});
    args.insert({DNNL_ARG_WORKSPACE, workspace_map.at(node->forwardNode).at(0)});
}

primitive PoolingBackwardOp::getPrimitive(Node *node, engine &engine,
                                          unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    pooling_backward::primitive_desc *b = (pooling_backward::primitive_desc*)a;
    return pooling_backward(*b);
}

void
PoolingBackwardOp::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    assert(0);
}

void
PoolingBackwardOp::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
                               unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    memory::desc *diff_dst_desc = node->inputs.at(0)->desc;
    memory::desc *diff_src_desc = node->forwardNode->inputs.at(0)->desc;
    node->desc = new memory::desc(diff_src_desc->data);
    primitive_desc_base *a = &(primitive_desc_map.at(node->forwardNode));
    pooling_forward::primitive_desc *b = (pooling_forward::primitive_desc*)a;
    pooling_backward::primitive_desc new_primitive_desc(
            {algorithm::pooling_max, *diff_src_desc, *diff_dst_desc, strides_dims, kernel,
             padding_dims_l, padding_dims_r}, eng, *b);
    primitive_desc_map.insert({node, new_primitive_desc});
}

FlatOp::FlatOp(Node *node1, Node *node) {
    node->inputs.push_back(node1);
}

FlatOp::~FlatOp() {
    delete node_new_desc;
}

void FlatOp::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                     unordered_map<Node *, vector<memory>> &workspace_map) {
    memory new_node_memory(*node_new_desc, input_memory.at(1).get_engine(),
                            input_memory.at(1).get_data_handle());
    args.insert({DNNL_ARG_FROM, input_memory.at(0)});
    args.insert({DNNL_ARG_TO, new_node_memory});
}

primitive
FlatOp::getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    reorder::primitive_desc *b = (reorder::primitive_desc*)a;
    return reorder(*b);
}

void FlatOp::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    Node *new_grad_node = CFlatBackwardOp(output_grad, newNodeList);
    new_grad_node->forwardNode = node;
    inputs_grad.push_back(new_grad_node);
}

void FlatOp::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
                         unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    memory::desc *src_desc = node->inputs.at(0)->desc;
    memory::dims src_dims = src_desc->dims();
    node_new_desc = new memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
    reorder::primitive_desc new_primitive_desc(eng, *src_desc, eng, *node_new_desc);
    node->desc = new memory::desc(
            {src_dims.at(0), src_dims.at(1)*src_dims.at(2)*src_dims.at(3)},memory::data_type::f32, memory::format_tag::ab);
    primitive_desc_map.insert({node, new_primitive_desc});
}

FlatBackwardOp::FlatBackwardOp(Node *node1, Node *node) {
    node->inputs.push_back(node1);
}

FlatBackwardOp::~FlatBackwardOp() {
    delete node_new_desc;
}

void FlatBackwardOp::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                             unordered_map<Node *, vector<memory>> &workspace_map) {
    memory new_node_memory(*node_new_desc, input_memory.at(0).get_engine(),
                           input_memory.at(0).get_data_handle());
    args.insert({DNNL_ARG_FROM, new_node_memory});
    args.insert({DNNL_ARG_TO, input_memory.at(1)});
}

primitive FlatBackwardOp::getPrimitive(Node *node, engine &engine,
                                       unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    reorder::primitive_desc *b = (reorder::primitive_desc*)a;
    return reorder(*b);
}

void FlatBackwardOp::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    assert(0);
}

void
FlatBackwardOp::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
                            unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    memory::desc *src_desc = node->inputs.at(0)->desc;
    memory::desc *dst_desc = node->forwardNode->inputs.at(0)->desc;
    node_new_desc = new memory::desc(dst_desc->dims(), memory::data_type::f32, memory::format_tag::nchw);
    reorder::primitive_desc new_primitive_desc(eng, *node_new_desc, eng, *dst_desc);
    node->desc = new memory::desc(dst_desc->data);
    primitive_desc_map.insert({node, new_primitive_desc});
}



SoftmaxCrossEntropy_Training1Op::SoftmaxCrossEntropy_Training1Op(Node *node1, Node *node) {
    node->inputs.push_back(node1);
}

SoftmaxCrossEntropy_Training1Op::~SoftmaxCrossEntropy_Training1Op() {

}

void
SoftmaxCrossEntropy_Training1Op::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
        unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_SRC, input_memory.at(0)});
    args.insert({DNNL_ARG_DST, input_memory.at(1)});
}

primitive SoftmaxCrossEntropy_Training1Op::getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    softmax_forward::primitive_desc *b = (softmax_forward::primitive_desc*)a;
    return softmax_forward(*b);
}

void SoftmaxCrossEntropy_Training1Op::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    inputs_grad.push_back(output_grad);
}

void SoftmaxCrossEntropy_Training1Op::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
        unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    node->desc = new memory::desc(node->inputs.at(0)->desc->data);
    softmax_forward::primitive_desc new_primitive_desc({prop_kind::forward_training, *(node->desc), 1}, eng);
    primitive_desc_map.insert({node, new_primitive_desc});
}

SoftmaxCrossEntropy_Training2Op::SoftmaxCrossEntropy_Training2Op(Node *node1, Node *node2, Node *node) {
    node->inputs.push_back(node1);
    node->inputs.push_back(node2);
}

SoftmaxCrossEntropy_Training2Op::~SoftmaxCrossEntropy_Training2Op() {

}

void
SoftmaxCrossEntropy_Training2Op::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
        unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_SRC_0, input_memory.at(0)});
    args.insert({DNNL_ARG_SRC_1, input_memory.at(1)});
    args.insert({DNNL_ARG_DST, input_memory.at(2)});
}

primitive SoftmaxCrossEntropy_Training2Op::getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    binary::primitive_desc *b = (binary::primitive_desc*)a;
    return binary(*b);
}

void SoftmaxCrossEntropy_Training2Op::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    inputs_grad.push_back(node);
    /* 第二个导数这里其实是没用的 */
    inputs_grad.push_back(node);
}

void SoftmaxCrossEntropy_Training2Op::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
        unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    node->desc = new memory::desc(node->inputs.at(0)->desc->data);
    binary::primitive_desc new_primitive_desc({algorithm::binary_sub,*(node->desc),*(node->desc),*(node->desc)}, eng);
    primitive_desc_map.insert({node, new_primitive_desc});
}

SgdOp::SgdOp(Node *node1, Node *node2, Node *node) {
    node->inputs.push_back(node1);
    node->inputs.push_back(node2);
}

SgdOp::~SgdOp() {

}

void SgdOp::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
        unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_SRC_0, input_memory.at(0)});
    args.insert({DNNL_ARG_SRC_1, input_memory.at(1)});
    args.insert({DNNL_ARG_DST, input_memory.at(2)});
}

primitive SgdOp::getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    binary::primitive_desc *b = (binary::primitive_desc*)a;
    return binary(*b);
}

void SgdOp::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    assert(0);
}

void SgdOp::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
        unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    node->desc = new memory::desc(node->inputs.at(0)->desc->data);
    binary::primitive_desc new_primitive_desc(
            {algorithm::binary_mul, *(node->desc), *(node->inputs.at(1)->desc), *(node->desc)},
            eng);
    primitive_desc_map.insert({node, new_primitive_desc});
}

Adam_Gt2Op::Adam_Gt2Op(Node *node1, float b2, Node *node) {
    node->inputs.push_back(node1);
    this->b2 = b2;
}

Adam_Gt2Op::~Adam_Gt2Op() {

}

void Adam_Gt2Op::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                        unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_SRC, input_memory.at(0)});
    args.insert({DNNL_ARG_DST, input_memory.at(1)});
}

primitive
Adam_Gt2Op::getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    eltwise_forward::primitive_desc *b = (eltwise_forward::primitive_desc*)a;
    return eltwise_forward(*b);
}

void Adam_Gt2Op::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    assert(0);
}

void Adam_Gt2Op::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
                            unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    node->desc = new memory::desc(node->inputs.at(0)->desc->data);
    eltwise_forward::primitive_desc new_primitive_desc(
            {prop_kind::forward_inference, algorithm::eltwise_pow, *(node->desc), 1-b2, 2}, eng);
    primitive_desc_map.insert({node, new_primitive_desc});
}

Adam_UpMOp::Adam_UpMOp(Node *node1, Node *node2, float b1, Node *node) {
    node->inputs.push_back(node1);
    node->inputs.push_back(node2);
    this->b1 = b1;
}

Adam_UpMOp::~Adam_UpMOp() {

}

void Adam_UpMOp::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                         unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_SRC, input_memory.at(0)});
    args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, input_memory.at(1)});
    args.insert({DNNL_ARG_DST, input_memory.at(2)});
}

primitive
Adam_UpMOp::getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    eltwise_forward::primitive_desc *b = (eltwise_forward::primitive_desc*)a;
    return eltwise_forward(*b);
}

void Adam_UpMOp::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    assert(0);
}

void Adam_UpMOp::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
                             unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    node->desc = new memory::desc(node->inputs.at(0)->desc->data);
    dnnl::post_ops po;
    po.append_binary(algorithm::binary_add, *(node->desc));
    primitive_attr attr;
    attr.set_post_ops(po);
    eltwise_forward::primitive_desc new_primitive_desc(
            {prop_kind::forward_inference, algorithm::eltwise_linear, *(node->desc), b1, 0},
            attr, eng);
    primitive_desc_map.insert({node, new_primitive_desc});
}

Adam_V_Op::Adam_V_Op(Node *node1, Node *node2, float e, Node *node) {
    node->inputs.push_back(node1);
    node->inputs.push_back(node2);
    this->e = e;
}

Adam_V_Op::~Adam_V_Op() {

}

void Adam_V_Op::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                        unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_SRC_0, input_memory.at(0)});
    args.insert({DNNL_ARG_SRC_1, input_memory.at(1)});
    args.insert({DNNL_ARG_DST, input_memory.at(2)});
}

primitive
Adam_V_Op::getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    binary::primitive_desc *b = (binary::primitive_desc*)a;
    return binary(*b);
}

void Adam_V_Op::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    assert(0);
}

void Adam_V_Op::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
                            unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    node->desc = new memory::desc(node->inputs.at(0)->desc->data);
    dnnl::post_ops po;
    po.append_eltwise(1, algorithm::eltwise_sqrt, 1, 0);
    po.append_eltwise(1, algorithm::eltwise_linear, 1, e);
    primitive_attr attr;
    attr.set_post_ops(po);
    binary::primitive_desc new_primitive_desc(
            {algorithm::binary_div, *(node->desc), *(node->inputs.at(1)->desc), *(node->desc)},
            attr, eng);
    primitive_desc_map.insert({node, new_primitive_desc});
}

Adam_M_Op::Adam_M_Op(Node *node1, Node *node2, Node *node3, float lr, Node *node) {
    node->inputs.push_back(node1);
    node->inputs.push_back(node2);
    node->inputs.push_back(node3);
    this->lr = lr;
}

Adam_M_Op::~Adam_M_Op() {

}

void Adam_M_Op::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                        unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_SRC_0, input_memory.at(0)});
    args.insert({DNNL_ARG_SRC_1, input_memory.at(1)});
    args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, input_memory.at(2)});
    args.insert({DNNL_ARG_DST, input_memory.at(3)});
}

primitive
Adam_M_Op::getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    binary::primitive_desc *b = (binary::primitive_desc*)a;
    return binary(*b);
}

void Adam_M_Op::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    assert(0);
}

void Adam_M_Op::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
                            unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    node->desc = new memory::desc(node->inputs.at(0)->desc->data);
    dnnl::post_ops po;
    po.append_binary(algorithm::binary_div, *(node->desc));
    po.append_eltwise(1, algorithm::eltwise_linear, lr, 0);
    primitive_attr attr;
    attr.set_post_ops(po);
    binary::primitive_desc new_primitive_desc(
            {algorithm::binary_div, *(node->desc), *(node->inputs.at(1)->desc), *(node->desc)},
            attr, eng);
    primitive_desc_map.insert({node, new_primitive_desc});
}

















SoftmaxCrossEntropy_Inference1Op::SoftmaxCrossEntropy_Inference1Op(Node *node1, Node *node) {
    node->inputs.push_back(node1);
}

SoftmaxCrossEntropy_Inference1Op::~SoftmaxCrossEntropy_Inference1Op() {

}

void
SoftmaxCrossEntropy_Inference1Op::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
        unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_SRC, input_memory.at(0)});
    args.insert({DNNL_ARG_DST, input_memory.at(1)});
}

primitive SoftmaxCrossEntropy_Inference1Op::getPrimitive(Node *node, engine &engine,
                                                         unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    logsoftmax_forward::primitive_desc *b = (logsoftmax_forward::primitive_desc*)a;
    return logsoftmax_forward(*b);
}

void SoftmaxCrossEntropy_Inference1Op::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    assert(0);
}

void SoftmaxCrossEntropy_Inference1Op::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
        unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    node->desc = new memory::desc(node->inputs.at(0)->desc->data);
    logsoftmax_forward::primitive_desc new_primitive_desc({prop_kind::forward_inference, *(node->desc), 1}, eng);
    primitive_desc_map.insert({node, new_primitive_desc});
}

SoftmaxCrossEntropy_Inference2Op::SoftmaxCrossEntropy_Inference2Op(Node *node1, Node *node2, Node *node) {
    node->inputs.push_back(node1);
    node->inputs.push_back(node2);
}

SoftmaxCrossEntropy_Inference2Op::~SoftmaxCrossEntropy_Inference2Op() {

}

void
SoftmaxCrossEntropy_Inference2Op::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
        unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_SRC_0, input_memory.at(0)});
    args.insert({DNNL_ARG_SRC_1, input_memory.at(1)});
    args.insert({DNNL_ARG_DST, input_memory.at(2)});
}

primitive SoftmaxCrossEntropy_Inference2Op::getPrimitive(Node *node, engine &engine,
                                                         unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    binary::primitive_desc *b = (binary::primitive_desc*)a;
    return binary(*b);
}

void SoftmaxCrossEntropy_Inference2Op::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    assert(0);
}

void SoftmaxCrossEntropy_Inference2Op::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
        unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    node->desc = new memory::desc(node->inputs.at(0)->desc->data);
    binary::primitive_desc new_primitive_desc({algorithm::binary_mul, *(node->desc), *(node->desc), *(node->desc)}, eng);
    primitive_desc_map.insert({node, new_primitive_desc});
}

SoftmaxCrossEntropy_Inference3Op::SoftmaxCrossEntropy_Inference3Op(Node *node1, Node *node) {
    node->inputs.push_back(node1);
}

SoftmaxCrossEntropy_Inference3Op::~SoftmaxCrossEntropy_Inference3Op() {

}

void
SoftmaxCrossEntropy_Inference3Op::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
        unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_SRC, input_memory.at(0)});
    args.insert({DNNL_ARG_DST, input_memory.at(1)});
}

primitive SoftmaxCrossEntropy_Inference3Op::getPrimitive(Node *node, engine &engine,
                                                         unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    reduction::primitive_desc *b = (reduction::primitive_desc*)a;
    return reduction(*b);
}

void SoftmaxCrossEntropy_Inference3Op::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    assert(0);
}

void SoftmaxCrossEntropy_Inference3Op::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
        unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    auto *inputdesc = node->inputs.at(0)->desc;
    node->desc = new memory::desc({1,1}, inputdesc->data_type(), memory::format_tag::ab);
    dnnl::post_ops po;
    po.append_eltwise(1, dnnl::algorithm::eltwise_linear, -1, 0);
    dnnl::primitive_attr attr;
    attr.set_post_ops(po);
    reduction::primitive_desc new_primitive_desc({algorithm::reduction_mean, *inputdesc, *(node->desc), 1, 0},
            attr, eng);
    primitive_desc_map.insert({node, new_primitive_desc});
}

SoftmaxCrossEntropy_Accuracy1Op::SoftmaxCrossEntropy_Accuracy1Op(Node *node1, Node *node) {
    node->inputs.push_back(node1);
}

SoftmaxCrossEntropy_Accuracy1Op::~SoftmaxCrossEntropy_Accuracy1Op() {

}

void
SoftmaxCrossEntropy_Accuracy1Op::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
        unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_SRC, input_memory.at(0)});
    args.insert({DNNL_ARG_DST, input_memory.at(1)});
}

primitive SoftmaxCrossEntropy_Accuracy1Op::getPrimitive(Node *node, engine &engine,
                                                        unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    reduction::primitive_desc *b = (reduction::primitive_desc*)a;
    return reduction(*b);;
}

void SoftmaxCrossEntropy_Accuracy1Op::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    assert(0);
}

void SoftmaxCrossEntropy_Accuracy1Op::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
        unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    auto *inputdesc = node->inputs.at(0)->desc;
    node->desc = new memory::desc({inputdesc->dims().at(0),1}, inputdesc->data_type(), memory::format_tag::ab);
    reduction::primitive_desc new_primitive_desc({algorithm::reduction_max, *inputdesc, *(node->desc), 1, 0}, eng);
    primitive_desc_map.insert({node, new_primitive_desc});
}

SoftmaxCrossEntropy_Accuracy2Op::SoftmaxCrossEntropy_Accuracy2Op(Node *node1, Node *node2, Node *node) {
    node->inputs.push_back(node1);
    node->inputs.push_back(node2);
}

SoftmaxCrossEntropy_Accuracy2Op::~SoftmaxCrossEntropy_Accuracy2Op() {

}

void
SoftmaxCrossEntropy_Accuracy2Op::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
        unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_SRC, input_memory.at(0)});
    args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, input_memory.at(1)});
    args.insert({DNNL_ARG_DST, input_memory.at(2)});
}

primitive SoftmaxCrossEntropy_Accuracy2Op::getPrimitive(Node *node, engine &engine,
                                                        unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    reduction::primitive_desc *b = (reduction::primitive_desc*)a;
    return reduction(*b);
}

void SoftmaxCrossEntropy_Accuracy2Op::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    assert(0);
}

void SoftmaxCrossEntropy_Accuracy2Op::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
        unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    auto *inputdesc = node->inputs.at(0)->desc;
    node->desc = new memory::desc({inputdesc->dims().at(0),1}, inputdesc->data_type(), memory::format_tag::ab);
    dnnl::post_ops po;
    po.append_binary(algorithm::binary_eq, *(node->desc));
    primitive_attr attr;
    attr.set_post_ops(po);
    reduction::primitive_desc new_primitive_desc({algorithm::reduction_sum, *inputdesc, *(node->desc), 1, 0},
            attr, eng);
    primitive_desc_map.insert({node, new_primitive_desc});
}

SoftmaxCrossEntropy_Accuracy3Op::SoftmaxCrossEntropy_Accuracy3Op(Node *node1, Node *node) {
    node->inputs.push_back(node1);
}

SoftmaxCrossEntropy_Accuracy3Op::~SoftmaxCrossEntropy_Accuracy3Op() {

}

void
SoftmaxCrossEntropy_Accuracy3Op::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
        unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_SRC, input_memory.at(0)});
    args.insert({DNNL_ARG_DST, input_memory.at(1)});
}

primitive SoftmaxCrossEntropy_Accuracy3Op::getPrimitive(Node *node, engine &engine,
                                                        unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    reduction::primitive_desc *b = (reduction::primitive_desc*)a;
    return reduction(*b);
}

void SoftmaxCrossEntropy_Accuracy3Op::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    assert(0);
}

void SoftmaxCrossEntropy_Accuracy3Op::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
        unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    auto *inputdesc = node->inputs.at(0)->desc;
    node->desc = new memory::desc({1,1}, inputdesc->data_type(), memory::format_tag::ab);
    reduction::primitive_desc new_primitive_desc({algorithm::reduction_mean, *inputdesc, *(node->desc), 1, 0}, eng);
    primitive_desc_map.insert({node, new_primitive_desc});
}


PredictOp::PredictOp(Node *node1, Node *node2, Node *node) {
    node->inputs.push_back(node1);
    node->inputs.push_back(node2);
}

PredictOp::~PredictOp() {

}

void PredictOp::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                        unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_SRC_0, input_memory.at(0)});
    args.insert({DNNL_ARG_SRC_1, input_memory.at(1)});
    args.insert({DNNL_ARG_DST, input_memory.at(2)});
}

primitive
PredictOp::getPrimitive(Node *node, engine &engine, unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    binary::primitive_desc *b = (binary::primitive_desc*)a;
    return binary(*b);
}

void PredictOp::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    assert(0);
}

void PredictOp::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
                            unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    node->desc = new memory::desc(node->inputs.at(0)->desc->data);
    binary::primitive_desc new_primitive_desc(
            {algorithm::binary_eq, *(node->desc), *(node->inputs.at(1)->desc), *(node->desc)}, eng);
    primitive_desc_map.insert({node, new_primitive_desc});
}

Mse_Training1Op::Mse_Training1Op(Node *node1, Node *node2, Node *node) {
    node->inputs.push_back(node1);
    node->inputs.push_back(node2);
}

Mse_Training1Op::~Mse_Training1Op() {

}

void Mse_Training1Op::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                             unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_SRC_0, input_memory.at(0)});
    args.insert({DNNL_ARG_SRC_1, input_memory.at(1)});
    args.insert({DNNL_ARG_DST, input_memory.at(2)});
}

primitive Mse_Training1Op::getPrimitive(Node *node, engine &engine,
                                       unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    binary::primitive_desc *b = (binary::primitive_desc*)a;
    return binary(*b);
}

void Mse_Training1Op::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    inputs_grad.push_back(CLinearOp(node, 2, 0, newNodeList));
    inputs_grad.push_back(node);
}

void
Mse_Training1Op::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
                            unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    node->desc = new memory::desc(node->inputs.at(0)->desc->data);
    binary::primitive_desc new_primitive_desc({algorithm::binary_sub,*(node->desc),*(node->desc),*(node->desc)}, eng);
    primitive_desc_map.insert({node, new_primitive_desc});
}

Mse_Training2Op::Mse_Training2Op(Node *node1, Node *node2, Node *node) {
    node->inputs.push_back(node1);
    node->inputs.push_back(node2);
}

Mse_Training2Op::~Mse_Training2Op() {

}

void Mse_Training2Op::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                              unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_SRC_0, input_memory.at(0)});
    args.insert({DNNL_ARG_SRC_1, input_memory.at(1)});
    args.insert({DNNL_ARG_DST, input_memory.at(2)});
}

primitive Mse_Training2Op::getPrimitive(Node *node, engine &engine,
                                        unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    binary::primitive_desc *b = (binary::primitive_desc*)a;
    return binary(*b);
}

void
Mse_Training2Op::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    inputs_grad.push_back(node);
    //用不上
    inputs_grad.push_back(node);
}

void
Mse_Training2Op::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
                             unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    node->desc = new memory::desc(node->inputs.at(0)->desc->data);
    dnnl::post_ops po;
    po.append_eltwise(1, dnnl::algorithm::eltwise_linear, 2, 0);
    dnnl::primitive_attr attr;
    attr.set_post_ops(po);
    binary::primitive_desc new_primitive_desc(
            {algorithm::binary_sub,*(node->desc),*(node->desc),*(node->desc)}, attr, eng);
    primitive_desc_map.insert({node, new_primitive_desc});
}


Mse_Inference1Op::Mse_Inference1Op(Node *node1, Node *node) {
    node->inputs.push_back(node1);
}

Mse_Inference1Op::~Mse_Inference1Op() {

}

void Mse_Inference1Op::getargs(Node *node, vector<memory> &input_memory, unordered_map<int, memory> &args,
                              unordered_map<Node *, vector<memory>> &workspace_map) {
    args.insert({DNNL_ARG_SRC, input_memory.at(0)});
    args.insert({DNNL_ARG_DST, input_memory.at(1)});
}

primitive Mse_Inference1Op::getPrimitive(Node *node, engine &engine,
                                        unordered_map<Node *, primitive_desc_base> &primitive_desc_map) {
    primitive_desc_base *a = &(primitive_desc_map.at(node));
    eltwise_forward::primitive_desc *b = (eltwise_forward::primitive_desc*)a;
    return eltwise_forward(*b);
}

void
Mse_Inference1Op::gradient(Node *node, Node *output_grad, vector<Node *> &inputs_grad, vector<Node *> &newNodeList) {
    assert(0);
}

void
Mse_Inference1Op::infer_shape(Node *node, engine &eng, unordered_map<Node *, primitive_desc_base> &primitive_desc_map,
                             unordered_map<Node *, vector<memory::desc>> &workspace_desc_map) {
    node->desc = new memory::desc(node->inputs.at(0)->desc->data);
    eltwise_forward::primitive_desc new_primitive_desc(
            {prop_kind::forward_inference, algorithm::eltwise_square, *(node->desc)}, eng);
    primitive_desc_map.insert({node, new_primitive_desc});
}