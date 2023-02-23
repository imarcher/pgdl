


CREATE FUNCTION create_placeholder(integer[], integer, integer) RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION create_variable(integer[], integer, integer) RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION c_reverse_op(integer) RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION c_relu_op(integer) RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION c_bn_op(integer) RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION c_bn_relu_op(integer) RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION c_dropout_op(integer, double precision) RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION c_dropout_i_op(integer, double precision) RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION c_add_op(integer, integer) RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION c_sub_op(integer, integer) RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION c_matmul_op(integer, integer, integer) RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION c_convolution_op(integer, integer, integer, integer[], integer[], integer[]) RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION c_pooling_op(integer, integer[], integer[], integer[], integer[]) RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION c_flat_op(integer) RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION create_manager() RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION del_manager() RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION softmax_cross_entropy_training(integer, integer, double precision, integer, integer) RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION softmax_cross_entropy_inference(integer, integer, integer) RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION softmax_cross_entropy_accuracy(integer, integer, integer) RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION softmax_cross_entropy_training_accuracy(integer, integer, double precision, integer, integer) RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION predict(integer, integer) RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION mse_training(integer, integer, double precision, integer, integer) RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION mse_training_inference(integer, integer, double precision, integer, integer) RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION inference(integer, integer, integer) RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION read_to_node_scalar(text, text, integer, integer, integer, integer) RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION read_to_node_real(text, text, integer, integer, integer, integer) RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION read_to_node_arr(text, text, integer, integer, integer, integer) RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION init_variables() RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION prepare_memory() RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION compute() RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION compute_ex() RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION compute_accuracy() RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION wait() RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION get_loss() RETURNS real
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION get_accuracy() RETURNS real
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION save_predict(text) RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION update_sgd_lr(double precision) RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION save_model(text) RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION load_model(text) RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;

CREATE FUNCTION get_oid(integer) RETURNS integer
     AS '$libdir/pgdl'
     LANGUAGE C STRICT;