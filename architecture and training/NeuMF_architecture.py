# -*- coding: utf-8 -*-

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, BatchNormalization, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam

def NeuMF(num_users, num_items, gmf_embedding_dim, mlp_embedding_dim):
    # Define input vectors for embedding
    u_input = Input(shape = [1,])
    i_input = Input(shape = [1,])
    
    # GMF embedding
    u_embedding_gmf = Embedding(input_dim = num_users, output_dim = gmf_embedding_dim)(u_input)
    u_vec_gmf = Flatten()(u_embedding_gmf)
    
    i_embedding_gmf = Embedding(input_dim = num_items, output_dim = gmf_embedding_dim)(i_input)
    i_vec_gmf = Flatten()(i_embedding_gmf)
    
    # MLP embedding
    u_embedding_mlp = Embedding(input_dim = num_users, output_dim = mlp_embedding_dim)(u_input)
    u_vec_mlp = Flatten()(u_embedding_mlp)
    
    i_embedding_mlp = Embedding(input_dim = num_items, output_dim = mlp_embedding_dim)(i_input)
    i_vec_mlp = Flatten()(i_embedding_mlp)
    
    # GMF path
    gmf_output = Dot(axes = 1)([u_vec_gmf, i_vec_gmf])
    
    # MLP path
    mlp_input_concat = Concatenate()([u_vec_mlp, i_vec_mlp])
    
    mlp_dense_1 = Dense(units = 16, activation = "relu")(mlp_input_concat)
    mlp_bn_1 = BatchNormalization()(mlp_dense_1)
    #mlp_drop_2 = Dropout(0.2)(mlp_bn_1)
    
    mlp_dense_2 = Dense(units = 8, activation = "relu")(mlp_bn_1)
    mlp_output = BatchNormalization()(mlp_dense_2)
    #mlp_output = Dropout(0.2)(mlp_bn_2)

    # Concatenate GMF and MLP pathways
    paths_concat = Concatenate()([gmf_output, mlp_output])
    
    # Prediction
    output = Dense(units = 1, activation = "sigmoid")(paths_concat)
    
    # Create model 
    
    return Model(inputs = [u_input, i_input], outputs = output)
    